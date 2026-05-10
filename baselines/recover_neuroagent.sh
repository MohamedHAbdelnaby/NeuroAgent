#!/bin/bash

#SBATCH --job-name=neuro_NA_recover
#SBATCH --partition=v100_normal_q
#SBATCH --qos=fal_v100_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/na_recover_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/na_recover_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail

cd "$PROJECT"

: "${SERVED_NAME:=Llama-3.1-8B-Instruct}"
: "${MODEL_PATH:=meta-llama--Llama-3.1-8B-Instruct}"
: "${MAX_SUBJECTS:=9999}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${MAX_MODEL_LEN:=8192}"

NODE=$(hostname)
PORT=8000
URL="http://${NODE}:${PORT}/v1"
ENDPOINT_FILE="$PROJECT/baselines/vllm_endpoints/${SERVED_NAME}.json"

echo "NeuroAgent recovery: $SERVED_NAME"
echo "  Job:        $SLURM_JOB_ID  on $NODE"
echo "  Model:      ${MODEL_ROOT}/$MODEL_PATH"
echo "  Subjects:   $MAX_SUBJECTS per task"
echo "  Tasks:      $TASKS"
echo "  Max len:    $MAX_MODEL_LEN"
echo "  GPUs:       $(nvidia-smi -L | wc -l)"
echo "  Start:      $(date)"

source ${VENV_PATH}/bin/activate
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME}

TOOL_FLAGS=""
case "$MODEL_PATH" in
    *Llama-3*|*llama-3*) TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser llama3_json" ;;
    *Qwen*|*qwen*)        TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes" ;;
    *Mistral*|*mistral*)  TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser mistral" ;;
esac
echo "  Tool flags: $TOOL_FLAGS"

TP_SIZE=$(nvidia-smi -L | wc -l)
echo ""
echo "Starting vLLM (TP=$TP_SIZE)..."
vllm serve "${MODEL_ROOT}/$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --api-key local-vllm-key \
    --tensor-parallel-size $TP_SIZE \
    --dtype bfloat16 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.90 \
    $TOOL_FLAGS \
    > /tmp/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up, killing vLLM (PID $VLLM_PID)"
    kill $VLLM_PID 2>/dev/null || true
    rm -f "$ENDPOINT_FILE"
}
trap cleanup EXIT

echo "Waiting for vLLM to come up..."
for i in $(seq 1 360); do
    if curl -s -m 5 -H "Authorization: Bearer local-vllm-key" \
         "$URL/models" 2>/dev/null | grep -q "object.*list"; then
        echo "vLLM ready at $URL ($((i*5))s elapsed)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Last 30 lines of log:"
        tail -30 /tmp/vllm_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
    [ $((i % 12)) -eq 0 ] && echo "  ...still waiting ($((i*5))s)"
done

mkdir -p "$(dirname "$ENDPOINT_FILE")"
cat > "$ENDPOINT_FILE" <<EOF
{"served_name": "$SERVED_NAME", "model_path": "${MODEL_ROOT}/$MODEL_PATH",
 "node": "$NODE", "port": $PORT, "url": "$URL", "api_key": "local-vllm-key",
 "job_id": "$SLURM_JOB_ID"}
EOF
echo "Endpoint file written: $ENDPOINT_FILE"

export STUDY_NAME="study_full"
echo ""
echo "Running NeuroAgent eval for $SERVED_NAME..."
python eval/comprehensive_study.py run \
    --models "$SERVED_NAME" \
    --tasks $TASKS \
    --modes neuroagent \
    --max-subjects "$MAX_SUBJECTS" \
    --sleep 0.5 \
    --resume || echo "Study had errors but continuing"

echo ""
echo "Done at $(date)"
