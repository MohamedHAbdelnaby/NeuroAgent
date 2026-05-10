#!/bin/bash

#SBATCH --job-name=neuro_recover
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/recover_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/recover_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail

cd "$PROJECT"

: "${SERVED_NAME:?SERVED_NAME required}"
: "${MODE:?MODE required (direct|neuroagent)}"
: "${MODEL_PATH:=}"
: "${MAX_SUBJECTS:=9999}"
: "${MAX_MODEL_LEN:=16384}"
: "${STUDY_NAME:=study_full}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"

echo "Recovery: $SERVED_NAME  mode=$MODE  study=$STUDY_NAME"
echo "  Job:          $SLURM_JOB_ID  on $(hostname)"
echo "  Subjects:     $MAX_SUBJECTS"
echo "  Tasks:        $TASKS"
echo "  Start:        $(date)"

source ${VENV_PATH}/bin/activate
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME}
export STUDY_NAME

if [ -n "$MODEL_PATH" ]; then
    NODE=$(hostname)
    PORT=8001
    URL="http://${NODE}:${PORT}/v1"

    TOOL_FLAGS=""
    case "$MODEL_PATH" in
        *Llama-3*|*llama-3*) TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser llama3_json" ;;
        *Qwen*|*qwen*)        TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes" ;;
        *Mistral*|*mistral*)  TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser mistral" ;;
    esac

    echo ""
    echo "Starting local vLLM ($MODEL_PATH)..."
    TP=$(nvidia-smi -L | wc -l)
    vllm serve "${MODEL_ROOT}/$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --host 0.0.0.0 --port $PORT \
        --api-key local-vllm-key \
        --tensor-parallel-size $TP \
        --dtype bfloat16 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.85 \
        $TOOL_FLAGS \
        > /tmp/vllm_${SLURM_JOB_ID}.log 2>&1 &
    VLLM_PID=$!
    trap "echo Cleaning up; kill $VLLM_PID 2>/dev/null || true" EXIT

    echo "Waiting for vLLM at $URL..."
    for i in $(seq 1 360); do
        if curl -s -m 5 -H "Authorization: Bearer local-vllm-key" \
             "$URL/models" 2>/dev/null | grep -q "object.*list"; then
            echo "vLLM ready ($((i*5))s)"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM died. Last 40 lines:"
            tail -40 /tmp/vllm_${SLURM_JOB_ID}.log
            exit 1
        fi
        sleep 5
        [ $((i % 12)) -eq 0 ] && echo "  ...still waiting ($((i*5))s)"
    done

    mkdir -p "$PROJECT/baselines/vllm_endpoints"
    cat > "$PROJECT/baselines/vllm_endpoints/${SERVED_NAME}.json" <<EOF
{"served_name": "$SERVED_NAME", "url": "$URL", "api_key": "local-vllm-key", "node": "$NODE", "port": $PORT, "job_id": "$SLURM_JOB_ID"}
EOF
else
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY before running}"
    : "${OPENAI_BASE_URL:=https://llm-api.arc.vt.edu/api/v1}"
    export OPENAI_API_KEY OPENAI_BASE_URL
    rm -f "$PROJECT/baselines/vllm_endpoints/${SERVED_NAME}.json"
fi

SLEEP=4
[ -n "$MODEL_PATH" ] && SLEEP=0.5

PARALLEL=1
if [ "$MODE" = "neuroagent" ] && [ -n "$MODEL_PATH" ]; then
    PARALLEL="${PARALLEL_SUBJECTS:-4}"
fi

echo ""
echo "Running comprehensive_study.py ($MODE, $SERVED_NAME, parallel=$PARALLEL)..."
python eval/comprehensive_study.py run \
    --models "$SERVED_NAME" \
    --tasks $TASKS \
    --modes "$MODE" \
    --max-subjects "$MAX_SUBJECTS" \
    --sleep "$SLEEP" \
    --parallel "$PARALLEL" \
    --resume

echo ""
echo "Done at $(date)"
