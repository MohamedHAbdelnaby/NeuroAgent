#!/bin/bash


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail

cd "$PROJECT"

: "${SERVED_NAME:?SERVED_NAME required}"
: "${MODEL_PATH:?MODEL_PATH required}"
: "${ABLATION:?ABLATION required}"
: "${MAX_SUBJECTS:=9999}"
: "${MAX_MODEL_LEN:=16384}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${PARALLEL_SUBJECTS:=4}"

echo "Ablation: $ABLATION  Model: $SERVED_NAME"
echo "Tasks:    $TASKS"
echo "Job:      ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "Start:    $(date)"

source ${VENV_PATH}/bin/activate
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME}

NODE=$(hostname)
PORT=$((8000 + (${SLURM_JOB_ID:-0} % 999) + 1))
URL="http://${NODE}:${PORT}/v1"

TOOL_FLAGS=""
case "$MODEL_PATH" in
    *Llama-3*|*llama-3*)   TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser llama3_json" ;;
    *Qwen3*|*qwen3*|*Qwen2*|*qwen2*) TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes" ;;
    *Mistral*|*mistral*|*Ministral*|*ministral*) TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser mistral" ;;
    *gpt-oss*)             TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser openai" ;;
    *phi-4*|*Phi-4*)       TOOL_FLAGS="" ;;
    *gemma*)               TOOL_FLAGS="" ;;
esac
EXTRA_FLAGS=""
case "$MODEL_PATH" in
    *phi-4*|*Phi-4*) EXTRA_FLAGS="--trust-remote-code" ;;
    *gemma*)         EXTRA_FLAGS="--trust-remote-code" ;;
esac

TP=$(nvidia-smi -L | wc -l)
VLLM_LOG="$PROJECT/ablations/logs/vllm_${SERVED_NAME}_${ABLATION}_${SLURM_JOB_ID:-local}.log"
vllm serve "${MODEL_ROOT}/$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port $PORT \
    --api-key local-vllm-key \
    --tensor-parallel-size $TP \
    --dtype bfloat16 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.85 \
    $TOOL_FLAGS $EXTRA_FLAGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "echo Cleaning up vLLM; kill $VLLM_PID 2>/dev/null || true" EXIT

echo "Waiting for vLLM at $URL..."
for i in $(seq 1 360); do
    if curl -s -m 5 -H "Authorization: Bearer local-vllm-key" \
         "$URL/models" 2>/dev/null | grep -q "object.*list"; then
        echo "vLLM ready ($((i*5))s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died. Last 200 lines of $VLLM_LOG:"; tail -200 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
    [ $((i % 12)) -eq 0 ] && echo "  ...still waiting ($((i*5))s)"
done

mkdir -p "$PROJECT/baselines/vllm_endpoints"
cat > "$PROJECT/baselines/vllm_endpoints/${SERVED_NAME}.json" <<EOF
{"served_name": "$SERVED_NAME", "url": "$URL", "api_key": "local-vllm-key", "node": "$NODE", "port": $PORT, "job_id": "${SLURM_JOB_ID:-local}"}
EOF

python ablations/scripts/ablate_agents.py \
    --ablation "$ABLATION" \
    --model "$SERVED_NAME" \
    --tasks $TASKS \
    --max-subjects "$MAX_SUBJECTS" \
    --parallel "$PARALLEL_SUBJECTS" \
    --mode neuroagent \
    --resume

echo ""
echo "DONE  $ABLATION x $SERVED_NAME at $(date)"
