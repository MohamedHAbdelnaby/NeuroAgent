#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail

cd "$PROJECT"

: "${SERVED_NAME:?SERVED_NAME required}"
: "${MODEL_PATH:?MODEL_PATH required (use baselines/recover_run.sh for ARC API models)}"
: "${MAX_SUBJECTS:=9999}"
: "${MAX_MODEL_LEN:=16384}"
: "${STUDY_NAME:=study_full_v2}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${PARALLEL_SUBJECTS:=4}"

echo "v2 corrected run: $SERVED_NAME -> $STUDY_NAME"
echo "  Job:          ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "  Model path:   ${MODEL_ROOT}/$MODEL_PATH"
echo "  Subjects:     $MAX_SUBJECTS"
echo "  Tasks:        $TASKS"
echo "  Parallel:     $PARALLEL_SUBJECTS"
echo "  Start:        $(date)"

source ${VENV_PATH}/bin/activate
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME}
export STUDY_NAME

NODE=$(hostname)
PORT=$((8000 + (${SLURM_JOB_ID:-0} % 999) + 1))
URL="http://${NODE}:${PORT}/v1"
echo "Using port $PORT (derived from job $SLURM_JOB_ID)"

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
    *phi-4*|*Phi-4*)  EXTRA_FLAGS="--trust-remote-code" ;;
    *gemma*)          EXTRA_FLAGS="--trust-remote-code" ;;
esac

echo ""
echo "Starting local vLLM ($MODEL_PATH)..."
TP=$(nvidia-smi -L | wc -l)
echo "  TP=$TP  TOOL_FLAGS=$TOOL_FLAGS  EXTRA_FLAGS=$EXTRA_FLAGS  MAX_MODEL_LEN=$MAX_MODEL_LEN"
VLLM_LOG="$PROJECT/baselines/logs/vllm_${SERVED_NAME}_${SLURM_JOB_ID:-local}.log"
vllm serve "${MODEL_ROOT}/$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port $PORT \
    --api-key local-vllm-key \
    --tensor-parallel-size $TP \
    --dtype bfloat16 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.85 \
    $TOOL_FLAGS \
    $EXTRA_FLAGS \
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
        echo "ERROR: vLLM died. Last 200 lines of $VLLM_LOG:"
        tail -200 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
    [ $((i % 12)) -eq 0 ] && echo "  ...still waiting ($((i*5))s)"
done

mkdir -p "$PROJECT/baselines/vllm_endpoints"
cat > "$PROJECT/baselines/vllm_endpoints/${SERVED_NAME}.json" <<EOF
{"served_name": "$SERVED_NAME", "url": "$URL", "api_key": "local-vllm-key", "node": "$NODE", "port": $PORT, "job_id": "${SLURM_JOB_ID:-local}"}
EOF

: "${MODES:=neuroagent neuroagent_llm direct_matched direct}"

i=0
n_modes=$(echo "$MODES" | wc -w)
for mode in $MODES; do
    i=$((i + 1))
    echo ""
    echo "[$i/$n_modes]  MODE=$mode"
    parallel_n="$PARALLEL_SUBJECTS"
    case "$mode" in
        direct|direct_matched|rule_only) parallel_n=1 ;;
    esac
    python eval/comprehensive_study.py run \
        --models "$SERVED_NAME" \
        --tasks $TASKS \
        --modes $mode \
        --max-subjects "$MAX_SUBJECTS" \
        --sleep 0.5 \
        --parallel "$parallel_n" \
        ${BALANCED:+--balanced} \
        --resume
done

echo ""
echo "DONE  $SERVED_NAME at $(date)"
