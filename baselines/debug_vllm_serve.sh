#!/bin/bash
#SBATCH --job-name=vllm_debug
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00
#SBATCH --output=${PROJECT}/baselines/logs/vllm_debug_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/vllm_debug_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set +e
cd "$PROJECT"
source ${VENV_PATH}/bin/activate
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME}

: "${MODEL_PATH:?MODEL_PATH required}"
: "${SERVED_NAME:?SERVED_NAME required}"
: "${MAX_MODEL_LEN:=8192}"

echo "Host: $(hostname)  GPUs: $(nvidia-smi -L)"
echo "Model: $MODEL_PATH"

TOOL_FLAGS=""
case "$MODEL_PATH" in
    *Llama-3*|*llama-3*)   TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser llama3_json" ;;
    *Qwen3*|*qwen3*|*Qwen2*|*qwen2*) TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes" ;;
    *gpt-oss*)             TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser openai" ;;
esac
EXTRA_FLAGS=""
case "$MODEL_PATH" in
    *phi-4*|*Phi-4*) EXTRA_FLAGS="--trust-remote-code" ;;
esac

timeout 600 vllm serve "${MODEL_ROOT}/$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8001 \
    --api-key local-vllm-key \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.85 \
    $TOOL_FLAGS $EXTRA_FLAGS 2>&1 | tee "$PROJECT/baselines/logs/vllm_debug_${SERVED_NAME}_${SLURM_JOB_ID}.log"

echo "Done"
