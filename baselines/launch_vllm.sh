#!/bin/bash

#SBATCH --job-name=vllm
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=${PROJECT}/baselines/logs/vllm_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/vllm_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH required (e.g. MODEL_PATH=qwen--Qwen3-14B)}"
GPUS="${GPUS:-1}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-$(basename $MODEL_PATH)}"
API_KEY="local-vllm-key"

ENDPOINTS_DIR="$PROJECT/baselines/vllm_endpoints"
mkdir -p "$ENDPOINTS_DIR"

MODEL_FULL_PATH="${MODEL_ROOT}/${MODEL_PATH}"
if [ ! -d "$MODEL_FULL_PATH" ]; then
    echo "ERROR: model path not found: $MODEL_FULL_PATH"
    ls ${MODEL_ROOT}/ | grep -i "$(echo $MODEL_PATH | tr '_' '-' | head -c 10)" || true
    exit 1
fi

NODE=$(hostname)
URL="http://${NODE}:${PORT}/v1"
ENDPOINT_FILE="$ENDPOINTS_DIR/${SERVED_NAME}.json"

cat > "$ENDPOINT_FILE" <<EOF
{
  "served_name": "$SERVED_NAME",
  "model_path":  "$MODEL_FULL_PATH",
  "node":        "$NODE",
  "port":        $PORT,
  "url":         "$URL",
  "api_key":     "$API_KEY",
  "job_id":      "$SLURM_JOB_ID",
  "gpus":        $GPUS,
  "started":     "$(date -Iseconds)"
}
EOF

echo "vLLM launcher"
echo "  Model:        $MODEL_PATH"
echo "  Served as:    $SERVED_NAME"
echo "  Node:         $NODE"
echo "  Port:         $PORT"
echo "  URL:          $URL"
echo "  GPUs:         $GPUS"
echo "  Endpoint:     $ENDPOINT_FILE"

source ${VENV_PATH}/bin/activate
module load cuda12.6/toolkit/12.6.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME}

python -c "import vllm" 2>/dev/null || {
    echo "ERROR: vllm not installed in venv. Run from login node (with cuda12.6 module loaded):"
    echo "  pip install vllm"
    exit 1
}

TP_SIZE=$(nvidia-smi -L | wc -l)
echo "  Detected $TP_SIZE GPUs, using tensor-parallel-size=$TP_SIZE"

TOOL_FLAGS=""
case "$MODEL_PATH" in
    *Llama-3*|*llama-3*)
        TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser llama3_json"
        ;;
    *Qwen*|*qwen*)
        TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes"
        ;;
    *Mistral*|*mistral*)
        TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser mistral"
        ;;
    *)
        echo "  No tool parser configured for $MODEL_PATH, tool calling will fail"
        ;;
esac
echo "  Tool flags: $TOOL_FLAGS"

exec vllm serve "$MODEL_FULL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    $TOOL_FLAGS
