#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail
cd "$PROJECT"

: "${STUDY_NAME:=study_postfix2}"
: "${QOS:=fal_a30_normal_base}"
: "${PARALLEL_SUBJECTS:=4}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${MAX_SUBJECTS:=9999}"
: "${WALLTIME:=20:00:00}"

mkdir -p "$PROJECT/baselines/logs"

declare -A MODEL_PATH=(
    ["Llama-3.1-8B-Instruct"]="meta-llama--Llama-3.1-8B-Instruct"
    ["Llama-3.2-3B-Instruct"]="meta-llama--Llama-3.2-3B-Instruct"
    ["Mistral-7B-v0.3"]="mistralai--Mistral-7B-Instruct-v0.3"
    ["Qwen3-8B"]="qwen--Qwen3-8B"
    ["Qwen3-14B"]="qwen--Qwen3-14B"
    ["Phi-4"]="microsoft--phi-4"
    ["gpt-oss-20b"]="openai--gpt-oss-20b"
    ["Qwen3-32B"]="qwen--Qwen3-32B"
)
declare -A MODEL_GPU=(
    ["Llama-3.1-8B-Instruct"]=1
    ["Llama-3.2-3B-Instruct"]=1
    ["Mistral-7B-v0.3"]=1
    ["Qwen3-8B"]=1
    ["Qwen3-14B"]=2
    ["Phi-4"]=2
    ["gpt-oss-20b"]=1
    ["Qwen3-32B"]=4
)

submit_job () {
    local served_name="$1"
    local model_path="$2"
    local n_gpu="$3"
    local cpus=$((n_gpu * 8))
    local mem=$((n_gpu * 64))G
    sbatch \
        --job-name="postfix2_${served_name}" \
        --partition=a30_normal_q \
        --qos="$QOS" \
        --account=aml \
        --nodes=1 --ntasks=1 --cpus-per-task=$cpus --mem=$mem \
        --gres=gpu:$n_gpu \
        --time="$WALLTIME" \
        --output="$PROJECT/baselines/logs/postfix2_${served_name}_%j.out" \
        --error="$PROJECT/baselines/logs/postfix2_${served_name}_%j.err" \
        --export=ALL,SERVED_NAME="$served_name",MODEL_PATH="$model_path",MAX_SUBJECTS="$MAX_SUBJECTS",STUDY_NAME="$STUDY_NAME",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS",TASKS="$TASKS",BALANCED= \
        "$PROJECT/baselines/recover_run_v2.sh"
}

: "${MODELS:=Llama-3.2-3B-Instruct Qwen3-14B Phi-4 gpt-oss-20b}"

echo "Multi-model post-fix run -> $STUDY_NAME"
echo "  Tasks:        $TASKS"
echo "  Models:       $MODELS"
echo "  Max subjects: $MAX_SUBJECTS  (per task per model)"
echo "  Walltime:     $WALLTIME each"

for m in $MODELS; do
    p="${MODEL_PATH[$m]:-}"
    g="${MODEL_GPU[$m]:-1}"
    if [ -z "$p" ]; then
        echo "  SKIP $m  (no MODEL_PATH mapping; add to script)"; continue
    fi
    if [ ! -d "${MODEL_ROOT}/$p" ]; then
        echo "  SKIP $m  (model dir not found at ${MODEL_ROOT}/$p)"; continue
    fi
    submit_job "$m" "$p" "$g"
done

echo ""
echo "Done submitting. Watch: squeue -u \$USER | grep postfix2"
