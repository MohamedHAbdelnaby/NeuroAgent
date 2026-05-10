#!/bin/bash


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail
cd "$PROJECT"

: "${ABLATIONS:=baseline no_structural no_functional no_rag no_atlas no_struct_no_func}"
: "${MODELS:=Llama-3.1-8B-Instruct Qwen3-14B}"
: "${MAX_SUBJECTS:=9999}"
: "${PARALLEL_SUBJECTS:=4}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${WALLTIME:=12:00:00}"
: "${QOS:=fal_a30_normal_base}"

declare -A MODEL_PATH=(
    ["Llama-3.1-8B-Instruct"]="meta-llama--Llama-3.1-8B-Instruct"
    ["Llama-3.2-3B-Instruct"]="meta-llama--Llama-3.2-3B-Instruct"
    ["Qwen3-8B"]="qwen--Qwen3-8B"
    ["Qwen3-14B"]="qwen--Qwen3-14B"
    ["Qwen3-32B"]="qwen--Qwen3-32B"
    ["Phi-4"]="microsoft--phi-4"
    ["gpt-oss-20b"]="openai--gpt-oss-20b"
    ["Mistral-7B-v0.3"]="mistralai--Mistral-7B-Instruct-v0.3"
)
declare -A MODEL_GPU=(
    ["Llama-3.1-8B-Instruct"]=1
    ["Llama-3.2-3B-Instruct"]=1
    ["Qwen3-8B"]=1
    ["Qwen3-14B"]=2
    ["Qwen3-32B"]=4
    ["Phi-4"]=2
    ["gpt-oss-20b"]=1
    ["Mistral-7B-v0.3"]=1
)

mkdir -p "$PROJECT/ablations/logs"

for ablation in $ABLATIONS; do
  for model in $MODELS; do
    p="${MODEL_PATH[$model]:-}"; g="${MODEL_GPU[$model]:-1}"
    if [ -z "$p" ]; then echo "  SKIP $model (no path)"; continue; fi
    cpus=$((g * 8)); mem=$((g * 64))G
    sbatch \
      --job-name="ab_${ablation}_${model}" \
      --partition=a30_normal_q --qos="$QOS" --account=aml \
      --nodes=1 --ntasks=1 --cpus-per-task=$cpus --mem=$mem \
      --gres=gpu:$g --time="$WALLTIME" \
      --output="$PROJECT/ablations/logs/${ablation}_${model}_%j.out" \
      --error="$PROJECT/ablations/logs/${ablation}_${model}_%j.err" \
      --export=ALL,SERVED_NAME="$model",MODEL_PATH="$p",ABLATION="$ablation",MAX_SUBJECTS="$MAX_SUBJECTS",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS",TASKS="$TASKS" \
      "$PROJECT/ablations/run_one_ablation.sh"
    echo "Submitted: ${ablation} x ${model}"
  done
done
echo "Done. squeue -u \$USER | grep ab_"
