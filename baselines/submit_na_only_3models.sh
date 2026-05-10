#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail
cd "$PROJECT"

: "${STUDY_NAME:=study_balanced_200}"
: "${QOS:=fal_a30_normal_short}"
: "${PARALLEL_SUBJECTS:=4}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${MAX_SUBJECTS:=200}"
: "${BALANCED:=1}"

submit_job () {
    local served_name="$1"
    local model_path="$2"
    local gpus="${3:-1}"
    sbatch \
        --job-name="naX_${served_name}" \
        --partition=a30_normal_q \
        --qos="$QOS" \
        --account=aml \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --gres=gpu:${gpus} \
        --time="3:00:00" \
        --output="$PROJECT/baselines/logs/naX_${served_name}_%j.out" \
        --error="$PROJECT/baselines/logs/naX_${served_name}_%j.err" \
        --export=ALL,SERVED_NAME="$served_name",MODEL_PATH="$model_path",MODE=neuroagent,MAX_SUBJECTS="$MAX_SUBJECTS",STUDY_NAME="$STUDY_NAME",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS",TASKS="$TASKS",BALANCED="$BALANCED",MAX_MODEL_LEN=16384 \
        "$PROJECT/baselines/recover_run.sh"
}

echo "Submitting NA-only re-runs to $STUDY_NAME..."
submit_job "Llama-3.1-8B-Instruct" "meta-llama--Llama-3.1-8B-Instruct" 1
submit_job "Mistral-7B-v0.3"       "mistralai--Mistral-7B-Instruct-v0.3" 1
submit_job "gpt-oss-20b"           "openai--gpt-oss-20b" 2
echo "Done."
squeue -u $USER --format="%.10i %.30j %.10T %.10M %.10l %R"
