#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail
cd "$PROJECT"

: "${STUDY_NAME:=study_full_v2}"
: "${QOS:=fal_a30_normal_short}"
: "${PARALLEL_SUBJECTS:=4}"

submit_job () {
    local served_name="$1"
    local model_path="$2"
    sbatch \
        --job-name="full3_${served_name}" \
        --partition=a30_normal_q \
        --qos="$QOS" \
        --account=aml \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --gres=gpu:1 \
        --time="20:00:00" \
        --output="$PROJECT/baselines/logs/full3_${served_name}_%j.out" \
        --error="$PROJECT/baselines/logs/full3_${served_name}_%j.err" \
        --export=ALL,SERVED_NAME="$served_name",MODEL_PATH="$model_path",MODE=neuroagent,TASKS="adhd_binary tumor_grade stroke_lat",MAX_SUBJECTS=9999,STUDY_NAME="$STUDY_NAME",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS" \
        "$PROJECT/baselines/recover_run.sh"
}

echo "Submitting full 3-task neuroagent re-runs to $STUDY_NAME..."
submit_job "Llama-3.1-8B-Instruct" "meta-llama--Llama-3.1-8B-Instruct"
submit_job "Mistral-7B-v0.3"       "mistralai--Mistral-7B-Instruct-v0.3"
echo "Done. squeue -u \$USER"
