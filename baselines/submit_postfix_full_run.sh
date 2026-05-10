#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail
cd "$PROJECT"

: "${STUDY_NAME:=study_postfix}"
: "${QOS:=fal_a30_normal_base}"
: "${PARALLEL_SUBJECTS:=4}"
: "${TASKS:=adhd_binary tumor_grade stroke_lat}"
: "${MAX_SUBJECTS:=9999}"
: "${BALANCED:=}"
: "${WALLTIME:=20:00:00}"

mkdir -p "$PROJECT/baselines/logs"

submit_job () {
    local served_name="$1"
    local model_path="$2"
    sbatch \
        --job-name="postfix_${served_name}" \
        --partition=a30_normal_q \
        --qos="$QOS" \
        --account=aml \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --gres=gpu:1 \
        --time="$WALLTIME" \
        --output="$PROJECT/baselines/logs/postfix_${served_name}_%j.out" \
        --error="$PROJECT/baselines/logs/postfix_${served_name}_%j.err" \
        --export=ALL,SERVED_NAME="$served_name",MODEL_PATH="$model_path",MAX_SUBJECTS="$MAX_SUBJECTS",STUDY_NAME="$STUDY_NAME",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS",TASKS="$TASKS",BALANCED="$BALANCED" \
        "$PROJECT/baselines/recover_run_v2.sh"
}

echo "Post-fix full rerun -> $STUDY_NAME"
echo "  Tasks:        $TASKS"
echo "  Max subjects: $MAX_SUBJECTS  (per task per model)"
echo "  Balanced:     ${BALANCED:-no}"
echo "  Walltime:     $WALLTIME"
echo "  Parallel:     $PARALLEL_SUBJECTS"

submit_job "Llama-3.1-8B-Instruct" "meta-llama--Llama-3.1-8B-Instruct"
submit_job "Mistral-7B-v0.3"       "mistralai--Mistral-7B-Instruct-v0.3"

echo ""
echo "Done submitting. Watch with: squeue -u \$USER"
