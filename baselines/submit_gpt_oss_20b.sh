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

sbatch \
    --job-name="bal200_gpt-oss-20b" \
    --partition=a30_normal_q \
    --qos="$QOS" \
    --account=aml \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
    --gres=gpu:2 \
    --time="6:00:00" \
    --output="$PROJECT/baselines/logs/bal200_gpt-oss-20b_%j.out" \
    --error="$PROJECT/baselines/logs/bal200_gpt-oss-20b_%j.err" \
    --export=ALL,SERVED_NAME="gpt-oss-20b",MODEL_PATH="openai--gpt-oss-20b",MAX_SUBJECTS="$MAX_SUBJECTS",STUDY_NAME="$STUDY_NAME",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS",TASKS="$TASKS",BALANCED="$BALANCED",MAX_MODEL_LEN=16384 \
    "$PROJECT/baselines/recover_run_v2.sh"

echo "Submitted gpt-oss-20b balanced 200 job"
squeue -u $USER --format="%.10i %.30j %.10T %.10M %.10l %R"
