#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail

cd "$PROJECT"

: "${STUDY_NAME:=study_full_v2}"
: "${QOS:=fal_a30_normal_short}"
: "${MAX_SUBJECTS:=9999}"
: "${PARALLEL_SUBJECTS:=4}"

mkdir -p baselines/logs "eval/results/$STUDY_NAME"

echo "Submitting corrected study to: eval/results/$STUDY_NAME"
echo " QOS:        $QOS"
echo " Subjects:   $MAX_SUBJECTS"
echo " Parallel:   $PARALLEL_SUBJECTS"

submit_job () {
    local served_name="$1"
    local model_path="$2"

    sbatch \
        --job-name="v2_${served_name}" \
        --partition=a30_normal_q \
        --qos="$QOS" \
        --account=aml \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --gres=gpu:1 \
        --time="1-00:00:00" \
        --output="$PROJECT/baselines/logs/v2_${served_name}_%j.out" \
        --error="$PROJECT/baselines/logs/v2_${served_name}_%j.err" \
        --export=ALL,SERVED_NAME="$served_name",MODEL_PATH="$model_path",MAX_SUBJECTS="$MAX_SUBJECTS",STUDY_NAME="$STUDY_NAME",PARALLEL_SUBJECTS="$PARALLEL_SUBJECTS" \
        "$PROJECT/baselines/recover_run_v2.sh"
}

echo ""
echo "[1/2] Llama-3.1-8B-Instruct"
submit_job "Llama-3.1-8B-Instruct" "meta-llama--Llama-3.1-8B-Instruct"

echo "[2/2] Mistral-7B-v0.3"
submit_job "Mistral-7B-v0.3" "mistralai--Mistral-7B-Instruct-v0.3"

echo ""
echo "2 jobs submitted. Status:"
echo "   squeue -u \$USER --format=\"%i %j %T %M %R\""
