#!/bin/bash
#SBATCH --job-name=rule_only
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=${PROJECT}/baselines/logs/rule_only_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/rule_only_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail
cd ${PROJECT}
source ${VENV_PATH}/bin/activate

echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"

export STUDY_NAME=study_publication

python eval/comprehensive_study.py run \
    --models RULE_ONLY \
    --tasks adhd_binary tumor_grade stroke_lat \
    --modes rule_only \
    --max-subjects 9999 \
    --parallel 1 \
    --resume

echo "End: $(date)"
