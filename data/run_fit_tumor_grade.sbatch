#!/bin/bash
#SBATCH --job-name=fit_tumor_grade
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --output=${PROJECT}/preprocessing/logs/fit_tumor_grade_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fit_tumor_grade_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail
cd ${PROJECT}
source ${VENV_PATH}/bin/activate
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
python data/fit_tumor_grade_classifier.py
echo "End: $(date)"
