#!/bin/bash
#SBATCH --job-name=adhd_loso_fit
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=${PROJECT}/preprocessing/logs/adhd_loso_fit_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/adhd_loso_fit_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail
cd ${PROJECT}
source ${VENV_PATH}/bin/activate

echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
export N_WORKERS=$SLURM_CPUS_PER_TASK

python data/fit_adhd_loso_classifier.py
echo "End: $(date)"
