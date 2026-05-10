#!/bin/bash
#SBATCH --job-name=adhd_validate
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=${PROJECT}/preprocessing/logs/adhd_validate_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/adhd_validate_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail
cd ${PROJECT}
source ${VENV_PATH}/bin/activate

echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo "Start: $(date)"

python data/validate_adhd_pipeline.py --n 30 --seed 42

echo "End: $(date)"
