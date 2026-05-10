#!/bin/bash
#SBATCH --job-name=fmri_norms
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=${PROJECT}/preprocessing/logs/fmri_norms_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fmri_norms_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail

cd ${PROJECT}

source ${VENV_PATH}/bin/activate

echo "Host:  $(hostname)"
echo "CPUs:  $SLURM_CPUS_PER_TASK"
echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date)"

export N_WORKERS=$SLURM_CPUS_PER_TASK

python data/compute_adhd200_fmri_norms.py

echo "End:   $(date)"
