#!/bin/bash
#SBATCH --job-name=cnn3d_baseline
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=${PROJECT}/ablations/logs/cnn3d_%j.out
#SBATCH --error=${PROJECT}/ablations/logs/cnn3d_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail
cd ${PROJECT}
source ${VENV_PATH}/bin/activate
mkdir -p ablations/results/cnn3d

python baselines/cnn_3d.py \
    --tasks adhd_binary tumor_lat \
    --epochs 50 \
    --out-dir ablations/results/cnn3d
