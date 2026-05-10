#!/bin/bash
#SBATCH --job-name=fastsurfer_batch
#SBATCH --partition=a100_normal_q
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${PROJECT}/preprocessing/logs/fastsurfer_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fastsurfer_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail

NEUROAGENT="$PROJECT"
FASTSURFER_DIR=$NEUROAGENT/FastSurfer
OUTPUT_DIR=$NEUROAGENT/preprocessing/fastsurfer_output
MANIFEST=$NEUROAGENT/preprocessing/manifests/all_subjects.csv
LOG_DIR=$NEUROAGENT/preprocessing/logs

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "FastSurfer SLURM Job"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

module reset
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

source ${VENV_PATH}/bin/activate
unset PYTHONPATH

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python "$NEUROAGENT/run_fastsurfer_batch.py" \
    --fastsurfer-dir "$FASTSURFER_DIR" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --threads 8 \
    --skip-existing \
    "$@"

echo ""
echo "Done : $(date)"
