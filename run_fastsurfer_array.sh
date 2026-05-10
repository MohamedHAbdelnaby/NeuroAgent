#!/bin/bash

#SBATCH --job-name=fastsurfer_array
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-28%8
#SBATCH --output=${PROJECT}/preprocessing/logs/fastsurfer_arr_%A_%a.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fastsurfer_arr_%A_%a.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail

NEUROAGENT="$PROJECT"
FASTSURFER_DIR=$NEUROAGENT/FastSurfer
OUTPUT_DIR=$NEUROAGENT/preprocessing/fastsurfer_output
MANIFEST=$NEUROAGENT/preprocessing/manifests/all_subjects.csv
LOG_DIR=$NEUROAGENT/preprocessing/logs

CHUNK_SIZE=150
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "FastSurfer Array Job"
echo "Array job  : $SLURM_ARRAY_JOB_ID"
echo "Task ID    : $SLURM_ARRAY_TASK_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start idx  : $START_IDX  (chunk size $CHUNK_SIZE)"
echo "Start time : $(date)"

module reset
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

source ${VENV_PATH}/bin/activate
unset PYTHONPATH

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "PyTorch CUDA check:"
python3 - <<'PYCHECK'
import sys
try:
    import torch
    avail = torch.cuda.is_available()
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA available: {avail}")
    if avail:
        print(f"  CUDA device   : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version  : {torch.version.cuda}")
    else:
        print("  ERROR: torch.cuda.is_available() returned False!")
        print("  FastSurfer will fail silently. Fix the conda environment.")
        sys.exit(1)
except ImportError as e:
    print(f"  ERROR: could not import torch: {e}")
    sys.exit(1)
PYCHECK

echo ""

python "$NEUROAGENT/run_fastsurfer_batch.py" \
    --fastsurfer-dir "$FASTSURFER_DIR" \
    --manifest     "$MANIFEST" \
    --output-dir   "$OUTPUT_DIR" \
    --device       cuda \
    --threads      8 \
    --batch-size   8 \
    --skip-existing \
    --start-idx    "$START_IDX" \
    --chunk-size   "$CHUNK_SIZE" \
    --log-suffix   "task${SLURM_ARRAY_TASK_ID}" \
    "$@"

echo ""
echo "Done : $(date)"
