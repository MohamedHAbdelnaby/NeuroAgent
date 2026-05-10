#!/bin/bash

#SBATCH --job-name=fastsurfer_diag
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=${PROJECT}/preprocessing/logs/fastsurfer_diag_%j.out
#SBATCH --error=${PROJECT}/preprocessing/logs/fastsurfer_diag_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail

NEUROAGENT="$PROJECT"
module reset
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

source ${VENV_PATH}/bin/activate
unset PYTHONPATH

echo "FastSurfer Environment Diagnostic"
echo "Node    : $SLURMD_NODENAME"
echo "Start   : $(date)"

echo ""
echo "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "Python / PyTorch"
python3 - <<'PYCHECK'
import sys, torch
print(f"Python        : {sys.version.split()[0]}")
print(f"Torch version : {torch.__version__}")
avail = torch.cuda.is_available()
print(f"CUDA available: {avail}")
if avail:
    print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version  : {torch.version.cuda}")
    x = torch.zeros(1, device='cuda')
    print(f"Alloc test    : OK (tensor on {x.device})")
else:
    print("FAIL: CUDA not available, FastSurfer will silently produce no output.")
    sys.exit(1)
PYCHECK

echo ""
echo "Test run: 1 subject per dataset"

python "$NEUROAGENT/run_fastsurfer_batch.py" \
    --fastsurfer-dir "$NEUROAGENT/FastSurfer" \
    --manifest       "$NEUROAGENT/preprocessing/manifests/all_subjects.csv" \
    --output-dir     "$NEUROAGENT/preprocessing/fastsurfer_output" \
    --device         cuda \
    --threads        8 \
    --batch-size     1 \
    --log-suffix     "diag" \
    --dataset        ADHD-200 \
    --max-subjects   1

echo ""
python "$NEUROAGENT/run_fastsurfer_batch.py" \
    --fastsurfer-dir "$NEUROAGENT/FastSurfer" \
    --manifest       "$NEUROAGENT/preprocessing/manifests/all_subjects.csv" \
    --output-dir     "$NEUROAGENT/preprocessing/fastsurfer_output" \
    --device         cuda \
    --threads        8 \
    --batch-size     1 \
    --log-suffix     "diag" \
    --dataset        ATLAS-v2 \
    --max-subjects   1

echo ""
echo "Done : $(date)"
