#!/bin/bash

#SBATCH --job-name=neuroagent_baseline
#SBATCH --array=0-1
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=${PROJECT}/baselines/logs/%x_%A_%a.out
#SBATCH --error=${PROJECT}/baselines/logs/%x_%A_%a.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"

set -euo pipefail

PROJECT_ROOT="${PROJECT}"
VENV="${VENV_PATH}"

CONDITIONS=(adhd tumor)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "Job:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Condition: $CONDITION"
echo "Start:     $(date)"

source "$VENV/bin/activate"

export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY before running}"
export OPENAI_BASE_URL="https://llm-api.arc.vt.edu/api/v1"

cd "$PROJECT_ROOT"

mkdir -p baselines/logs baselines/results eval/results eval/ground_truth

echo ""
echo "[Step 1/3] Extracting ground-truth labels for $CONDITION..."
python eval/extract_ground_truth.py --condition "$CONDITION"

echo ""
echo "[Step 2/3] Training 3D CNN baseline for $CONDITION..."
python baselines/cnn_3d.py \
    --condition "$CONDITION" \
    --epochs 30 \
    --batch-size 4 \
    --lr 1e-4 \
    --out-dir baselines/results

echo ""
echo "[Step 3/3] Running NeuroAgent evaluation for $CONDITION..."
python eval/evaluate.py \
    --condition "$CONDITION" \
    --max-subjects 100 \
    --out-dir eval/results

echo ""
echo "Done: $(date)"
echo "Results in: baselines/results/ and eval/results/"
