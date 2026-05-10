#!/bin/bash

#SBATCH --job-name=neuroagent_study
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/study_orchestrator_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/study_orchestrator_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail

mkdir -p "$PROJECT/baselines/logs"

echo "NeuroAgent Full Comprehensive Study"
echo "  Orchestrator job: $SLURM_JOB_ID"
echo "  Node:             $(hostname)"
echo "  Start:            $(date)"
echo "  Max subjects:     ${MAX_SUBJECTS:-9999} (per task per model)"
echo "  Tasks:            ${TASKS:-adhd_binary tumor_lat}"

export MAX_SUBJECTS="${MAX_SUBJECTS:-9999}"
export TASKS="${TASKS:-adhd_binary tumor_lat}"

cd "$PROJECT"

bash baselines/run_vllm_study.sh

echo ""
echo "Done at $(date)"
echo ""
echo " Summary CSV:  $PROJECT/eval/results/study/summary.csv"
echo " Plots:        $PROJECT/eval/results/study/plots/*.png"
echo " HTML report:  $PROJECT/eval/results/study/REPORT.html"
echo " PDF report:   $PROJECT/eval/results/study/REPORT.pdf"
