#!/bin/bash

#SBATCH --job-name=neuro_NA
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/neuroagent_study_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/neuroagent_study_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail

mkdir -p "$PROJECT/baselines/logs"

echo "FULL NeuroAgent study (eval/results/study_full/)"
echo "  Job:          $SLURM_JOB_ID  on $(hostname)"
echo "  Start:        $(date)"

export STUDY_NAME="study_full"
export MODES="neuroagent"
export MAX_SUBJECTS="${MAX_SUBJECTS:-9999}"
export TASKS="adhd_binary tumor_grade stroke_lat"
export MODELS="Llama-3.1-8B-Instruct Qwen3-14B Mistral-7B-v0.3"
export ARC_MODELS=""

cd "$PROJECT"
bash baselines/run_vllm_study.sh

echo ""
echo "NeuroAgent mode done at $(date)"
