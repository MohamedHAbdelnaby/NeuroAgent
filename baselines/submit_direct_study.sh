#!/bin/bash

#SBATCH --job-name=neuro_fdir
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/direct_study_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/direct_study_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail

mkdir -p "$PROJECT/baselines/logs"

echo "FULL Direct study (all subjects, eval/results/study_full/)"
echo "  Job:          $SLURM_JOB_ID  on $(hostname)"
echo "  Start:        $(date)"

export STUDY_NAME="study_full"
export MODES="direct"
export MAX_SUBJECTS="${MAX_SUBJECTS:-9999}"
export TASKS="adhd_binary tumor_grade stroke_lat"
export MODELS="Mistral-7B-v0.3 Llama-3.1-8B-Instruct Gemma-3-12b-it Qwen3-14B gpt-oss-120b Kimi-K2.6 MiniMax-M2.7"
export ARC_MODELS="gpt-oss-120b Kimi-K2.6 MiniMax-M2.7"
export SLEEP="${SLEEP:-4}"
export SKIP_PLOTS=1

cd "$PROJECT"
bash baselines/run_vllm_study.sh

echo ""
echo "Direct mode done at $(date)"
