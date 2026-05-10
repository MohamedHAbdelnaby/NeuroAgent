#!/bin/bash

#SBATCH --job-name=neuro_qdir
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/quick_direct_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/quick_direct_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail
cd ${PROJECT}
mkdir -p baselines/logs

echo "QUICK direct study (100 subjects/task)"
echo "Output: eval/results/study_quick/"

export STUDY_NAME="study_quick"
export MODES="direct"
export MAX_SUBJECTS=100
export TASKS="adhd_binary tumor_grade stroke_lat"
export MODELS="Mistral-7B-v0.3 Llama-3.1-8B-Instruct Gemma-3-12b-it Qwen3-14B gpt-oss-120b Kimi-K2.6 MiniMax-M2.7"
export ARC_MODELS="gpt-oss-120b Kimi-K2.6 MiniMax-M2.7"
export SLEEP=4
export SKIP_PLOTS=1

bash baselines/run_vllm_study.sh
echo "Quick direct done at $(date)"
