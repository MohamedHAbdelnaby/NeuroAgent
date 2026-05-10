#!/bin/bash

#SBATCH --job-name=neuro_qNA
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --account=aml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=${PROJECT}/baselines/logs/quick_neuroagent_%j.out
#SBATCH --error=${PROJECT}/baselines/logs/quick_neuroagent_%j.err


PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"

set -euo pipefail
cd ${PROJECT}
mkdir -p baselines/logs

echo "QUICK NeuroAgent study (100 subjects/task, vLLM models only)"
echo "Output: eval/results/study_quick/"

export STUDY_NAME="study_quick"
export MODES="neuroagent"
export MAX_SUBJECTS=100
export TASKS="adhd_binary tumor_grade stroke_lat"
export MODELS="Llama-3.1-8B-Instruct Qwen3-14B Mistral-7B-v0.3"
export ARC_MODELS=""

bash baselines/run_vllm_study.sh
echo "Quick NeuroAgent done at $(date)"
