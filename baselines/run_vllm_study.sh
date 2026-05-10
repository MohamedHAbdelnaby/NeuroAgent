#!/bin/bash

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PROJECT:?PROJECT could not be resolved}"
: "${VENV_PATH:?VENV_PATH must be set (path to your conda or virtualenv)}"
MODEL_ROOT="${MODEL_ROOT:-/common/data/models}"

set -euo pipefail

ENDPOINTS="$PROJECT/baselines/vllm_endpoints"
mkdir -p "$ENDPOINTS"

export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY before running}"
export OPENAI_BASE_URL="https://llm-api.arc.vt.edu/api/v1"

MAX_SUBJECTS="${MAX_SUBJECTS:-9999}"
TASKS="${TASKS:-adhd_binary tumor_grade stroke_lat}"
MODES="${MODES:-direct}"

declare -A MODEL_PATHS=(
    [Llama-3.1-8B-Instruct]="meta-llama--Llama-3.1-8B-Instruct"
    [Qwen3-14B]="qwen--Qwen3-14B"
    [Mistral-7B-v0.3]="mistralai--Mistral-7B-Instruct-v0.3"
    [gpt-oss-20b]="openai--gpt-oss-20b"
    [Gemma-3-12b-it]="google--gemma-3-12b-it"
)

DEFAULT_ARC_MODELS="gpt-oss-120b Kimi-K2.6 MiniMax-M2.7"
DEFAULT_VLLM_MODELS="Mistral-7B-v0.3 Llama-3.1-8B-Instruct Gemma-3-12b-it Qwen3-14B"
DEFAULT_MODELS="$DEFAULT_ARC_MODELS $DEFAULT_VLLM_MODELS"

MODELS="${MODELS:-$DEFAULT_MODELS}"
ARC_MODELS="${ARC_MODELS:-$DEFAULT_ARC_MODELS}"

source ${VENV_PATH}/bin/activate
cd "$PROJECT"

is_arc_model() {
    local m="$1"
    for a in $ARC_MODELS; do
        if [ "$a" = "$m" ]; then return 0; fi
    done
    return 1
}

run_one_model() {
    local served="$1"

    if is_arc_model "$served"; then
        echo "ARC API model: $served (modes=$MODES, sleep=${SLEEP:-4}s)"
        python eval/comprehensive_study.py run \
            --models "$served" \
            --tasks $TASKS \
            --modes $MODES \
            --max-subjects "$MAX_SUBJECTS" \
            --sleep "${SLEEP:-4}" \
            --resume || echo "  Errors but continuing"
        return 0
    fi

    local model_path="${MODEL_PATHS[$served]:-}"
    if [ -z "$model_path" ]; then
        echo "WARN: unknown model $served, skipping"
        return 0
    fi

    local endpoint_file="$ENDPOINTS/${served}.json"

    echo "vLLM model: $served  (path=$model_path, modes=$MODES)"

    local all_done=true
    local study_root="$PROJECT/eval/results/${STUDY_NAME:-study}"
    for task in $TASKS; do
        for mode in $MODES; do
            local folder="$served"
            if [ "$mode" = "neuroagent" ]; then folder="${served}__neuroagent"; fi
            if [ ! -f "${study_root}/${folder}/${task}_results.json" ]; then
                all_done=false; break 2
            fi
        done
    done
    if $all_done; then
        echo "  All tasks/modes already done for $served, skipping vLLM launch"
        return 0
    fi

    local URL=""
    if [ -f "$endpoint_file" ]; then
        URL=$(python -c "import json; print(json.load(open('$endpoint_file'))['url'])" 2>/dev/null || echo "")
        if [ -n "$URL" ] && curl -s -m 5 -H "Authorization: Bearer local-vllm-key" \
             "$URL/models" 2>/dev/null | grep -q "object.*list"; then
            echo "  Reusing existing vLLM endpoint at $URL"
            python eval/comprehensive_study.py run \
                --models "$served" --tasks $TASKS --modes $MODES \
                --max-subjects "$MAX_SUBJECTS" --resume || echo "  Errors continuing"
            return 0
        fi
    fi

    local URL=""
    local JOBID=""
    local attempt
    for attempt in 1 2; do
        rm -f "$endpoint_file"
        echo "  Attempt $attempt: submitting vLLM job for $served..."
        JOBID=$(sbatch --gres=gpu:1 \
                       --export=ALL,MODEL_PATH=$model_path,SERVED_NAME=$served,PORT=8000 \
                       --parsable baselines/launch_vllm.sh)
        echo "  SLURM job: $JOBID"

        echo "  Waiting for vLLM to come up (timeout 30 min)..."
        URL=""
        local i
        for i in $(seq 1 360); do
            if [ -f "$endpoint_file" ]; then
                URL=$(python -c "import json; print(json.load(open('$endpoint_file'))['url'])" 2>/dev/null || echo "")
                if [ -n "$URL" ] && curl -s -m 5 -H "Authorization: Bearer local-vllm-key" \
                     "$URL/models" 2>/dev/null | grep -q "object.*list"; then
                    echo "  vLLM ready at $URL"
                    break 2
                fi
            fi
            sleep 5
            if [ $((i % 12)) -eq 0 ]; then
                echo "    ...still waiting ($((i*5))s elapsed)"
            fi
        done

        echo "  Attempt $attempt timed out, cancelling job $JOBID"
        scancel $JOBID 2>/dev/null || true
        rm -f "$endpoint_file"
        URL=""
    done

    if [ -z "$URL" ]; then
        echo "  ERROR: vLLM did not come up after 2 attempts, skipping $served"
        return 0
    fi

    python eval/comprehensive_study.py run \
        --models "$served" \
        --tasks $TASKS \
        --modes $MODES \
        --max-subjects "$MAX_SUBJECTS" \
        --resume || echo "  Study had errors but continuing"

    echo "  Cancelling vLLM job $JOBID"
    scancel $JOBID 2>/dev/null || true
    rm -f "$endpoint_file"
    sleep 5
}

echo "Running models: $MODELS"
echo "Tasks:          $TASKS"
echo "Modes:          $MODES"
echo "Max subjects:   $MAX_SUBJECTS"

for m in $MODELS; do
    run_one_model "$m"
done

if [ -z "${SKIP_PLOTS:-}" ]; then
    echo ""
    echo "All models done. Generating plots and report..."
    bash eval/finalize_study.sh
fi
