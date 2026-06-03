#!/usr/bin/env bash
# run_all_models.sh — Phase 2 LLM evaluation for all 8 target models.
#
# The message bank (Phase 1) is already built at:
#   run/data/validated_message_bank.csv
#
# BEFORE RUNNING: export all required API keys:
#
#   export DEEPINFRA_API="<your-deepinfra-key>"       # covers 6 models
#   export OPENAI_API_KEY="<your-openai-key>"         # GPT-4o
#   export ANTHROPIC_API_KEY="<your-anthropic-key>"   # Claude-3.5-Haiku
#   export GOOGLE_API="<your-google-ai-studio-key>"   # Gemini-1.5-Pro
#
# Usage:
#   bash run_all_models.sh                 # run all 8 models sequentially
#   bash run_all_models.sh GPT-4o          # run single model
#   bash run_all_models.sh Llama-3.1-70B Qwen-2.5-72B-Instruct  # subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python3"
BANK="${SCRIPT_DIR}/data/validated_message_bank.csv"
N_WORKERS=5     # parallel worker processes
ROW_WORKERS=10  # threads per worker (total concurrency = N_WORKERS × ROW_WORKERS)

# ── Map model name → required env var (bash 3.2 compatible) ──────────────────
model_api_key() {
    case "$1" in
        "GPT-4o")               echo "OPENAI_API_KEY" ;;
        "Claude-3.5-Haiku")     echo "ANTHROPIC_API_KEY" ;;
        "Gemini-1.5-Pro")       echo "GOOGLE_API" ;;
        "Llama-3.1-70B")        echo "DEEPINFRA_API" ;;
        "Qwen-2.5-72B-Instruct")echo "DEEPINFRA_API" ;;
        "Llama-3.1-8B")         echo "DEEPINFRA_API" ;;
        "Phi-3.5")              echo "DEEPINFRA_API" ;;
        "Gemma-2-9B-IT")        echo "DEEPINFRA_API" ;;
        "DeepSeek-V3")          echo "DEEPINFRA_API" ;;
        *) echo "UNKNOWN" ;;
    esac
}

# Default: run all models; override with command-line args
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("GPT-4o" "Claude-3.5-Haiku" "Gemini-1.5-Pro"
            "Llama-3.1-70B" "Qwen-2.5-72B-Instruct"
            "Llama-3.1-8B" "Phi-3.5" "Gemma-2-9B-IT" "DeepSeek-V3")
fi

# ── Check keys before starting ────────────────────────────────────────────────
echo "=== Key check ==="
MISSING=0
for model in "${MODELS[@]}"; do
    key="$(model_api_key "$model")"
    val="$(eval echo \"\${${key}:-}\")"
    if [ -z "$val" ]; then
        echo "  MISSING  $key  (needed for $model)"
        MISSING=1
    else
        echo "  OK       $key  (for $model)"
    fi
done
if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Missing API keys. Export them and re-run."
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
now()      { date "+%Y-%m-%d %H:%M:%S"; }
elapsed()  {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dm %02ds" $(( secs / 60 )) $(( secs % 60 ))
}

# ── Run each model ────────────────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
COMPLETED=0
FAILED=0
SUITE_START=$(date +%s)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  Conditional Bias Evaluation — ${TOTAL} model(s)  |  started $(now)"
echo "╚══════════════════════════════════════════════════════════╝"

for model in "${MODELS[@]}"; do
    IDX=$(( COMPLETED + FAILED + 1 ))
    MODEL_START=$(date +%s)

    echo ""
    echo "┌─────────────────────────────────────────────────────────"
    echo "│  [${IDX}/${TOTAL}]  ${model}"
    echo "│  Started : $(now)"
    echo "└─────────────────────────────────────────────────────────"

    cd "$SCRIPT_DIR"
    if "$VENV_PYTHON" run_pilot.py \
            --message_bank "$BANK" \
            --model        "$model" \
            --n_workers    "$N_WORKERS" \
            --row_workers  "$ROW_WORKERS"; then
        COMPLETED=$(( COMPLETED + 1 ))
        echo ""
        echo "  ✔  ${model}  finished in $(elapsed $MODEL_START)"
    else
        FAILED=$(( FAILED + 1 ))
        echo ""
        echo "  ✘  ${model}  FAILED after $(elapsed $MODEL_START)"
    fi

    REMAINING=$(( TOTAL - IDX ))
    if [ $REMAINING -gt 0 ]; then
        echo "  Remaining : ${REMAINING} model(s) — ${MODELS[*]:$IDX}"
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  Done  |  ${COMPLETED}/${TOTAL} succeeded  |  ${FAILED} failed"
echo "  Total wall time: $(elapsed $SUITE_START)"
echo "  Finished: $(now)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Results: ${SCRIPT_DIR}/data/conditional_decision_results/"
echo ""
echo "Next step — run multi-model analysis:"
echo "  $VENV_PYTHON ${SCRIPT_DIR}/multi_llm_analysis.py"
