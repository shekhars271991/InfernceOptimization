#!/usr/bin/env bash
# lm-evaluation-harness against an OpenAI-compatible vLLM server (chat).
# Prerequisites: pip install 'lm-eval[api]' (tenacity for API models), server at BASE_URL.
# Note: some MC/loglikelihood tasks prefer the /v1/completions API; if MMLU fails,
# drop it from LM_EVAL_TASKS or switch to a completions-based flow (see lm-eval docs).
# TruthfulQA: lm-eval registers truthfulqa_mc1 / truthfulqa_mc2 / truthfulqa_gen (not truthfulqa_mc).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -d "$ROOT/.venv" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi

MODEL_NAME="${BENCH_MODEL:-google/gemma-7b-it}"
BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
CHAT_URL="${BASE_URL%/}/v1/chat/completions"
TASKS="${LM_EVAL_TASKS:-mmlu,gsm8k,hellaswag,truthfulqa_mc1}"
LIMIT="${LM_EVAL_LIMIT:-}" # e.g. export LM_EVAL_LIMIT=50 for smoke tests

# local-chat-completions requires OpenAI-style messages; --apply_chat_template formats prompts that way.
# Pass more flags after the script name, e.g. ./eval/run_lm_eval.sh --trust_remote_code
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

ARGS=(
  --model local-chat-completions
  --model_args "model=${MODEL_NAME},base_url=${CHAT_URL},num_concurrent=${LM_EVAL_NUM_CONCURRENT:-4},tokenized_requests=False"
  --apply_chat_template
  --tasks "${TASKS}"
  --batch_size "${LM_EVAL_BATCH_SIZE:-auto}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "Running lm_eval with tasks=${TASKS} chat_url=${CHAT_URL}"
python3 -m lm_eval "${ARGS[@]}" "$@"
