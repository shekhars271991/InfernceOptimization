#!/usr/bin/env bash
# lm-evaluation-harness against an OpenAI-compatible vLLM server.
# Prerequisites: pip install 'lm-eval[api]' (tenacity for API models), server at BASE_URL.
#
# Uses **local-completions** + `/v1/completions` so tasks can use **loglikelihood** (MMLU,
# truthfulqa_mc1, hellaswag, …). `local-chat-completions` + chat API only supports
# generate_until (e.g. GSM8K) and fails with: "Loglikelihood is not supported for chat completions".
#
# TruthfulQA: lm-eval registers truthfulqa_mc1 / truthfulqa_mc2 / truthfulqa_gen (not truthfulqa_mc).
# If vLLM errors on logprobs count, raise server `--max-logprobs` (see vLLM OpenAI server docs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -d "$ROOT/.venv" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi

MODEL_NAME="${BENCH_MODEL:-google/gemma-7b-it}"
BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
COMPLETIONS_URL="${BENCH_COMPLETIONS_URL:-${BASE_URL%/}/v1/completions}"
MAX_LENGTH="${LM_EVAL_MAX_LENGTH:-4096}"
TOKENIZER_BACKEND="${LM_EVAL_TOKENIZER_BACKEND:-huggingface}"
NUM_CONCURRENT="${LM_EVAL_NUM_CONCURRENT:-4}"
TOKENIZED="${LM_EVAL_TOKENIZED_REQUESTS:-True}"

TASKS="${LM_EVAL_TASKS:-mmlu,gsm8k,hellaswag,truthfulqa_mc1}"
# Default 100 examples per task (quick run). Full datasets: `export LM_EVAL_LIMIT=` (empty) first.
if [[ ! -v LM_EVAL_LIMIT ]]; then
  export LM_EVAL_LIMIT=100
fi
LIMIT="${LM_EVAL_LIMIT}"

# --apply_chat_template: instruct prompts on the completions API (see lm-eval docs).
# Pass more flags after the script name, e.g. ./eval/run_lm_eval.sh --trust_remote_code
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL_ARGS="model=${MODEL_NAME},base_url=${COMPLETIONS_URL},num_concurrent=${NUM_CONCURRENT},tokenized_requests=${TOKENIZED},tokenizer_backend=${TOKENIZER_BACKEND},max_length=${MAX_LENGTH}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},token=${HF_TOKEN}"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},token=${HUGGING_FACE_HUB_TOKEN}"
fi

ARGS=(
  --model local-completions
  --model_args "${MODEL_ARGS}"
  --apply_chat_template
  --tasks "${TASKS}"
  --batch_size "${LM_EVAL_BATCH_SIZE:-1}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "Running lm_eval with tasks=${TASKS} limit=${LIMIT:-none} completions_url=${COMPLETIONS_URL}"
python3 -m lm_eval "${ARGS[@]}" "$@"
