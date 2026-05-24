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
if [ -z "${LM_EVAL_LIMIT+x}" ]; then
  export LM_EVAL_LIMIT=100
fi
LIMIT="${LM_EVAL_LIMIT}"

# --apply_chat_template: instruct prompts on the completions API (see lm-eval docs).
# Gemma (e.g. google/gemma-7b-it) chat templates raise "System role not supported" because
# few-shot tasks include system messages — skip unless LM_EVAL_APPLY_CHAT_TEMPLATE=1 forces it.
# LM_EVAL_APPLY_CHAT_TEMPLATE=0 always skips.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

_apply_chat_template_flag() {
  case "${LM_EVAL_APPLY_CHAT_TEMPLATE:-auto}" in
    0|false|False|no|NO) return 1 ;;
    1|true|True|yes|YES) return 0 ;;
    auto)
      if echo "${MODEL_NAME}" | grep -qi gemma; then
        return 1
      fi
      return 0
      ;;
    *)
      echo "WARN: unknown LM_EVAL_APPLY_CHAT_TEMPLATE=${LM_EVAL_APPLY_CHAT_TEMPLATE}; using auto" >&2
      if echo "${MODEL_NAME}" | grep -qi gemma; then
        return 1
      fi
      return 0
      ;;
  esac
}

MODEL_ARGS="model=${MODEL_NAME},base_url=${COMPLETIONS_URL},num_concurrent=${NUM_CONCURRENT},tokenized_requests=${TOKENIZED},tokenizer_backend=${TOKENIZER_BACKEND},max_length=${MAX_LENGTH}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},token=${HF_TOKEN}"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},token=${HUGGING_FACE_HUB_TOKEN}"
fi

ARGS=(
  --model local-completions
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${LM_EVAL_BATCH_SIZE:-1}"
)
if _apply_chat_template_flag; then
  ARGS+=(--apply_chat_template)
fi

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

# Save JSON (default: results/lm_eval/run-<timestamp>.json, gitignored). LM_EVAL_OUTPUT_PATH=..., LM_EVAL_NO_SAVE=1, LM_EVAL_OUT_DIR=...
if [[ "${LM_EVAL_NO_SAVE:-0}" != "1" ]]; then
  if [[ -n "${LM_EVAL_OUTPUT_PATH:-}" ]]; then
    OUTPUT_PATH="$LM_EVAL_OUTPUT_PATH"
  else
    OUT_DIR="${LM_EVAL_OUT_DIR:-$ROOT/results/lm_eval}"
    OUTPUT_PATH="$OUT_DIR/run-$(date +%Y%m%d-%H%M%S).json"
  fi
  mkdir -p "$(dirname "$OUTPUT_PATH")"
  ARGS+=(--output_path "$OUTPUT_PATH")
fi

_chat_note="off"
if _apply_chat_template_flag; then _chat_note="on"; fi
echo "Running lm_eval with tasks=${TASKS} limit=${LIMIT:-none} completions_url=${COMPLETIONS_URL} apply_chat_template=${_chat_note}"
if [[ "${LM_EVAL_NO_SAVE:-0}" != "1" ]]; then
  echo "Results JSON: ${OUTPUT_PATH:-}"
fi
python3 -m lm_eval "${ARGS[@]}" "$@"
