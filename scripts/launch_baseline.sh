#!/usr/bin/env bash
# Single-node vLLM OpenAI server — FP16/BF16 baseline (no disaggregated prefill).
set -euo pipefail

# Default: Gemma 7B Instruct (accept license on Hugging Face first).
MODEL_NAME="${HF_MODEL_NAME:-google/gemma-7b-it}"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-0}"

EXTRA=()
if [[ "$PREFIX_CACHE" == "1" ]]; then
  EXTRA+=(--enable-prefix-caching)
fi

exec vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --enable-chunked-prefill \
  "${EXTRA[@]}" \
  "$@"
