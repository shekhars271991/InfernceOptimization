#!/usr/bin/env bash
# vLLM prefill node (kv_producer) for disaggregated prefill — see vLLM docs.
# On a second machine, set VLLM_HOST_IP to this host's reachable IP.
set -euo pipefail

MODEL_NAME="${HF_MODEL_NAME:-google/gemma-7b-it}"
: "${VLLM_HOST_IP:=127.0.0.1}"

HOST="${VLLM_PREFILL_HOST:-0.0.0.0}"
PORT="${VLLM_PREFILL_PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
export CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-0}"

KV_PORT="${VLLM_PREFILL_KV_PORT:-14579}"
PROXY_PORT="${VLLM_KV_PROXY_PORT:-30001}"

# shellcheck disable=SC2089
KV_JSON="$(printf '%s' "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":\"1e9\",\"kv_port\":\"${KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${VLLM_HOST_IP}\",\"proxy_port\":\"${PROXY_PORT}\",\"http_ip\":\"${VLLM_HOST_IP}\",\"http_port\":\"${PORT}\",\"send_type\":\"PUT_ASYNC\"}}")"

exec vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-transfer-config "$KV_JSON" \
  "$@"
