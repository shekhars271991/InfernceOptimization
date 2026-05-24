#!/usr/bin/env bash
# vLLM decode node (kv_consumer) for disaggregated prefill — see vLLM docs.
# kv_port must differ from prefill (vLLM upstream example uses 14580 for decode).
set -euo pipefail

MODEL_NAME="${HF_MODEL_NAME:-google/gemma-7b-it}"
: "${VLLM_HOST_IP:=127.0.0.1}"

HOST="${VLLM_DECODE_HOST:-0.0.0.0}"
PORT="${VLLM_DECODE_PORT:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
export CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-1}"

KV_PORT="${VLLM_DECODE_KV_PORT:-14580}"
PROXY_PORT="${VLLM_KV_PROXY_PORT:-30001}"

KV_JSON="$(printf '%s' "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":2,\"kv_buffer_size\":\"1e10\",\"kv_port\":\"${KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${VLLM_HOST_IP}\",\"proxy_port\":\"${PROXY_PORT}\",\"http_ip\":\"${VLLM_HOST_IP}\",\"http_port\":\"${PORT}\",\"send_type\":\"PUT_ASYNC\"}}")"

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
