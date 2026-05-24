#!/usr/bin/env bash
# Serve an AutoAWQ-exported directory (W4A16) via official vllm/vllm-openai image.
# Run `python3 quantization/quantize_awq.py` first; default mount: ./gemma-7b-awq at repo root.
# Prereqs: Docker + NVIDIA Container Toolkit — see README.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
if [[ -f "$_SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$_SCRIPT_DIR/.env"
  set +a
fi

AWQ_QUANT_PATH="${AWQ_QUANT_PATH:-$_REPO_ROOT/gemma-7b-awq}"
if [[ ! -d "$AWQ_QUANT_PATH" ]]; then
  echo "ERROR: AWQ weights not found at: $AWQ_QUANT_PATH"
  echo "       Run: python3 quantization/quantize_awq.py --quant-path \"$AWQ_QUANT_PATH\""
  exit 1
fi
AWQ_HOST_PATH="$(cd "$AWQ_QUANT_PATH" && pwd)"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.93}"
PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-0}"
# Match BENCH_MODEL / OpenAI client "model" id when pointing at this server.
SERVED_NAME="${AWQ_SERVED_MODEL_NAME:-google/gemma-7b-it}"

VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE:-vllm/vllm-openai:latest}"
VLLM_DOCKER_NAME="${VLLM_DOCKER_NAME:-vllm-awq}"
HF_CACHE_MOUNT="${HF_CACHE:-$HOME/.cache/huggingface}"
DETACH="${VLLM_DOCKER_DETACH:-0}"

EXTRA=()
if [[ "$PREFIX_CACHE" == "1" ]]; then
  EXTRA+=(--enable-prefix-caching)
fi

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install: sudo apt-get install -y docker.io"
  exit 1
fi

if [[ "${VLLM_DOCKER_SKIP_PULL:-0}" != "1" ]]; then
  echo "Pulling $VLLM_DOCKER_IMAGE ..."
  docker pull "$VLLM_DOCKER_IMAGE"
fi

mkdir -p "$HF_CACHE_MOUNT"

DOCKER_OPTS=(
  --gpus all
  --ipc=host
  --network host
  -v "${HF_CACHE_MOUNT}:/root/.cache/huggingface"
  -v "${AWQ_HOST_PATH}:/models/awq:ro"
)
if [[ -n "${HF_TOKEN:-}" ]]; then DOCKER_OPTS+=(-e "HF_TOKEN=${HF_TOKEN}"); fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then DOCKER_OPTS+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"); fi

# ENTRYPOINT is `vllm serve`; local AWQ folder inside container + --quantization awq
VLLM_ARGS=(
  /models/awq
  --quantization awq
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --trust-remote-code
  --enable-chunked-prefill
  "${EXTRA[@]}"
  "$@"
)

if [[ "$DETACH" == "1" ]]; then
  docker rm -f "$VLLM_DOCKER_NAME" 2>/dev/null || true
  echo "Starting detached AWQ container: $VLLM_DOCKER_NAME (port $PORT)"
  docker run -d --name "$VLLM_DOCKER_NAME" "${DOCKER_OPTS[@]}" "$VLLM_DOCKER_IMAGE" "${VLLM_ARGS[@]}"
  echo "Logs: docker logs -f $VLLM_DOCKER_NAME"
  echo "Stop: docker rm -f $VLLM_DOCKER_NAME"
else
  exec docker run --rm -it "${DOCKER_OPTS[@]}" "$VLLM_DOCKER_IMAGE" "${VLLM_ARGS[@]}"
fi
