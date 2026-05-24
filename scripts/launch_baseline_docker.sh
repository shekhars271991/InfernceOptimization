#!/usr/bin/env bash
# Single-node vLLM via official Docker image (avoids host nvcc / FlashInfer JIT on minimal Ubuntu).
# Prereqs: Docker + NVIDIA Container Toolkit — see README.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$_SCRIPT_DIR/.env"
  set +a
fi

MODEL_NAME="${HF_MODEL_NAME:-google/gemma-7b-it}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-0}"

VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE:-vllm/vllm-openai:latest}"
VLLM_DOCKER_NAME="${VLLM_DOCKER_NAME:-vllm-baseline}"
HF_CACHE_MOUNT="${HF_CACHE:-$HOME/.cache/huggingface}"
DETACH="${VLLM_DOCKER_DETACH:-0}"

EXTRA=()
if [[ "$PREFIX_CACHE" == "1" ]]; then
  EXTRA+=(--enable-prefix-caching)
fi

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install: sudo apt-get install -y docker.io"
  echo "       GPU: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
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
)
if [[ -n "${HF_TOKEN:-}" ]]; then DOCKER_OPTS+=(-e "HF_TOKEN=${HF_TOKEN}"); fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then DOCKER_OPTS+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"); fi

VLLM_ARGS=(
  --model "$MODEL_NAME"
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
  echo "Starting detached container: $VLLM_DOCKER_NAME"
  docker run -d --name "$VLLM_DOCKER_NAME" "${DOCKER_OPTS[@]}" "$VLLM_DOCKER_IMAGE" "${VLLM_ARGS[@]}"
  echo "Logs: docker logs -f $VLLM_DOCKER_NAME"
  echo "Stop: docker rm -f $VLLM_DOCKER_NAME"
else
  exec docker run --rm -it "${DOCKER_OPTS[@]}" "$VLLM_DOCKER_IMAGE" "${VLLM_ARGS[@]}"
fi
