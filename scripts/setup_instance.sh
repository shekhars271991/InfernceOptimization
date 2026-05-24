#!/usr/bin/env bash
# One-shot GPU instance prep: CUDA sanity check, Python deps, HF auth hints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== CUDA / GPU =="
if ! command -v nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not found. Install NVIDIA drivers before vLLM."
else
  nvidia-smi || true
fi

if command -v nvcc &>/dev/null; then
  echo "nvcc: $(nvcc --version | tail -1)"
else
  echo "NOTE: nvcc not in PATH (often OK if using prebuilt CUDA wheels)."
fi

echo "== Python =="
python3 -V

echo "== pip upgrade + requirements =="
python3 -m pip install --upgrade pip wheel
python3 -m pip install -r requirements.txt

echo "== Hugging Face token =="
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  echo "Set HUGGING_FACE_HUB_TOKEN or HF_TOKEN for gated models (e.g. Gemma license acceptance on HF)."
else
  echo "HF token env is set (HUGGING_FACE_HUB_TOKEN or HF_TOKEN)."
fi

echo "== Optional: vLLM prebuilt wheel =="
echo "If pip vLLM fails to build, follow https://docs.vllm.ai/en/stable/getting_started/installation.html for your CUDA version."

echo "Done. Next: export HF token, then run scripts/launch_baseline.sh or disagg scripts."
