#!/usr/bin/env bash
# One-shot GPU instance prep: CUDA sanity check, Python venv + deps (avoids Ubuntu PEP 668).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${INFERENCEOPT_VENV:-$ROOT/.venv}"

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

echo "== Python (system) =="
python3 -V

ensure_venv_os_packages() {
  # Ubuntu/Debian: need python3-venv for `python3 -m venv`; python3-full avoids partial venvs (see README.venv).
  if command -v apt-get &>/dev/null; then
    if ! python3 -m venv --help &>/dev/null; then
      echo "Installing python3-venv python3-full via apt (sudo)..."
      sudo apt-get update -qq
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-full
    fi
  elif command -v dnf &>/dev/null; then
    if ! python3 -m venv --help &>/dev/null; then
      echo "Installing Python venv support via dnf (sudo)..."
      sudo dnf install -y python3
    fi
  elif command -v yum &>/dev/null; then
    if ! python3 -m venv --help &>/dev/null; then
      sudo yum install -y python3
    fi
  fi
}

echo "== Virtualenv + pip (project-local, PEP 668–safe) =="
ensure_venv_os_packages

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY missing after venv create"
  exit 1
fi

"$PY" -m pip install --upgrade pip wheel
"$PIP" install -r "$ROOT/requirements.txt"

echo "== Hugging Face token =="
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  echo "Set HUGGING_FACE_HUB_TOKEN or HF_TOKEN for gated models (e.g. Gemma license acceptance on HF)."
else
  echo "HF token env is set (HUGGING_FACE_HUB_TOKEN or HF_TOKEN)."
fi

echo "== Optional: vLLM prebuilt wheel =="
echo "If pip vLLM fails to build, follow https://docs.vllm.ai/en/stable/getting_started/installation.html for your CUDA version."

echo ""
echo "Done. venv: $VENV_DIR"
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Or use:    $VENV_DIR/bin/python ...   # launch scripts prepend .venv to PATH automatically"
echo "Next: export HF token, then ./scripts/launch_baseline.sh (or disagg scripts)."
