#!/usr/bin/env bash
# One-shot ViP-LLaVA setup on RunPod: Miniconda -> conda env + libs -> HF weights (hf_transfer) -> Gradio demo.
# Usage: bash setup_runpod.sh [--skip-install] [--skip-download] [--no-launch]
set -eo pipefail

SKIP_INSTALL=0
SKIP_DOWNLOAD=0
LAUNCH=1
for arg in "$@"; do
    case "$arg" in
        --skip-install) SKIP_INSTALL=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --no-launch) LAUNCH=0 ;;
        -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

source "$(dirname "${BASH_SOURCE[0]}")/runpod/common.sh"
load_env
cd "$REPO_DIR"

log "Directories: DATA_DIR=$DATA_DIR | HF_HOME=$HF_HOME | LOG_DIR=$LOG_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found, GPU may not be available."
fi

if [[ $SKIP_INSTALL -eq 0 ]]; then
    # ---------- 1. Miniconda ----------
    if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
        log "Installing Miniconda into $CONDA_DIR"
        installer=/tmp/miniconda.sh
        curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash "$installer" -b -p "$CONDA_DIR"
        rm -f "$installer"
    else
        log "Miniconda already installed at $CONDA_DIR"
    fi
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"

    # Recent conda refuses to use the Anaconda channels until their ToS is accepted.
    export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

    # ---------- 2. Conda env ----------
    if [[ ! -d "$CONDA_DIR/envs/$CONDA_ENV_NAME" ]]; then
        log "Creating conda env '$CONDA_ENV_NAME' (python $PYTHON_VERSION)"
        conda create -y -n "$CONDA_ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VERSION" pip
    else
        log "Conda env '$CONDA_ENV_NAME' already exists"
    fi
    conda activate "$CONDA_ENV_NAME"

    # Auto-activate the env in new terminals (~/.bashrc lives on the container disk, re-added each setup).
    activate_line="source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $CONDA_ENV_NAME"
    grep -qxF "$activate_line" ~/.bashrc 2>/dev/null || echo "$activate_line" >> ~/.bashrc

    # ---------- 3. Python packages ----------
    log "Installing PyTorch 2.1.2 (cu121 wheels run on the CUDA 12.8 driver)"
    python -m pip install --upgrade pip
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

    log "Installing pinned dependencies (runpod/requirements.txt)"
    pip install -r runpod/requirements.txt

    log "Installing ViP-LLaVA (editable, deps already pinned above)"
    pip install -e . --no-deps

    log "Sanity check"
    python - <<'PY'
import torch, transformers, gradio, numpy
import google.protobuf  # noqa: F401
# llava/model/__init__.py silently swallows import errors, so import the class directly.
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM  # noqa: F401
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__} "
      f"| gradio {gradio.__version__} | numpy {numpy.__version__}")
assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
print("GPU:", torch.cuda.get_device_name(0))
PY
else
    activate_conda
fi

# ---------- 4. Weights ----------
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    log "Downloading weights into $HF_HOME"
    python runpod/download_weights.py
fi

# ---------- 5. Demo ----------
if [[ $LAUNCH -eq 1 ]]; then
    bash runpod/start_demo.sh
else
    log "Setup done. Start the demo with: bash runpod/start_demo.sh"
fi
