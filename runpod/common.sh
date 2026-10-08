#!/usr/bin/env bash
# Shared helpers for setup_runpod.sh and runpod/start_demo.sh. Source it, don't execute it.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }

load_env() {
    local env_file="$REPO_DIR/.env"
    if [[ ! -f "$env_file" ]]; then
        cp "$REPO_DIR/.env.example" "$env_file"
        echo "[env] Created .env from .env.example (edit HF_TOKEN / MODEL_PATHS there if needed)."
    fi
    set -a
    # Strip CR so a .env edited on Windows still parses.
    # shellcheck disable=SC1090
    source <(sed 's/\r$//' "$env_file")
    set +a

    : "${DATA_DIR:=/workspace/data}"
    : "${HF_HOME:=$DATA_DIR/huggingface}"
    : "${HF_HUB_ENABLE_HF_TRANSFER:=1}"
    : "${CONDA_DIR:=/workspace/miniconda3}"
    : "${CONDA_ENV_NAME:=vip-llava}"
    : "${PYTHON_VERSION:=3.10}"
    : "${MODEL_PATHS:=mucai/vip-llava-7b}"
    : "${LOAD_MODE:=fp16}"
    : "${CONTROLLER_PORT:=10000}"
    : "${WORKER_BASE_PORT:=40000}"
    : "${GRADIO_PORT:=7860}"
    : "${GRADIO_SHARE:=1}"
    : "${GRADIO_ANALYTICS_ENABLED:=False}"
    : "${LOG_DIR:=$DATA_DIR/logs}"
    : "${WORKER_TIMEOUT:=1800}"
    export DATA_DIR HF_HOME HF_HUB_ENABLE_HF_TRANSFER CONDA_DIR CONDA_ENV_NAME PYTHON_VERSION \
        MODEL_PATHS LOAD_MODE CONTROLLER_PORT WORKER_BASE_PORT GRADIO_PORT GRADIO_SHARE \
        GRADIO_ANALYTICS_ENABLED LOG_DIR WORKER_TIMEOUT
    export PIP_ROOT_USER_ACTION=ignore

    # An empty HF_TOKEN would be sent as an invalid token.
    if [[ -z "${HF_TOKEN:-}" ]]; then unset HF_TOKEN; fi

    mkdir -p "$DATA_DIR" "$HF_HOME" "$LOG_DIR"
}

activate_conda() {
    if [[ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
        echo "Conda not found at $CONDA_DIR. Run: bash setup_runpod.sh" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
}
