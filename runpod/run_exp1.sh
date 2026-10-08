#!/usr/bin/env bash
# Exp1 on the pod: stop the demo (frees VRAM), run test/exp1/run_exp1.py, then commit + push the new batch folder.
# Usage: bash runpod/run_exp1.sh [--no-push] [run_exp1.py args, e.g. --limit 1 --layers 10-29]
set -eo pipefail

PUSH=1
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-push" ]]; then PUSH=0; else ARGS+=("$arg"); fi
done

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env
activate_conda
cd "$REPO_DIR"

if [[ $PUSH -eq 1 ]] && ! git config user.email >/dev/null; then
    echo "git user is not configured. Run: git config --global user.name ... && git config --global user.email ..." >&2
    echo "(or pass --no-push to skip committing the results)" >&2
    exit 1
fi

log "Stopping the demo to free VRAM"
pkill -f "llava.serve." 2>/dev/null || true
sleep 2

log "Updating repo ($(git rev-parse --abbrev-ref HEAD))"
git pull --ff-only

python -c "import matplotlib" 2>/dev/null || pip install matplotlib==3.8.4

MODEL_PATH="$(echo "${MODEL_PATHS%%,*}" | xargs)"
log "Running exp1 with $MODEL_PATH ($LOAD_MODE)"
python test/exp1/run_exp1.py --model-path "$MODEL_PATH" --load-mode "$LOAD_MODE" "${ARGS[@]}"

BATCH_DIR="$(ls -1td test/exp1/output/*/ | head -n 1)"
BATCH_DIR="${BATCH_DIR%/}"
echo "Batch folder: $BATCH_DIR ($(ls -1 "$BATCH_DIR" | wc -l) files)"

if [[ $PUSH -eq 1 ]]; then
    log "Committing and pushing $BATCH_DIR"
    git add "$BATCH_DIR"
    git commit -m "exp1: results $(basename "$BATCH_DIR")"
    git push
    echo "Done. Locally run: git pull"
fi
