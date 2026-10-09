#!/usr/bin/env bash
# Exp1 on the pod: stop the demo (frees VRAM), then run test/exp1/run_exp1.py.
# Results stay in test/exp1/output/<timestamp>/; copy that folder off the pod yourself.
# Usage: bash runpod/run_exp1.sh [run_exp1.py args, e.g. --limit 1 --layers 10-29]
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env
activate_conda
cd "$REPO_DIR"

log "Stopping the demo to free VRAM"
pkill -f "llava.serve." 2>/dev/null || true
sleep 2

log "Updating repo ($(git rev-parse --abbrev-ref HEAD))"
git pull --ff-only

python -c "import matplotlib, scipy.ndimage" 2>/dev/null || pip install matplotlib==3.8.4 scipy==1.13.1

MODEL_PATH="$(echo "${MODEL_PATHS%%,*}" | xargs)"
log "Running exp1 with $MODEL_PATH ($LOAD_MODE)"
python test/exp1/run_exp1.py --model-path "$MODEL_PATH" --load-mode "$LOAD_MODE" "$@"

# Do not pipe ls into head: with pipefail, head's early close makes ls die on SIGPIPE.
BATCH_DIR="$(printf '%s\n' test/exp1/output/*/ | sort | tail -n 1)"
BATCH_DIR="${BATCH_DIR%/}"
if [[ ! -d "$BATCH_DIR" ]]; then
    echo "No batch folder found under test/exp1/output" >&2
    exit 1
fi
echo "Batch folder: $BATCH_DIR ($(ls -1 "$BATCH_DIR" | wc -l) files)"
echo "Copy it off the pod, for example: scp -r <pod>:$REPO_DIR/$BATCH_DIR ."
