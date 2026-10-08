#!/usr/bin/env bash
# Start (or restart) controller + model worker(s) + Gradio web UI in the background.
# Usage: bash runpod/start_demo.sh         start / restart everything
#        bash runpod/start_demo.sh stop    stop everything
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env
activate_conda

stop_all() {
    pkill -f "llava.serve." 2>/dev/null || true
    sleep 2
}

if [[ "${1:-}" == "stop" ]]; then
    stop_all
    echo "All ViP-LLaVA services stopped."
    exit 0
fi
stop_all

case "$LOAD_MODE" in
    fp16) QUANT_FLAG="" ;;
    8bit) QUANT_FLAG="--load-8bit" ;;
    4bit) QUANT_FLAG="--load-4bit" ;;
    *) echo "Invalid LOAD_MODE='$LOAD_MODE' (use fp16 | 8bit | 4bit)" >&2; exit 1 ;;
esac

CONTROLLER_URL="http://localhost:$CONTROLLER_PORT"

# llava.constants.LOGDIR is ".", so service logs and uploaded images are written to the cwd.
cd "$LOG_DIR"

registered_models() {
    curl -sf -X POST "$CONTROLLER_URL/list_models" 2>/dev/null || true
}

fail_with_log() {  # fail_with_log <message> <logfile>
    echo -e "\n[error] $1. Last lines of $2:" >&2
    tail -n 40 "$2" >&2 || true
    exit 1
}

# ---------- Controller ----------
log "Starting controller on :$CONTROLLER_PORT"
nohup python -m llava.serve.controller --host 0.0.0.0 --port "$CONTROLLER_PORT" > controller.out 2>&1 &
controller_pid=$!
for _ in $(seq 1 30); do
    [[ -n "$(registered_models)" ]] && break
    kill -0 "$controller_pid" 2>/dev/null || fail_with_log "Controller exited" controller.out
    sleep 2
done
[[ -n "$(registered_models)" ]] || fail_with_log "Controller did not come up" controller.out

# ---------- Model workers (loaded one at a time) ----------
IFS=',' read -ra MODELS <<< "$MODEL_PATHS"
port=$WORKER_BASE_PORT
for model in "${MODELS[@]}"; do
    model="$(echo "$model" | xargs)"
    [[ -z "$model" ]] && continue
    model="${model%/}"

    # Same naming rule as llava/serve/model_worker.py
    name="$(basename "$model")"
    if [[ "$name" == checkpoint-* ]]; then
        name="$(basename "$(dirname "$model")")_$name"
    fi

    log "Starting model worker '$name' ($LOAD_MODE) on :$port"
    worker_log="worker_${name}.out"
    nohup python -m llava.serve.model_worker --host 0.0.0.0 --port "$port" \
        --controller-address "$CONTROLLER_URL" --worker-address "http://localhost:$port" \
        --model-path "$model" $QUANT_FLAG > "$worker_log" 2>&1 &
    worker_pid=$!

    start=$SECONDS
    until registered_models | grep -qF "\"$name\""; do
        kill -0 "$worker_pid" 2>/dev/null || fail_with_log "Worker '$name' exited" "$worker_log"
        if (( SECONDS - start > WORKER_TIMEOUT )); then
            fail_with_log "Worker '$name' not ready after ${WORKER_TIMEOUT}s" "$worker_log"
        fi
        sleep 5
    done
    echo "Worker '$name' ready after $((SECONDS - start))s"
    port=$((port + 1))
done

# ---------- Gradio ----------
SHARE_FLAG=""
[[ "$GRADIO_SHARE" == "1" ]] && SHARE_FLAG="--share"

log "Starting Gradio web UI on :$GRADIO_PORT"
nohup python -m llava.serve.gradio_web_server --host 0.0.0.0 --port "$GRADIO_PORT" \
    --controller-url "$CONTROLLER_URL" --model-list-mode reload $SHARE_FLAG > gradio.out 2>&1 &
gradio_pid=$!

for _ in $(seq 1 60); do
    curl -sf -o /dev/null "http://localhost:$GRADIO_PORT" && break
    kill -0 "$gradio_pid" 2>/dev/null || fail_with_log "Gradio exited" gradio.out
    sleep 2
done
curl -sf -o /dev/null "http://localhost:$GRADIO_PORT" || fail_with_log "Gradio did not come up" gradio.out

share_url=""
if [[ -n "$SHARE_FLAG" ]]; then
    for _ in $(seq 1 30); do
        share_url="$(grep -oE 'https://[a-zA-Z0-9.-]+\.gradio\.live' gradio.out | tail -n 1 || true)"
        [[ -n "$share_url" ]] && break
        sleep 2
    done
fi

echo
echo "================ ViP-LLaVA demo is running ================"
echo "Models       : $(registered_models)"
echo "Local        : http://localhost:$GRADIO_PORT"
if [[ -n "$share_url" ]]; then
    echo "Public link  : $share_url   (valid 72h)"
elif [[ -n "$SHARE_FLAG" ]]; then
    echo "Public link  : not found yet, check: grep gradio.live $LOG_DIR/gradio.out"
fi
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "RunPod proxy : https://${RUNPOD_POD_ID}-${GRADIO_PORT}.proxy.runpod.net   (needs HTTP port $GRADIO_PORT exposed on the pod)"
fi
echo "Logs         : $LOG_DIR/{controller,worker_*,gradio}.out"
echo "Stop         : bash $REPO_DIR/runpod/start_demo.sh stop"
echo "============================================================"
