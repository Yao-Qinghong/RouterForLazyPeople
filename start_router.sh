#!/bin/bash
# ─────────────────────────────────────────────
# LLM Router (Port 9001)
# Auto-starts/stops llama-server backends on demand
# ─────────────────────────────────────────────
set -e

ROUTER_DIR="$HOME/llm-router"
LOG="$HOME/llama-router.log"
PORT=9001
VENV="$ROUTER_DIR/.venv"
LLAMA_DIR="$HOME/llama.cpp"

# ── Update & rebuild llama.cpp ───────────────
update_llama() {
    echo "── Updating llama.cpp ──"
    cd "$LLAMA_DIR"

    LOCAL=$(git rev-parse HEAD)
    git fetch --quiet
    REMOTE=$(git rev-parse @{u})

    if [ "$LOCAL" = "$REMOTE" ]; then
        echo "llama.cpp already up to date ($(git describe --tags --always))"
        return 0
    fi

    echo "Pulling latest changes..."
    git pull --quiet

    echo "Rebuilding (this may take a few minutes)..."
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native > /dev/null 2>&1
    cmake --build build --config Release -j$(nproc) > /dev/null 2>&1

    echo "Updated to $(git describe --tags --always)"
}

# ── Parse flags ──────────────────────────────
UPDATE_LLAMA=false
for arg in "$@"; do
    case "$arg" in
        --update-llama) UPDATE_LLAMA=true ;;
        --help|-h)
            echo "Usage: $0 [--update-llama]"
            echo "  --update-llama   Pull & rebuild llama.cpp before starting"
            exit 0
            ;;
    esac
done

if [ "$UPDATE_LLAMA" = true ]; then
    update_llama
fi

# ── Install deps if venv missing ─────────────
if [ ! -d "$VENV" ]; then
    echo "Creating venv and installing deps..."
    mkdir -p "$ROUTER_DIR"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q fastapi uvicorn httpx
fi

# ── Copy router.py into place if needed ──────
if [ -f "$(dirname "$0")/router.py" ] && [ "$(dirname "$0")" != "$ROUTER_DIR" ]; then
    cp "$(dirname "$0")/router.py" "$ROUTER_DIR/router.py"
fi

# ── Kill existing router if running ──────────
EXISTING=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Stopping existing router on port $PORT (PID $EXISTING)..."
    kill $EXISTING 2>/dev/null || true
    sleep 1
fi

# ── Start router ─────────────────────────────
echo "Starting LLM Router on port $PORT..."
echo "Log: $LOG"
nohup "$VENV/bin/uvicorn" router:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --app-dir "$ROUTER_DIR" \
    > "$LOG" 2>&1 &

PID=$!
echo "Started with PID $PID"
echo ""
echo "  Status:  curl http://localhost:$PORT/status"
echo "  Logs:    tail -f $LOG"
echo "  Stop:    kill $PID"
