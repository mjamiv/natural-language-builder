#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# NLB Web UI — Launch Script
# Usage: bash ui/start.sh [--port 8080] [--no-reload] [--install-deps]
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PORT=8080
RELOAD="--reload"
INSTALL_DEPS=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ -n "$2" ]] || { echo "--port requires a value"; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --no-reload)
      RELOAD=""
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]{1,5}$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "❌ Invalid port: $PORT (must be 1-65535)"
  exit 1
fi

cd "$ROOT_DIR"

# ── Check venv ──────────────────────────────────────────────────────────────
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "❌  Virtual environment not found at $ROOT_DIR/.venv"
  echo "    Create it with: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

# ── Verify deps (optional install) ──────────────────────────────────────────
if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "📦 Installing UI dependencies..."
  pip install -q -r ui/requirements.txt \
    && echo "    ✓ dependencies installed" \
    || { echo "❌ dependency install failed"; exit 1; }
else
  python - <<'PY' >/dev/null || {
import importlib
for mod in ("fastapi", "uvicorn", "sse_starlette"):
    importlib.import_module(mod)
PY
    echo "❌ Missing dependencies. Run: bash ui/start.sh --install-deps"
    exit 1
  }
fi

# ── Check server file exists ────────────────────────────────────────────────
if [[ ! -f "ui/server.py" ]]; then
  echo "❌  ui/server.py not found. Has Track 2 been completed?"
  exit 1
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "  🔴  Natural Language Builder — Web UI"
echo "  ─────────────────────────────────────"
echo "  URL:     http://localhost:$PORT"
echo "  Root:    $ROOT_DIR"
echo "  Reload:  ${RELOAD:+enabled}${RELOAD:-disabled}"
echo ""
echo "  Try pasting this example:"
echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │ 3-span continuous steel plate girder over the Kishwaukee     │"
echo "  │ River on I-39 in northern Illinois. 315-420-315 ft spans,   │"
echo "  │ 7 girders at 9.5' spacing. ILM erection.                    │"
echo "  └──────────────────────────────────────────────────────────────┘"
echo ""
echo "  Press Ctrl-C to stop."
echo ""

# ── Launch ──────────────────────────────────────────────────────────────────
exec python -m uvicorn ui.server:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --log-level info \
  $RELOAD
