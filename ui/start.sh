#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# NLB Web UI — Launch Script
# Usage: bash ui/start.sh [--port 8080] [--no-reload]
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PORT=8080
RELOAD="--reload"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --no-reload) RELOAD=""; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$ROOT_DIR"

# ── Check venv ──────────────────────────────────────────────────────────────
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "❌  Virtual environment not found at $ROOT_DIR/.venv"
  echo "    Create it with: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

# ── Install/verify deps ─────────────────────────────────────────────────────
echo "📦  Checking dependencies..."
pip install -q fastapi "uvicorn[standard]" sse-starlette 2>/dev/null \
  && echo "    ✓ fastapi, uvicorn, sse-starlette ready" \
  || { echo "⚠️  pip install had issues — continuing anyway"; }

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
