# NLB Web UI — Build Spec

## Goal
A clean web UI around the existing `nlb` CLI pipeline so Michael can type a bridge description, click Run, watch progress, and view outputs (moment diagram, reports, results).

## Architecture
- **Backend:** Python FastAPI server (`ui/server.py`) on port 8080
- **Frontend:** Single HTML file (`ui/static/index.html`) with embedded CSS/JS (no build step)
- **Communication:** Server-Sent Events (SSE) for streaming run progress/logs
- **Working dir:** `/tmp/natural-language-builder`

## API Endpoints

### `POST /api/run`
- Body: `{ "description": "3-span steel plate girder..." }`
- Returns: `{ "run_id": "uuid" }`
- Starts pipeline in background thread

### `GET /api/run/{run_id}/status`
- Returns: `{ "status": "running|completed|failed", "step": "site_recon|building|assembling|analyzing|red_team|report", "progress": 0-100, "elapsed": 12.3 }`

### `GET /api/run/{run_id}/logs`
- SSE stream of log lines as they happen
- Format: `data: {"ts": "...", "level": "info|warn|error|success", "message": "...", "step": "site_recon"}\n\n`

### `GET /api/run/{run_id}/results`
- Returns full results JSON when complete

### `GET /api/run/{run_id}/artifacts`
- Returns: `{ "artifacts": [{"name": "moment-diagram.html", "type": "html", "url": "/api/run/{id}/artifact/moment-diagram.html"}, ...] }`

### `GET /api/run/{run_id}/artifact/{filename}`
- Serves the actual file (HTML, MD, JSON)

### `GET /api/examples`
- Returns list of example bridge descriptions for quick-start

## Frontend Layout
```
┌─────────────────────────────────────────┐
│  🔴 Red Team Your Bridge                │
├─────────────────────────────────────────┤
│  [Textarea: Describe your bridge...]    │
│  [Example dropdown] [▶ Run Analysis]    │
├─────────────────────────────────────────┤
│  Progress: ████████░░ 80% - Red Team    │
├──────────────────┬──────────────────────┤
│  📋 Live Logs    │  📊 Results          │
│  > Site recon... │  Risk: 🔴 RED        │
│  ✓ SDC B, V=115 │  3 CRITICAL          │
│  > Building...   │  2 WARNING           │
│  ✓ 653 nodes     │  [View Report]       │
│  ...             │  [View Diagram]      │
│                  │  [Download JSON]     │
└──────────────────┴──────────────────────┘
```

## Example Bridges (seed these)
1. "3-span continuous steel plate girder over the Kishwaukee River on I-39 in northern Illinois. 315-420-315 ft spans, 7 girders at 9.5' spacing. ILM erection."
2. "Single-span prestressed I-girder bridge, 80 ft span, 5 girders at 8' spacing, rural Tennessee."
3. "2-span concrete box girder over a creek in southern California, 120-120 ft spans."

## File Structure
```
ui/
  server.py          ← FastAPI backend (Track 2)
  static/
    index.html       ← Single-file frontend (Track 1)
  run_manager.py     ← Pipeline runner with log capture (Track 2)
  requirements.txt   ← fastapi, uvicorn
  README.md          ← Launch instructions (Track 3)
```

## Key Constraints
- Python venv at `/tmp/natural-language-builder/.venv` — install deps there
- The CLI is at `src/nlb/cli.py` — import `parse_description` and `run_pipeline` directly
- Output dirs go to `/tmp/natural-language-builder/nlb-run-{run_id}/`
- Moment diagram is an HTML file — serve it in an iframe
- Reports are markdown — render client-side with a simple MD→HTML converter
- No external CDN deps if possible (keep it self-contained), but marked.js from CDN is fine for MD rendering
