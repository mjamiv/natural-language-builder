# 🔴 NLB Web UI

A browser-based front-end for the **Natural Language Builder** (NLB) pipeline — type a bridge description, click **Run Analysis**, and watch the red-team report, moment diagram, and raw results appear in real time.

---

## What It Does

NLB takes a plain-English bridge description, runs it through a full structural pipeline (site recon → FEA model → load generation → OpenSeesPy analysis → adversarial red-team), and surfaces findings ranked by severity. The web UI streams live progress logs via SSE and renders the outputs (moment diagram, technical report, executive summary, raw JSON) without any page reload.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check: `python3 --version` |
| Virtual env | Already at `/tmp/natural-language-builder/.venv` |
| NLB source | At `/tmp/natural-language-builder/src/` |

---

## Install Dependencies

```bash
cd /tmp/natural-language-builder
.venv/bin/pip install fastapi "uvicorn[standard]" sse-starlette
```

> These are the only additions needed — the NLB pipeline deps are already installed.

---

## Start the Server

```bash
cd /tmp/natural-language-builder
.venv/bin/python -m uvicorn ui.server:app --host 0.0.0.0 --port 8080
```

Or use the launch script (recommended):

```bash
bash ui/start.sh
```

---

## Access the UI

Open your browser to: **http://localhost:8080**

---

## What to Try First

Paste the **I-39 Kishwaukee River** example (pre-loaded in the UI dropdown):

```
3-span continuous steel plate girder over the Kishwaukee River on I-39
in northern Illinois. 315-420-315 ft spans, 7 girders at 9.5' spacing.
ILM erection. No equipment in the river channel.
```

1. Select it from the **Examples** dropdown (or paste it manually)
2. Click **▶ Run Analysis**
3. Watch the **live log stream** on the left as each pipeline stage completes
4. When the progress bar hits 100%, the **Results** panel on the right populates:
   - 🔴/🟡/🟢 Risk badge with finding counts
   - **View Moment Diagram** — opens the interactive SVG/canvas chart in a panel
   - **View Technical Report** — rendered Markdown (full 10-section red-team report)
   - **View Executive Summary** — one-page owner brief
   - **Download JSON** — complete raw results for downstream use

---

## API Endpoints (Quick Reference)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/run` | Start a pipeline run; returns `{ run_id }` |
| `GET` | `/api/run/{id}/status` | Poll status, step, progress (0–100) |
| `GET` | `/api/run/{id}/logs` | SSE stream of live log lines |
| `GET` | `/api/run/{id}/results` | Full results JSON when complete |
| `GET` | `/api/run/{id}/artifacts` | List output files |
| `GET` | `/api/run/{id}/artifact/{file}` | Serve a specific artifact |
| `GET` | `/api/examples` | Seed bridge descriptions |
| `GET` | `/health` | Liveness probe |

---

## Known Limitations

- **OpenSeesPy required for real FEA** — if not installed, the pipeline falls back to mock analysis results. Moment diagrams still render; DCR values and reactions will be zero/placeholder.
- **Site recon requires internet** — the pipeline calls Nominatim/OSM and USGS APIs. Offline runs use conservative defaults automatically.
- **Single-process server** — runs are queued in-memory; restarting the server loses all in-flight run state. Artifacts on disk survive.
- **No auth** — this is a local dev tool. Do not expose port 8080 publicly.
- **Markdown rendering** — uses `marked.js` from CDN; requires internet access for the first load. Subsequent loads use the cached version.
- **Run output dirs** live at `/tmp/natural-language-builder/nlb-run-{run_id}/` — cleared on OS reboot.

---

## File Structure

```
ui/
├── server.py              ← FastAPI backend
├── run_manager.py         ← Pipeline runner with log capture
├── start.sh               ← One-command launcher
├── test_integration.py    ← Integration test suite
├── README.md              ← This file
├── static/
│   └── index.html         ← Single-file frontend (no build step)
└── templates/
    └── moment-diagram-template.html  ← Diagram HTML template
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
→ Run the install step above; confirm you're using `.venv/bin/python`.

**`ModuleNotFoundError: No module named 'nlb'`**
→ The server must be started from `/tmp/natural-language-builder/` (not `ui/`) so that `src/` is on the path. Use `start.sh` or the `uvicorn` command shown above.

**SSE logs not streaming**
→ Some browsers buffer SSE behind a proxy. Try `http://127.0.0.1:8080` directly (not via a reverse proxy or VS Code port forward).

**Port 8080 already in use**
→ Change `--port 8080` to any free port and update `http://localhost:<port>`.
