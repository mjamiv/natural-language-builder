"""server.py — FastAPI backend for the NLB Web UI.

Endpoints
---------
POST  /api/run                          Start a new analysis run
GET   /api/run/{run_id}/status          Poll current status/progress
GET   /api/run/{run_id}/logs            SSE stream of log events
GET   /api/run/{run_id}/results         Full results JSON (when complete)
GET   /api/run/{run_id}/artifacts       List output files
GET   /api/run/{run_id}/artifact/{fn}   Serve an artifact file
GET   /api/examples                     Example bridge descriptions
GET   /                                 Serve static/index.html
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from sse_starlette.sse import EventSourceResponse
    _HAS_SSE_STARLETTE = True
except ImportError:
    _HAS_SSE_STARLETTE = False

import ui.run_manager as rm

# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(title="NLB Web UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (served AFTER api routes) ────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)


# ── Example bridges ───────────────────────────────────────────────────

EXAMPLES = [
    {
        "id": "kishwaukee",
        "label": "Kishwaukee River — I-39 (3-span steel)",
        "description": (
            "3-span continuous steel plate girder over the Kishwaukee River on I-39 "
            "in northern Illinois. 315-420-315 ft spans, 7 girders at 9.5' spacing. "
            "ILM erection. No equipment in the river during construction."
        ),
    },
    {
        "id": "rural_tennessee",
        "label": "Rural Tennessee — 80 ft prestressed I-girder",
        "description": (
            "Single-span prestressed I-girder bridge, 80 ft span, 5 girders at 8' spacing, "
            "rural Tennessee. Creek crossing."
        ),
    },
    {
        "id": "california_box",
        "label": "Southern California — 2-span concrete box girder",
        "description": (
            "2-span cast-in-place concrete box girder over a creek in southern California, "
            "120-120 ft spans. Seismic Zone D."
        ),
    },
]


# ── Request/response models ───────────────────────────────────────────

class RunRequest(BaseModel):
    description: str


# ── API routes ────────────────────────────────────────────────────────

@app.get("/api/examples")
def get_examples():
    return {"examples": EXAMPLES}


@app.post("/api/run")
def post_run(body: RunRequest):
    if not body.description.strip():
        raise HTTPException(status_code=400, detail="description is required")
    run_id = rm.start_run(body.description.strip())
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/status")
def get_run_status(run_id: str):
    status = rm.get_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return status


@app.get("/api/run/{run_id}/results")
def get_run_results(run_id: str):
    status = rm.get_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if status["status"] == "running" or status["status"] == "pending":
        raise HTTPException(status_code=202, detail="Run still in progress")
    results = rm.get_results(run_id)
    if results is None:
        raise HTTPException(status_code=404, detail="No results available (run may have failed)")
    return results


@app.get("/api/run/{run_id}/artifacts")
def get_run_artifacts(run_id: str):
    artifacts = rm.get_artifacts(run_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"artifacts": artifacts}


@app.get("/api/run/{run_id}/artifact/{filename:path}")
def get_artifact_file(run_id: str, filename: str):
    path = rm.get_artifact_path(run_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found")

    # Guess MIME type
    mime, _ = mimetypes.guess_type(filename)
    if mime is None:
        mime = "application/octet-stream"

    # For markdown, serve as text so browser can render it
    if filename.endswith(".md"):
        mime = "text/plain; charset=utf-8"

    return FileResponse(str(path), media_type=mime, filename=filename)


@app.get("/api/run/{run_id}/logs")
async def get_run_logs_sse(run_id: str, request: Request):
    """Server-Sent Events stream of log events for a run.

    Sends already-buffered logs immediately, then streams new events
    as they arrive, and closes when the run completes (sentinel = None).
    """
    status = rm.get_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    async def event_generator():
        # 1. Drain existing logs first (for late-joining clients)
        existing = rm.get_logs(run_id) or []
        for log_event in existing:
            yield {"data": json.dumps(log_event)}
            await asyncio.sleep(0)

        # If run already finished, we're done
        current_status = rm.get_status(run_id)
        if current_status and current_status["status"] in ("completed", "failed"):
            yield {"data": json.dumps({"level": "info", "message": "__END__", "step": "done", "ts": ""})}
            return

        # 2. Register a queue for live events
        q: asyncio.Queue = asyncio.Queue()
        rm.register_sse_queue(run_id, q)

        try:
            while True:
                # Client disconnect check
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send a keepalive comment
                    yield {"comment": "keepalive"}
                    continue

                if event is None:
                    # Sentinel — run finished
                    yield {"data": json.dumps({"level": "info", "message": "__END__", "step": "done", "ts": ""})}
                    break

                yield {"data": json.dumps(event)}
        finally:
            rm.unregister_sse_queue(run_id, q)

    if _HAS_SSE_STARLETTE:
        return EventSourceResponse(event_generator())
    else:
        # Fallback: plain text/event-stream
        async def plain_stream():
            async for chunk in event_generator():
                if "data" in chunk:
                    yield f"data: {chunk['data']}\n\n"
                elif "comment" in chunk:
                    yield f": {chunk['comment']}\n\n"

        return StreamingResponse(
            plain_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


# ── Static file serving (index.html at root) ──────────────────────────

@app.get("/")
@app.get("/index.html")
async def serve_index():
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            status_code=200,
            content={"message": "NLB backend running. Place index.html in ui/static/"},
        )
    return FileResponse(str(index))


# Mount static files AFTER the api routes so /api/* takes priority
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Dev entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.server:app", host="0.0.0.0", port=8080, reload=True)
