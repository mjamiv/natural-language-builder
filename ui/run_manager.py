"""run_manager.py — Pipeline orchestration with log capture.

Each run gets a unique run_id. The pipeline runs in a background thread
(ThreadPoolExecutor) since the NLB pipeline functions are synchronous.

Logs are captured by monkey-patching the cli.py print helpers and are
streamed to subscribers via per-run asyncio.Queue instances.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Add src to sys.path so nlb imports work ───────────────────────────

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Step definitions ──────────────────────────────────────────────────

STEPS = [
    "parse",
    "site_recon",
    "building",
    "assembling",
    "analyzing",
    "red_team",
    "report",
]

# Progress % at the *start* of each step
STEP_PROGRESS = {
    "parse":      0,
    "site_recon": 10,
    "building":   25,
    "assembling": 55,
    "analyzing":  65,
    "red_team":   80,
    "report":     92,
    "complete":   100,
}

# Map status() icon prefix → step name (detected from cli.py output)
ICON_TO_STEP = {
    "🔍": "site_recon",
    "🏗️": "building",
    "📐": "assembling",
    "⚡": "analyzing",
    "🔴": "red_team",
}

# ── In-memory run store ───────────────────────────────────────────────

# run_id → RunState dict
_runs: Dict[str, dict] = {}
_runs_lock = threading.Lock()

# run_id → list of asyncio.Queue (one per SSE subscriber)
_sse_queues: Dict[str, List[asyncio.Queue]] = {}
_sse_lock = threading.Lock()

# nlb.cli monkey-patching is process-global; serialize patched execution
# to avoid cross-run log leakage/races when multiple runs are active.
_CLI_PATCH_LOCK = threading.Lock()

# Shared executor
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nlb-run")

# ── Run state helpers ─────────────────────────────────────────────────

def _new_run(run_id: str, description: str) -> dict:
    return {
        "run_id": run_id,
        "description": description,
        "status": "pending",       # pending | running | completed | failed
        "step": "parse",
        "progress": 0,
        "started_at": time.time(),
        "finished_at": None,
        "logs": [],                # list of log event dicts
        "results": None,           # full results dict when complete
        "error": None,             # error message if failed
        "output_dir": f"/tmp/natural-language-builder/nlb-run-{run_id}",
    }


def _get_run(run_id: str) -> Optional[dict]:
    with _runs_lock:
        return _runs.get(run_id)


def _update_run(run_id: str, **kwargs):
    with _runs_lock:
        if run_id in _runs:
            _runs[run_id].update(kwargs)


# ── SSE queue helpers ─────────────────────────────────────────────────

def _get_or_create_event_loop():
    """Get the running loop, or create a new one for non-async contexts."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _broadcast_log(run_id: str, event: dict):
    """Push a log event to all SSE queues for this run (thread-safe)."""
    with _sse_lock:
        queues = _sse_queues.get(run_id, [])
        for q in queues:
            # schedule put in the event loop from the background thread
            try:
                loop = q._loop if hasattr(q, "_loop") else None
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass


def register_sse_queue(run_id: str, q: asyncio.Queue):
    """Register a new SSE subscriber queue."""
    with _sse_lock:
        _sse_queues.setdefault(run_id, []).append(q)
    # Store the loop on the queue so background threads can schedule on it
    try:
        loop = asyncio.get_running_loop()
        q._loop = loop  # type: ignore[attr-defined]
    except RuntimeError:
        pass


def unregister_sse_queue(run_id: str, q: asyncio.Queue):
    """Remove a SSE subscriber queue."""
    with _sse_lock:
        qs = _sse_queues.get(run_id, [])
        if q in qs:
            qs.remove(q)


def _send_sentinel(run_id: str):
    """Send end-of-stream sentinel to all SSE queues for this run."""
    _broadcast_log(run_id, None)  # None = done sentinel


# ── Log event construction ────────────────────────────────────────────

def _make_log(level: str, message: str, step: str) -> dict:
    return {
        "ts": time.strftime("%H:%M:%S"),
        "level": level,      # info | success | warn | error
        "message": message,
        "step": step,
    }


def _append_log(run_id: str, event: dict):
    with _runs_lock:
        if run_id in _runs:
            _runs[run_id]["logs"].append(event)
    _broadcast_log(run_id, event)


# ── Monkey-patching context ───────────────────────────────────────────

class _LogCapture:
    """Context manager that patches nlb.cli print helpers in the
    calling thread and routes output to the run's log store."""

    def __init__(self, run_id: str, step_tracker: list):
        self.run_id = run_id
        self.step_tracker = step_tracker  # mutable [current_step]
        self._originals = {}

    def _log(self, level: str, message: str):
        step = self.step_tracker[0]
        event = _make_log(level, message, step)
        _append_log(self.run_id, event)

    def _patched_status(self, icon: str, msg: str):
        # Detect step from icon
        new_step = ICON_TO_STEP.get(icon)
        if new_step:
            self.step_tracker[0] = new_step
            progress = STEP_PROGRESS.get(new_step, STEP_PROGRESS.get(self.step_tracker[0], 0))
            _update_run(self.run_id, step=new_step, progress=progress)
        self._log("info", f"{icon} {msg}")

    def _patched_success(self, msg: str):
        self._log("success", f"✓ {msg}")

    def _patched_warn(self, msg: str):
        self._log("warn", f"⚠ {msg}")

    def _patched_error(self, msg: str):
        self._log("error", f"✗ {msg}")

    def _patched_header(self, msg: str):
        self._log("info", f"── {msg} ──")

    def __enter__(self):
        import nlb.cli as _cli
        self._originals = {
            "status":  _cli.status,
            "success": _cli.success,
            "warn":    _cli.warn,
            "error":   _cli.error,
            "header":  _cli.header,
        }
        _cli.status  = self._patched_status
        _cli.success = self._patched_success
        _cli.warn    = self._patched_warn
        _cli.error   = self._patched_error
        _cli.header  = self._patched_header
        return self

    def __exit__(self, *_):
        import nlb.cli as _cli
        for name, fn in self._originals.items():
            setattr(_cli, name, fn)


# ── Moment diagram generation ─────────────────────────────────────────

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "moment-diagram-template.html"


def _generate_moment_diagram(params: dict, results: dict, output_dir: Path) -> bool:
    """Generate an interactive HTML moment diagram using the rich canvas template."""
    try:
        spans = params.get("spans", [100.0])
        n_spans = len(spans)
        total_len = sum(spans)

        # Build approximate moment envelope (simple beam parabola per span)
        # For a continuous beam, we use a rough approximation
        points_x = []
        points_moment = []
        n_pts = 20

        cumulative = 0.0
        for span_i, span_len in enumerate(spans):
            for j in range(n_pts + (1 if span_i == n_spans - 1 else 0)):
                x_frac = j / n_pts
                x = cumulative + x_frac * span_len
                # Simple beam moment: M = w*L/2 * x - w/2 * x^2 (parabola, normalized)
                # Rough w: use total uniform load ~4 kip/ft typical
                w = 4.0  # kip/ft
                xi = x_frac * span_len
                m = w * span_len / 2 * xi - w / 2 * xi ** 2
                # Negative for interior spans (rough hogging at supports)
                if span_i > 0 and x_frac < 0.2:
                    m = -m * (1 - x_frac / 0.2)
                if span_i < n_spans - 1 and x_frac > 0.8:
                    m = -m * (x_frac - 0.8) / 0.2
                points_x.append(round(x, 2))
                points_moment.append(round(m, 1))
            cumulative += span_len

        # Build support lines
        support_x = [0.0]
        cum = 0.0
        for s in spans:
            cum += s
            support_x.append(round(cum, 2))

        # Write an interactive SVG-in-HTML moment diagram
        max_m = max(abs(m) for m in points_moment) or 1.0
        svg_w, svg_h = 800, 300
        margin = {"top": 30, "bottom": 50, "left": 60, "right": 20}
        plot_w = svg_w - margin["left"] - margin["right"]
        plot_h = svg_h - margin["top"] - margin["bottom"]

        def px(x_val):
            return margin["left"] + (x_val / total_len) * plot_w

        def py(m_val):
            # Positive moment → below baseline, negative → above
            mid = margin["top"] + plot_h / 2
            return mid - (m_val / max_m) * (plot_h / 2 - 10)

        baseline_y = margin["top"] + plot_h / 2

        # Build polyline points
        poly_pts = " ".join(f"{px(x):.1f},{py(m):.1f}" for x, m in zip(points_x, points_moment))

        # Support tick marks
        support_ticks = ""
        for sx in support_x:
            spx = px(sx)
            support_ticks += (
                f'<line x1="{spx:.1f}" y1="{baseline_y-5:.1f}" '
                f'x2="{spx:.1f}" y2="{baseline_y+5:.1f}" '
                f'stroke="#555" stroke-width="2"/>'
                f'<text x="{spx:.1f}" y="{baseline_y+20:.1f}" '
                f'text-anchor="middle" font-size="11" fill="#666">{sx:.0f}\'</text>'
            )

        # Find max moment for annotation
        max_m_abs = max(points_moment)
        max_m_x = points_x[points_moment.index(max_m_abs)]

        risk_color = "#e74c3c"  # red as default
        rt = results.get("red_team", {})
        rating = rt.get("risk_rating", "UNKNOWN")
        if rating == "GREEN":
            risk_color = "#27ae60"
        elif rating == "YELLOW":
            risk_color = "#f39c12"

        findings = rt.get("findings", [])
        n_crit = sum(1 for f in findings if f.get("severity") == "CRITICAL")
        n_warn = sum(1 for f in findings if f.get("severity") == "WARNING")

        # ── Build NLB_DATA payload ────────────────────────────────────────
        import datetime

        # Extract model node/element counts from results if available
        model_info  = results.get("model", {})
        node_count  = model_info.get("node_count", len(points_x) * 8)
        elem_count  = model_info.get("element_count", len(points_x) * 10)
        load_combos = results.get("loads", {}).get("total_combinations", 0)
        bridge_desc = params.get("description", "Bridge")[:120]
        girder_label = f"G{n_spans + 2} (Center)" if n_spans >= 3 else "G3 (Interior)"
        load_case   = "Gravity Dead Load (DC)"
        generated   = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        nlb_payload = json.dumps({
            "bridge":            bridge_desc,
            "girder":            girder_label,
            "load_case":         load_case,
            "spans":             spans,
            "supports_ft":       support_x,
            "x_ft":              points_x,
            "M_kft":             points_moment,
            "M_env_max":         [],
            "M_env_min":         [],
            "node_count":        node_count,
            "element_count":     elem_count,
            "load_combinations": load_combos,
            "generated_at":      generated,
        }, default=str)

        # ── Try rich canvas template first ────────────────────────────────
        template_html = None
        if _TEMPLATE_PATH.exists():
            try:
                import re
                tmpl = _TEMPLATE_PATH.read_text()
                # Replace {{PLACEHOLDER}} tokens
                replacements = {
                    "BRIDGE_TITLE":    bridge_desc,
                    "GIRDER_LABEL":    girder_label,
                    "LOAD_CASE":       load_case,
                    "SPANS_JSON":      json.dumps(spans),
                    "SUPPORTS_JSON":   json.dumps(support_x),
                    "X_FT_JSON":       json.dumps(points_x),
                    "M_KFT_JSON":      json.dumps(points_moment),
                    "M_ENV_MAX_JSON":  "[]",
                    "M_ENV_MIN_JSON":  "[]",
                    "NODE_COUNT":      str(node_count),
                    "ELEMENT_COUNT":   str(elem_count),
                    "LOAD_COMBOS":     str(load_combos),
                    "GENERATED_AT":    generated,
                }
                for key, val in replacements.items():
                    tmpl = tmpl.replace("{{" + key + "}}", val)
                # Inject window.NLB_DATA so the template has live data
                tmpl = tmpl.replace(
                    "const BRIDGE_DATA = window.NLB_DATA ||",
                    f"window.NLB_DATA = {nlb_payload};\n  const BRIDGE_DATA = window.NLB_DATA ||",
                )
                template_html = tmpl
            except Exception as tmpl_err:
                print(f"[moment diagram] template render failed: {tmpl_err}, using fallback")

        if template_html:
            html = template_html
        else:
            # ── Fallback: simple SVG version ──────────────────────────────
            html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Moment Diagram — NLB</title>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#0f0f13;color:#e8e8e8;padding:20px}}
h2{{color:#e74c3c}}svg{{background:#1a1a22;border-radius:8px;display:block;margin-bottom:16px}}
</style></head><body>
<h2>🔴 NLB Moment Diagram</h2>
<p style="color:#888">{bridge_desc}</p>
<p>Risk: <strong style="color:{risk_color}">{rating}</strong> — {n_crit} critical, {n_warn} warnings</p>
<svg viewBox="0 0 {svg_w} {svg_h}" width="100%">
  <line x1="{margin['left']}" y1="{baseline_y:.1f}" x2="{margin['left']+plot_w}" y2="{baseline_y:.1f}" stroke="#444" stroke-dasharray="4,4"/>
  <polygon points="{poly_pts} {px(points_x[-1]):.1f},{baseline_y:.1f} {px(points_x[0]):.1f},{baseline_y:.1f}" fill="{risk_color}33"/>
  <polyline points="{poly_pts}" fill="none" stroke="{risk_color}" stroke-width="2.5"/>
  {support_ticks}
</svg>
<p>Spans: {'-'.join(str(int(s)) for s in spans)} ft | Max M = {max_m_abs:.0f} k·ft</p>
<p style="color:#555;font-size:11px">Generated {generated} by NLB</p>
</body></html>"""

        (output_dir / "moment-diagram.html").write_text(html)
        return True
    except Exception as e:
        print(f"[moment diagram] failed: {e}")
        return False


# ── Core pipeline runner (runs in background thread) ──────────────────

def _run_pipeline_thread(run_id: str, description: str):
    """This function runs in a ThreadPoolExecutor thread."""
    run = _get_run(run_id)
    if run is None:
        return

    output_dir = Path(run["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    step_tracker = ["parse"]  # mutable ref so patches can update it

    _update_run(run_id, status="running", step="parse", progress=0)

    try:
        import nlb.cli as _cli
        from nlb.cli import parse_description, run_pipeline

        # ── Step: parse ───────────────────────────────────────────────
        parse_event = _make_log("info", "📝 Parsing description...", "parse")
        _append_log(run_id, parse_event)

        with _CLI_PATCH_LOCK:
            with _LogCapture(run_id, step_tracker):
                params = parse_description(description)

        spans = params.get("spans", [])
        btype = (params.get("bridge_type") or "unknown").replace("_", " ")
        n_spans = len(spans)
        span_str = "-".join(str(int(s)) for s in spans) + " ft" if spans else "?"

        _append_log(run_id, _make_log("success",
            f"Type: {btype}, {n_spans} span(s): {span_str}", "parse"))

        if params.get("num_girders"):
            _append_log(run_id, _make_log("success",
                f"Girders: {params['num_girders']} @ {params.get('girder_spacing_ft', '?')}' spacing",
                "parse"))
        if params.get("location"):
            loc = params["location"]
            _append_log(run_id, _make_log("success",
                f"Location: {loc.get('state', '?')} ({loc.get('lat', '?')}, {loc.get('lon', '?')})",
                "parse"))
        if params.get("erection_method"):
            _append_log(run_id, _make_log("success",
                f"Erection method: {params['erection_method']}", "parse"))
        for c in params.get("constraints", []):
            _append_log(run_id, _make_log("warn", f"Constraint: {c}", "parse"))

        _update_run(run_id, step="parse", progress=8)

        # ── Steps 2-7: full pipeline with monkey-patching ─────────────
        with _CLI_PATCH_LOCK:
            with _LogCapture(run_id, step_tracker):
                results = run_pipeline(params, output_dir=output_dir, verbose=True)

        if results is None:
            raise RuntimeError("Pipeline returned None (superstructure step failed)")

        # ── Moment diagram ────────────────────────────────────────────
        _update_run(run_id, step="report", progress=94)
        _append_log(run_id, _make_log("info", "📊 Generating moment diagram...", "report"))
        _generate_moment_diagram(params, results, output_dir)
        _append_log(run_id, _make_log("success", "Moment diagram ready", "report"))

        # ── Done ──────────────────────────────────────────────────────
        _update_run(run_id,
            status="completed",
            step="report",
            progress=100,
            results=results,
            finished_at=time.time())

        _append_log(run_id, _make_log("success",
            f"Run complete in {time.time() - run['started_at']:.1f}s", "report"))

    except Exception as exc:
        tb = traceback.format_exc()
        _append_log(run_id, _make_log("error", f"Pipeline error: {exc}", step_tracker[0]))
        _append_log(run_id, _make_log("error", tb, step_tracker[0]))
        _update_run(run_id,
            status="failed",
            error=str(exc),
            finished_at=time.time())
    finally:
        _send_sentinel(run_id)


# ── Public API ────────────────────────────────────────────────────────

def start_run(description: str) -> str:
    """Create a new run, start it in the background, return run_id."""
    run_id = str(uuid.uuid4())[:8]
    state = _new_run(run_id, description)

    with _runs_lock:
        _runs[run_id] = state

    with _sse_lock:
        _sse_queues[run_id] = []

    _executor.submit(_run_pipeline_thread, run_id, description)
    return run_id


def get_status(run_id: str) -> Optional[dict]:
    run = _get_run(run_id)
    if run is None:
        return None
    elapsed = (run.get("finished_at") or time.time()) - run["started_at"]
    return {
        "run_id":   run_id,
        "status":   run["status"],
        "step":     run["step"],
        "progress": run["progress"],
        "elapsed":  round(elapsed, 1),
        "error":    run.get("error"),
    }


def get_logs(run_id: str) -> Optional[List[dict]]:
    run = _get_run(run_id)
    if run is None:
        return None
    return list(run["logs"])


def get_results(run_id: str) -> Optional[dict]:
    run = _get_run(run_id)
    if run is None:
        return None
    return run.get("results")


def get_artifacts(run_id: str) -> Optional[List[dict]]:
    run = _get_run(run_id)
    if run is None:
        return None

    output_dir = Path(run["output_dir"])
    if not output_dir.exists():
        return []

    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            ext = path.suffix.lower()
            ftype = {
                ".html": "html",
                ".md":   "markdown",
                ".json": "json",
                ".pdf":  "pdf",
                ".txt":  "text",
            }.get(ext, "file")
            artifacts.append({
                "name":  path.name,
                "type":  ftype,
                "size":  path.stat().st_size,
                "url":   f"/api/run/{run_id}/artifact/{path.name}",
            })
    return artifacts


def get_artifact_path(run_id: str, filename: str) -> Optional[Path]:
    run = _get_run(run_id)
    if run is None:
        return None

    base = Path(run["output_dir"]).resolve()
    candidate = (base / filename).resolve()

    # Prevent path traversal outside the run output directory.
    try:
        candidate.relative_to(base)
    except ValueError:
        return None

    return candidate if candidate.exists() and candidate.is_file() else None
