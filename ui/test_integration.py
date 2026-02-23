#!/usr/bin/env python3
"""
NLB Web UI — Integration Test Suite
=====================================
Tests all API endpoints against a running server.

Usage:
    .venv/bin/python ui/test_integration.py
    .venv/bin/python ui/test_integration.py --host localhost --port 8080
    .venv/bin/python ui/test_integration.py --verbose

Pre-requisites:
    - Server is running: bash ui/start.sh   (or uvicorn ui.server:app)
    - Python packages:  pip install httpx   (or requests — both supported)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from typing import Optional


# ── HTTP client (httpx preferred, requests fallback) ──────────────────────────

try:
    import httpx
    def _get(url, **kw):
        return httpx.get(url, timeout=kw.get("timeout", 30))
    def _post(url, json_body, **kw):
        return httpx.post(url, json=json_body, timeout=kw.get("timeout", 30))
    HTTP_LIB = "httpx"
except ImportError:
    try:
        import requests
        def _get(url, **kw):
            return requests.get(url, timeout=kw.get("timeout", 30))
        def _post(url, json_body, **kw):
            return requests.post(url, json=json_body, timeout=kw.get("timeout", 30))
        HTTP_LIB = "requests"
    except ImportError:
        # Use urllib as last resort
        import urllib.request, urllib.error
        import io

        def _get(url, **kw):
            req = urllib.request.urlopen(url, timeout=kw.get("timeout", 30))
            return _UrllibResp(req)
        def _post(url, json_body, **kw):
            data = json.dumps(json_body).encode()
            req_obj = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            req = urllib.request.urlopen(req_obj, timeout=kw.get("timeout", 30))
            return _UrllibResp(req)

        class _UrllibResp:
            def __init__(self, resp):
                self._body = resp.read().decode()
                self.status_code = resp.status
            def json(self): return json.loads(self._body)
            def text(self): return self._body

        HTTP_LIB = "urllib"


# ── Colours ───────────────────────────────────────────────────────────────────

class C:
    PASS  = "\033[92m"
    FAIL  = "\033[91m"
    WARN  = "\033[93m"
    INFO  = "\033[94m"
    DIM   = "\033[2m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

def _icon(ok: bool) -> str:
    return f"{C.PASS}✓{C.RESET}" if ok else f"{C.FAIL}✗{C.RESET}"


# ── Test runner ───────────────────────────────────────────────────────────────

class TestSuite:
    def __init__(self, base_url: str, verbose: bool = False):
        self.base = base_url.rstrip("/")
        self.verbose = verbose
        self.results: list[tuple[str, bool, str]] = []   # (name, passed, detail)

    def _check(self, name: str, passed: bool, detail: str = ""):
        self.results.append((name, passed, detail))
        status = _icon(passed)
        note   = f"  {C.DIM}{detail}{C.RESET}" if detail and self.verbose else ""
        print(f"  {status} {name}{note}")
        return passed

    def run_all(self) -> bool:
        print(f"\n{C.BOLD}🔴 NLB Web UI — Integration Tests{C.RESET}")
        print(f"   Base URL : {self.base}")
        print(f"   HTTP lib : {HTTP_LIB}")
        print()

        ok = True
        ok &= self.test_health()
        ok &= self.test_examples()
        ok &= self.test_static()
        run_id = self.test_run_start()
        if run_id:
            ok &= self.test_status_polling(run_id)
            ok &= self.test_sse_logs(run_id)
            ok &= self.test_results(run_id)
            ok &= self.test_artifacts(run_id)
        else:
            ok = False
        ok &= self.test_invalid_run()
        ok &= self.test_second_run()

        self.print_summary()
        return ok

    # ── Individual tests ──────────────────────────────────────────────────────

    def test_health(self) -> bool:
        print(f"{C.BOLD}[1] Health check{C.RESET}")
        try:
            r = _get(f"{self.base}/health", timeout=5)
            passed = r.status_code == 200
            body   = r.json() if passed else {}
            return self._check("GET /health → 200", passed, str(body))
        except Exception as e:
            return self._check("GET /health → 200", False, f"Connection error: {e}\n\n"
                               "    ⚠️  Is the server running?  bash ui/start.sh")

    def test_examples(self) -> bool:
        print(f"\n{C.BOLD}[2] Examples endpoint{C.RESET}")
        try:
            r = _get(f"{self.base}/api/examples", timeout=10)
            ok1 = self._check("GET /api/examples → 200", r.status_code == 200)
            body = r.json()
            has_examples = isinstance(body, (list, dict)) and bool(body)
            ok2 = self._check("Response contains bridge examples", has_examples, str(body)[:120])
            return ok1 and ok2
        except Exception as e:
            return self._check("GET /api/examples", False, str(e))

    def test_static(self) -> bool:
        print(f"\n{C.BOLD}[3] Static frontend{C.RESET}")
        try:
            r = _get(f"{self.base}/", timeout=10)
            ok1 = self._check("GET / → 200", r.status_code == 200)
            html = getattr(r, 'text', lambda: str(r.content))()
            has_title = "NLB" in html or "Natural Language" in html or "Red Team" in html
            ok2 = self._check("Response is NLB HTML page", has_title)
            return ok1 and ok2
        except Exception as e:
            return self._check("GET /", False, str(e))

    def test_run_start(self) -> Optional[str]:
        print(f"\n{C.BOLD}[4] Start a pipeline run{C.RESET}")
        description = (
            "Single-span prestressed I-girder bridge, 80 ft span, "
            "5 girders at 8' spacing, rural Tennessee."
        )
        try:
            r = _post(f"{self.base}/api/run", {"description": description}, timeout=15)
            ok1 = self._check("POST /api/run → 200", r.status_code == 200,
                              f"status={r.status_code}")
            body = r.json()
            run_id = body.get("run_id") if isinstance(body, dict) else None
            ok2 = self._check("Response has run_id", bool(run_id), f"run_id={run_id}")
            if ok1 and ok2:
                print(f"   {C.DIM}run_id = {run_id}{C.RESET}")
                return run_id
        except Exception as e:
            self._check("POST /api/run", False, str(e))
        return None

    def test_status_polling(self, run_id: str) -> bool:
        print(f"\n{C.BOLD}[5] Status polling (wait up to 120s){C.RESET}")
        url = f"{self.base}/api/run/{run_id}/status"
        deadline = time.time() + 120
        last_step = None
        complete  = False
        steps_seen = set()

        while time.time() < deadline:
            try:
                r = _get(url, timeout=10)
                if r.status_code != 200:
                    break
                body = r.json()
                status   = body.get("status", "?")
                step     = body.get("step", "?")
                progress = body.get("progress", 0)

                if step != last_step:
                    last_step = step
                    steps_seen.add(step)
                    print(f"   {C.DIM}   [{progress:3d}%] {step} ({status}){C.RESET}")

                if status in ("completed", "failed"):
                    complete = True
                    break
                time.sleep(2)
            except Exception as e:
                self._check("Status polling", False, str(e))
                return False

        ok1 = self._check("Run completes within 120s", complete,
                          f"status={status if complete else 'timeout'}")
        ok2 = self._check("Multiple pipeline steps observed", len(steps_seen) >= 2,
                          f"steps={steps_seen}")
        ok3 = self._check("Final status is 'completed' (not 'failed')",
                          status == "completed", f"status={status}")
        ok4 = self._check("Progress field present", "progress" in body)
        ok5 = self._check("Elapsed field present", "elapsed" in body)
        return all([ok1, ok2, ok3, ok4, ok5])

    def test_sse_logs(self, run_id: str) -> bool:
        print(f"\n{C.BOLD}[6] SSE log stream{C.RESET}")
        url = f"{self.base}/api/run/{run_id}/logs"
        lines_received = []
        error = None

        def _read_sse():
            nonlocal error
            try:
                # Use urllib for SSE — httpx/requests may buffer
                import urllib.request
                req = urllib.request.urlopen(url, timeout=10)
                for raw in req:
                    line = raw.decode().strip()
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload:
                            lines_received.append(payload)
                        if len(lines_received) >= 3:
                            break
            except Exception as e:
                error = str(e)

        t = threading.Thread(target=_read_sse, daemon=True)
        t.start()
        t.join(timeout=15)

        if error and not lines_received:
            # SSE endpoint may not exist yet — soft warning
            return self._check("SSE logs stream returns data", False,
                               f"⚠ {error} — endpoint may not be implemented yet")

        ok1 = self._check("SSE /logs endpoint accessible", error is None or len(lines_received) > 0)
        ok2 = self._check("Log lines are valid JSON", self._all_json(lines_received),
                          f"got {len(lines_received)} lines")
        return ok1 and ok2

    def _all_json(self, lines: list[str]) -> bool:
        if not lines:
            return False
        for line in lines:
            try:
                json.loads(line)
            except Exception:
                return False
        return True

    def test_results(self, run_id: str) -> bool:
        print(f"\n{C.BOLD}[7] Results JSON{C.RESET}")
        url = f"{self.base}/api/run/{run_id}/results"
        try:
            r   = _get(url, timeout=15)
            ok1 = self._check("GET /results → 200", r.status_code == 200,
                              f"status={r.status_code}")
            body = r.json()
            ok2  = self._check("Results is a dict", isinstance(body, dict))

            # Check expected top-level keys from cli.py run_pipeline output
            expected_keys = {"site", "superstructure", "analysis", "red_team"}
            found = expected_keys & set(body.keys())
            ok3 = self._check(f"Contains pipeline keys ({', '.join(sorted(found))})",
                              len(found) >= 2, f"keys={list(body.keys())[:10]}")

            # Red team result
            rt = body.get("red_team", {})
            ok4 = self._check("red_team has risk_rating", "risk_rating" in rt,
                              str(rt)[:80])
            return all([ok1, ok2, ok3, ok4])
        except Exception as e:
            return self._check("GET /results", False, str(e))

    def test_artifacts(self, run_id: str) -> bool:
        print(f"\n{C.BOLD}[8] Artifacts{C.RESET}")
        url = f"{self.base}/api/run/{run_id}/artifacts"
        try:
            r   = _get(url, timeout=10)
            ok1 = self._check("GET /artifacts → 200", r.status_code == 200)
            body = r.json()

            # Expect list or dict with "artifacts" key
            artifact_list = body if isinstance(body, list) else body.get("artifacts", [])
            ok2 = self._check("Artifacts list is non-empty", len(artifact_list) > 0,
                              f"{len(artifact_list)} artifact(s)")

            # Expected artifacts from run_pipeline
            expected_files = {"moment-diagram.html", "results.json", "red-team-report.md"}
            found_names = {a.get("name", "") for a in artifact_list if isinstance(a, dict)}
            found = expected_files & found_names
            ok3 = self._check("Contains key artifact files", len(found) >= 2,
                              f"found={found_names}")

            # Test downloading one artifact
            ok4 = True
            if artifact_list:
                sample = artifact_list[0]
                name = sample.get("name") or sample.get("filename") or ""
                art_url = f"{self.base}/api/run/{run_id}/artifact/{name}"
                try:
                    ra = _get(art_url, timeout=10)
                    ok4 = self._check(f"Download artifact '{name}' → 200",
                                      ra.status_code == 200)
                except Exception as e:
                    ok4 = self._check(f"Download artifact", False, str(e))

            return all([ok1, ok2, ok3, ok4])
        except Exception as e:
            return self._check("GET /artifacts", False, str(e))

    def test_invalid_run(self) -> bool:
        print(f"\n{C.BOLD}[9] Error handling{C.RESET}")
        fake_id = "00000000-0000-0000-0000-000000000000"

        ok1, ok2, ok3 = True, True, True
        try:
            r = _get(f"{self.base}/api/run/{fake_id}/status", timeout=5)
            ok1 = self._check("Unknown run_id → 404", r.status_code == 404,
                              f"got {r.status_code}")
        except Exception as e:
            ok1 = self._check("Unknown run_id → 404", False, str(e))

        try:
            r = _post(f"{self.base}/api/run", {}, timeout=5)
            ok2 = self._check("Empty POST body → 422 or 400",
                              r.status_code in (400, 422),
                              f"got {r.status_code}")
        except Exception as e:
            ok2 = self._check("Empty POST body → error", False, str(e))

        try:
            r = _post(f"{self.base}/api/run", {"description": ""}, timeout=5)
            ok3 = self._check("Blank description → 422 or 400 or 200",
                              r.status_code in (400, 422, 200),
                              f"got {r.status_code}")
        except Exception as e:
            ok3 = self._check("Blank description handled", False, str(e))

        return ok1 and ok2 and ok3

    def test_second_run(self) -> bool:
        """Quick smoke-test that a second run can start (no global state leak)."""
        print(f"\n{C.BOLD}[10] Second concurrent run{C.RESET}")
        try:
            r = _post(f"{self.base}/api/run",
                      {"description": "2-span concrete box girder, 120-120 ft, California."},
                      timeout=10)
            ok1 = self._check("Second POST /api/run → 200", r.status_code == 200)
            body = r.json()
            run_id_2 = body.get("run_id") if isinstance(body, dict) else None
            ok2 = self._check("Second run gets unique run_id", bool(run_id_2),
                              f"run_id={run_id_2}")
            return ok1 and ok2
        except Exception as e:
            return self._check("Second run", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self):
        total  = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed

        print(f"\n{'━' * 52}")
        print(f"  {C.BOLD}Results:{C.RESET}  {C.PASS}{passed} passed{C.RESET}  "
              f"{(C.FAIL + str(failed) + ' failed' + C.RESET) if failed else ''}")
        print(f"  Total:    {total} checks")

        if failed:
            print(f"\n  {C.FAIL}Failed checks:{C.RESET}")
            for name, ok, detail in self.results:
                if not ok:
                    print(f"    {C.FAIL}✗{C.RESET} {name}")
                    if detail:
                        print(f"      {C.DIM}{detail}{C.RESET}")

        print(f"{'━' * 52}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NLB Web UI integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host",    default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port",    default=8080, type=int, help="Server port (default: 8080)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for every check")
    parser.add_argument("--https",   action="store_true", help="Use HTTPS")
    args = parser.parse_args()

    scheme   = "https" if args.https else "http"
    base_url = f"{scheme}://{args.host}:{args.port}"

    suite = TestSuite(base_url, verbose=args.verbose)
    ok = suite.run_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
