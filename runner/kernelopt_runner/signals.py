"""Graphsignal /signals reader (attribution only — never gate timings).

Reads http://127.0.0.1:<port>/signals from within the profiled process after a
short settle (CUPTI flushes on ~1 s ticks; SKILL.md recommends ~3 s). The
endpoint only exists while the profiled workload is alive, so we must read
before exit.
"""
from __future__ import annotations

import json
import time
import urllib.request


def read_signals(port: int = 18259, settle_s: float = 3.0, timeout_s: float = 10.0) -> dict:
    """Read /signals; returns {} on any failure (profiling is best-effort)."""
    if settle_s > 0:
        time.sleep(settle_s)
    deadline = time.time() + timeout_s
    last_err = None
    url = f"http://127.0.0.1:{port}/signals"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(0.5)
    return {"error": f"could not read {url}: {last_err}"}
