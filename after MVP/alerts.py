#!/usr/bin/env python3
"""
Lightweight local alerts utility (no Slack dependency).

Usage:
  from alerts import notify
  notify("Artifacts stale → retrain", level="info", source="sniper", extra={"freshness_days": 30})

Behavior:
  - Appends JSONL to ALERT_LOG_PATH (default: ~/Desktop/Hyperindex/alerts.log)
  - Prints to stdout when ALERT_ECHO=1
  - Designed to be safe in restricted environments (no network calls)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _log_path() -> Path:
    p = os.getenv("ALERT_LOG_PATH") or os.path.expanduser("~/Desktop/Hyperindex/alerts.log")
    return Path(p)


def notify(message: str, level: str = "info", source: str = "pipeline", extra: Optional[Dict[str, Any]] = None) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": str(level or "info").lower(),
        "source": str(source or "pipeline").lower(),
        "message": str(message or "")[:2000],
        "extra": extra or {},
    }
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        if os.getenv("ALERT_ECHO", "1") == "1":
            print(f"[alert][{rec['level']}][{rec['source']}] {rec['message']}")
            if rec["extra"]:
                print(f"  extra: {json.dumps(rec['extra'], ensure_ascii=False)}")
    except Exception:
        pass


if __name__ == "__main__":
    # Simple CLI: echo a test alert
    level = sys.argv[1] if len(sys.argv) > 1 else "info"
    msg = sys.argv[2] if len(sys.argv) > 2 else "test alert"
    src = sys.argv[3] if len(sys.argv) > 3 else "pipeline"
    notify(msg, level=level, source=src)

