#!/usr/bin/env python3
"""
Real-Data Sniper Pipeline Runner (one-shot)

Runs: fetch_real_data -> weak-label -> train (logreg_l1)
Uses the same config/env conventions as sniper_pipeline_runner.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional, List


def _load_env_from_file() -> None:
    env_path = os.path.expanduser("~/Desktop/Hyperindex/.env")
    try:
        if os.path.isfile(env_path):
            with open(env_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export ") :].strip()
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        os.environ.setdefault(
            "SNIPER_CONFIG_PATH",
            os.path.expanduser("~/Desktop/Hyperindex/snipers/config.yaml"),
        )
    except Exception:
        pass


def _ensure_project_root_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)


def run_fetch(config_path: str) -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    fetcher = os.path.join(here, "fetch_real_data.py")
    cmd = [sys.executable, fetcher, "--config", config_path]
    subprocess.run(cmd, check=True)


def run_cli(argv: Optional[List[str]] = None) -> None:
    _ensure_project_root_on_path()
    from sniper_ml_pipeline import main_cli  # type: ignore
    main_cli(argv)


def main() -> None:
    _load_env_from_file()
    ap = argparse.ArgumentParser(description="Run real-data pipeline: fetch -> weak-label -> train")
    ap.add_argument("--config", type=str, default=os.getenv("SNIPER_CONFIG_PATH"))
    ap.add_argument("--model", type=str, default="logreg_l1", choices=["logreg_l1", "xgboost"])
    ap.add_argument("--no-fetch", action="store_true", help="Skip fetch_real_data step")
    ap.add_argument("--snorkel", type=int, default=0, help="Use snorkel if available (1) or fallback (0)")
    args = ap.parse_args()

    assert args.config, "--config or SNIPER_CONFIG_PATH is required"

    if not args.no_fetch:
        print("[1/3] Fetching real data via Dune...")
        run_fetch(args.config)

    print("[2/3] Building weak labels...")
    run_cli(["--config", args.config, "weak-label", "--snorkel", str(args.snorkel)])

    print("[3/3] Training model...")
    run_cli(["--config", args.config, "train", "--model", args.model])


if __name__ == "__main__":
    main()

