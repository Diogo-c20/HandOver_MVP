#!/usr/bin/env python3
"""
Sniper Pipeline Runner

hyperindex/snipers 폴더에서 루트의 sniper_ml_pipeline CLI를 그대로 실행합니다.

Usage examples:
  python hyperindex/snipers/sniper_pipeline_runner.py discover --days 90 --chains ethereum
  python hyperindex/snipers/sniper_pipeline_runner.py train --model logreg_l1
  python hyperindex/snipers/sniper_pipeline_runner.py score --input ./dataset/early_trades.parquet --out ./dataset/scored.parquet
"""

import os
import sys


def _load_env_from_file() -> None:
    """Load environment variables from ~/Desktop/Hyperindex/.env if present.

    - Supports optional leading 'export ' per line
    - Ignores comments and blank lines
    - Does not override already-set environment variables
    - Also sets a default SNIPER_CONFIG_PATH if missing
    """
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
                    if (val.startswith("\"") and val.endswith("\"")) or (
                        val.startswith("'") and val.endswith("'")
                    ):
                        val = val[1:-1]
                    os.environ.setdefault(key, val)
        # Ensure default config path if not provided
        os.environ.setdefault(
            "SNIPER_CONFIG_PATH",
            os.path.expanduser("~/Desktop/Hyperindex/snipers/config.yaml"),
        )
    except Exception:
        # Fail-soft: env loading should not block runner
        pass


def _ensure_project_root_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _load_env_from_file()
    _ensure_project_root_on_path()
    from sniper_ml_pipeline import main_cli  # type: ignore
    # Delegate CLI argv as-is
    main_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
