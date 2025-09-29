from __future__ import annotations

import argparse
import json
import sys
from typing import List

from dev_risk import compute_developer_subscore


def _read_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Developer risk subscore batch")
    parser.add_argument("--chain", required=True, choices=["eth", "bnb", "sol"], help="chain code")
    parser.add_argument("--tokens", required=True, help="file with token addresses (one per line)")
    args = parser.parse_args()

    tokens = _read_tokens(args.tokens)
    rows = []
    for t in tokens:
        res = compute_developer_subscore(t, args.chain)
        print(json.dumps(res, ensure_ascii=False))  # JSON line per token
        rows.append((t, res.get("dev_holdings_pct", 0.0), res.get("dev_risk_score", 0), res.get("developer_subscore", 0.0)))

    # Summary table
    if rows:
        print("\nSummary:")
        print(f"{'Token':<44} {'Dev_Hold%':>10} {'Dev_Risk':>9} {'f(Dev)':>8}")
        for (t, h, r, fdev) in rows:
            print(f"{t:<44} {h:10.2f} {int(r):9d} {fdev:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

