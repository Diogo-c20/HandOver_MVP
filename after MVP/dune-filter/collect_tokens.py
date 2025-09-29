from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict

from src.core.logging import setup_logging
from src.core.config import Settings
from src.workers.dune_collect import run_collect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Dune prefilter to collect token candidates (absolute cuts).",
    )
    p.add_argument("--query", required=True, help="검색어 (예: pepe)")
    p.add_argument("--chain", default=None, help="체인 힌트 (예: ETH/BNB/SOL)")
    p.add_argument("--config", dest="config_path", default=None, help="YAML 구성 파일 경로")
    p.add_argument("--preset", default="balanced", help="프리셋 이름 (기본: balanced)")
    p.add_argument("--max-candidates", type=int, default=200)

    # Absolute-cut overrides
    p.add_argument("--min-liq-usd", type=int, default=None)
    p.add_argument("--min-vol24h-usd", type=int, default=None)
    p.add_argument("--min-trades24h", type=int, default=None)
    p.add_argument("--min-unique-traders", type=int, default=None)
    p.add_argument("--min-mcap", type=int, default=None)
    p.add_argument("--max-mcap", type=int, default=None)
    p.add_argument("--min-age-d", type=int, default=None)

    p.add_argument(
        "--base-symbols",
        default=None,
        help="CSV 심볼 (예: USDC,USDT,DAI,FRAX,TUSD,FDUSD,WETH,WBTC,WBNB,SOL,WSOL)",
    )
    p.add_argument(
        "--include-legacy-stables",
        action="store_true",
        help="BUSD 등 레거시 스테이블 허용 (기본 비허용)",
    )

    p.add_argument("--out-dir", default=None, help="출력 폴더 (기본: ~/Desktop/Hyperindex/dune-filter)")
    p.add_argument("--out-name", default=None, help="출력 파일명(확장자 제외). 기본: query 기반")
    p.add_argument("--format", choices=["csv", "md"], default="csv", help="출력 형식")
    p.add_argument("--dry-run", action="store_true", help="SQL만 출력하고 종료")
    return p.parse_args()


def _default_out_dir(arg_dir: str | None) -> str:
    import os
    if arg_dir:
        return arg_dir
    home = os.path.expanduser("~")
    return os.path.join(home, "Desktop", "Hyperindex", "dune-filter")


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    import csv
    fields = [
        "chain",
        "token_address",
        "liq_usd",
        "vol24h_usd",
        "trades24h",
        "unique_traders24h",
        "mcap_usd",
        "age_d",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _write_md(path: str, rows: list[dict[str, Any]]) -> None:
    headers = [
        "chain",
        "token_address",
        "liq_usd",
        "vol24h_usd",
        "trades24h",
        "unique_traders24h",
        "mcap_usd",
        "age_d",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "---|" * len(headers) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n")


def collect_and_save(
    *,
    query: str,
    chain: str | None = None,
    config_path: str | None = None,
    preset: str = "balanced",
    max_candidates: int = 200,
    min_liq_usd: int | None = None,
    min_vol24h_usd: int | None = None,
    min_trades24h: int | None = None,
    min_unique_traders: int | None = None,
    min_mcap: int | None = None,
    max_mcap: int | None = None,
    min_age_d: int | None = None,
    base_symbols: str | None = None,
    include_legacy_stables: bool = False,
    out_dir: str | None = None,
    out_name: str | None = None,
    fmt: str = "csv",
) -> str:
    import os
    from datetime import datetime
    
    # Fallback: ensure .env is loaded if pydantic-settings didn't pick it up
    def _load_dotenv(path: str = ".env") -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass
    _load_dotenv()

    settings = Settings()
    rows, sql = asyncio.run(
        run_collect(
            query=query,
            chain=chain,
            config_path=config_path,
            preset=preset,
            max_candidates=max_candidates,
            min_liq_usd=min_liq_usd,
            min_vol24h_usd=min_vol24h_usd,
            min_trades24h=min_trades24h,
            min_unique_traders=min_unique_traders,
            min_mcap=min_mcap,
            max_mcap=max_mcap,
            min_age_d=min_age_d,
            base_symbols_csv=base_symbols,
            include_legacy_stables=include_legacy_stables,
            api_key=settings.DUNE_API_KEY,
            api_url=settings.DUNE_API_URL,
        )
    )

    out_dir = _default_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = out_name or f"tokens_{query}_{ts}"
    if fmt == "csv":
        path = os.path.join(out_dir, base + ".csv")
        _write_csv(path, rows)
    elif fmt == "md":
        path = os.path.join(out_dir, base + ".md")
        _write_md(path, rows)
    else:
        raise ValueError("fmt must be 'csv' or 'md'")
    return path


def main() -> int:
    args = parse_args()
    setup_logging()

    if args.dry_run:
        settings = Settings()
        # Build SQL only
        rows, sql = asyncio.run(
            run_collect(
                query=args.query,
                chain=args.chain,
                config_path=args.config_path,
                preset=args.preset,
                max_candidates=args.max_candidates,
                min_liq_usd=args.min_liq_usd,
                min_vol24h_usd=args.min_vol24h_usd,
                min_trades24h=args.min_trades24h,
                min_unique_traders=args.min_unique_traders,
                min_mcap=args.min_mcap,
                max_mcap=args.max_mcap,
                min_age_d=args.min_age_d,
                base_symbols_csv=args.base_symbols,
                include_legacy_stables=args.include_legacy_stables,
                api_key=settings.DUNE_API_KEY,
                api_url=settings.DUNE_API_URL,
            )
        )
        print(sql)
        return 0

    path = collect_and_save(
        query=args.query,
        chain=args.chain,
        config_path=args.config_path,
        preset=args.preset,
        max_candidates=args.max_candidates,
        min_liq_usd=args.min_liq_usd,
        min_vol24h_usd=args.min_vol24h_usd,
        min_trades24h=args.min_trades24h,
        min_unique_traders=args.min_unique_traders,
        min_mcap=args.min_mcap,
        max_mcap=args.max_mcap,
        min_age_d=args.min_age_d,
        base_symbols=args.base_symbols,
        include_legacy_stables=args.include_legacy_stables,
        out_dir=args.out_dir,
        out_name=args.out_name,
        fmt=args.format,
    )
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
