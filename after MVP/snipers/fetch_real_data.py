#!/usr/bin/env python3
"""
Fetch real data from Dune Saved Queries and write into the pipeline's expected files.
- Reads IDs and paths from config.yaml (SNIPER_CONFIG_PATH or --config)
- Writes:
  - cache/discovered_pairs.parquet (t0 discovery)
  - dataset/early_trades.parquet (early trades window)
Requires: requests, pandas, pyyaml, (pyarrow or fastparquet optional)
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml

API_BASE = "https://api.dune.com/api/v1"


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dune_headers(api_key: str) -> Dict[str, str]:
    return {"X-Dune-Api-Key": api_key, "Content-Type": "application/json"}


def dune_execute(saved_query_id: int, params: Dict[str, Any], api_key: str) -> str:
    url = f"{API_BASE}/query/{saved_query_id}/execute"
    r = requests.post(url, headers=dune_headers(api_key), json={"query_parameters": params or {}})
    r.raise_for_status()
    data = r.json()
    # v1 returns execution_id under different keys depending on endpoint version
    exec_id = data.get("execution_id") or data.get("id") or data.get("executionId")
    if not exec_id:
        raise RuntimeError(f"No execution_id in response: {data}")
    return str(exec_id)


def dune_poll_results(execution_id: str, api_key: str, poll_sec: int, max_poll_sec: int) -> List[Dict[str, Any]]:
    status_url = f"{API_BASE}/execution/{execution_id}/status"
    results_url = f"{API_BASE}/execution/{execution_id}/results"
    deadline = time.time() + max_poll_sec
    state = "PENDING"
    while time.time() < deadline:
        s = requests.get(status_url, headers=dune_headers(api_key))
        s.raise_for_status()
        st = s.json()
        state = st.get("state") or st.get("execution_state") or ""
        if state in {"QUERY_STATE_COMPLETED", "COMPLETED", "SUCCESS"}:
            res = requests.get(results_url, headers=dune_headers(api_key), params={"limit": 50000})
            res.raise_for_status()
            body = res.json()
            rows = body.get("result", {}).get("rows") or body.get("rows") or []
            return rows
        if state in {"FAILED", "QUERY_STATE_FAILED"}:
            raise RuntimeError(f"Dune execution failed: {st}")
        time.sleep(poll_sec)
    raise TimeoutError(f"Dune execution timed out (last_state={state})")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def fetch_t0(cfg: Dict[str, Any], api_key: str) -> pd.DataFrame:
    qid = int(cfg["discover"]["query_id_discover"])
    poll = int(cfg.get("dune", {}).get("poll_interval_sec", 2))
    maxp = int(cfg.get("dune", {}).get("max_poll_sec", 180))
    params = {}  # saved query can be parameterized later
    exec_id = dune_execute(qid, params, api_key)
    rows = dune_poll_results(exec_id, api_key, poll, maxp)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("t0 query returned 0 rows")
    df = normalize_columns(df)
    # Required columns: chain, token_address, pair_address, t0, dex_type, tvl
    # Convert block_time/t0 to epoch seconds if needed
    if "t0" in df.columns and not pd.api.types.is_integer_dtype(df["t0"]):
        try:
            df["t0"] = pd.to_datetime(df["t0"]).astype("int64") // 10**9
        except Exception:
            pass
    # tvl may be named differently; ensure float
    for col in ["tvl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing optional columns
    if "mcap_bin" not in df.columns:
        df["mcap_bin"] = "micro"
    return df[[c for c in ["chain","token_address","pair_address","t0","dex_type","tvl","mcap_bin"] if c in df.columns]]


def fetch_early(cfg: Dict[str, Any], api_key: str) -> pd.DataFrame:
    qid = int(cfg["discover"]["query_id_early_trades"])
    poll = int(cfg.get("dune", {}).get("poll_interval_sec", 2))
    maxp = int(cfg.get("dune", {}).get("max_poll_sec", 180))
    params = {}
    exec_id = dune_execute(qid, params, api_key)
    rows = dune_poll_results(exec_id, api_key, poll, maxp)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("early-trades query returned 0 rows")
    df = normalize_columns(df)
    # Expected cols: chain, token_address, pair_address, block_time, t0, tx_hash, wallet, dex_type, usd_value
    # Normalize types
    for col in ["block_time", "t0"]:
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col]).astype("int64") // 10**9
            except Exception:
                pass
    if "usd_value" in df.columns:
        df["usd_value"] = pd.to_numeric(df["usd_value"], errors="coerce")
    # Derive same_block_swaps (approx by same second)
    if "block_time" in df.columns:
        df["same_block_swaps"] = df.groupby(["token_address", "block_time"]).tx_hash.transform("count").fillna(1).astype(int)
    else:
        df["same_block_swaps"] = 1
    # Side is buy by construction
    df["side"] = "buy"
    return df


def main() -> None:
    def load_env_from_file() -> None:
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

    load_env_from_file()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.getenv("SNIPER_CONFIG_PATH"))
    args = ap.parse_args()
    assert args.config, "--config or SNIPER_CONFIG_PATH is required"
    cfg = load_config(args.config)
    api_key = os.getenv("DUNE_API_KEY")
    assert api_key, "DUNE_API_KEY is required"

    paths = cfg["paths"]
    cache_dir = ensure_dir(paths["cache_dir"])  # type: ignore
    dataset_dir = ensure_dir(paths["dataset_dir"])  # type: ignore

    print("Fetching t0 discovery...")
    t0_df = fetch_t0(cfg, api_key)
    disc_path = Path(paths["cache_dir"]) / "discovered_pairs.parquet"
    save_table(t0_df, disc_path)
    print(f"Saved discovery: {disc_path}")

    print("Fetching early trades...")
    early_df = fetch_early(cfg, api_key)
    # Join TVL and t0 if present in t0_df
    if not t0_df.empty:
        join_cols = [c for c in ["tvl", "dex_type", "t0"] if c in t0_df.columns]
        if join_cols:
            early_df = early_df.merge(t0_df[["token_address"] + join_cols].drop_duplicates("token_address"), on="token_address", how="left")
    # Coalesce duplicate columns after merge (e.g., t0_x/t0_y, dex_type_x/dex_type_y)
    for base in ["t0", "dex_type"]:
        bx, by = f"{base}_x", f"{base}_y"
        if base not in early_df.columns:
            if bx in early_df.columns and by in early_df.columns:
                early_df[base] = early_df[bx].where(early_df[bx].notna(), early_df[by])
            elif bx in early_df.columns:
                early_df[base] = early_df[bx]
            elif by in early_df.columns:
                early_df[base] = early_df[by]
    # Normalize types
    if "t0" in early_df.columns and not pd.api.types.is_integer_dtype(early_df["t0"]):
        try:
            early_df["t0"] = pd.to_datetime(early_df["t0"]).astype("int64") // 10**9
        except Exception:
            early_df["t0"] = pd.to_numeric(early_df["t0"], errors="coerce").fillna(0).astype(int)
    # t_since_t0_sec
    if "t0" in early_df.columns and "block_time" in early_df.columns:
        early_df["t_since_t0_sec"] = early_df["block_time"].astype(int) - early_df["t0"].astype(int)
    # Defaults for missing optional fields
    for col, val in {
        "gas_price_gwei": 0.0,
        "priority_fee_gwei": 0.0,
        "router": "default",
        "is_contract_caller": 0,
        "tvl_quartile": "Q2",
        "mcap_bin": "micro",
        "exclusion_bucket": None,
    }.items():
        if col not in early_df.columns:
            early_df[col] = val

    ds_path = Path(paths["dataset_dir"]) / "early_trades.parquet"
    save_table(early_df, ds_path)
    print(f"Saved early trades: {ds_path} ({len(early_df)} rows)")


if __name__ == "__main__":
    main()
