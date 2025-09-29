"""
Sniper ML Pipeline (CPU-only, single file)

Overview:
- Discover early trades for new meme coins, build features, apply weak labeling,
  train a simple model (LogReg L1, optional XGBoost), export artifacts, and
  provide a runtime scorer usable by main_pipeline.py.

Quickstart (demo; runs without API keys using stubs and synthetic data):
  python sniper_ml_pipeline.py discover --days 90 --chains ethereum bnb solana
  python sniper_ml_pipeline.py build-dataset --from-cache
  python sniper_ml_pipeline.py weak-label --snorkel 0
  python sniper_ml_pipeline.py train --model logreg_l1
  python sniper_ml_pipeline.py export-config --out ./artifacts
  python sniper_ml_pipeline.py score --input ./dataset/early_trades.parquet --out ./dataset/scored.parquet
  python sniper_ml_pipeline.py maybe-update-weights --stale-days 30

Adapter for main_pipeline.py:
  from sniper_ml_pipeline import get_sniper_scorer, maybe_update_weights
  maybe_update_weights("./config.yaml", freshness_days=30)
  scorer = get_sniper_scorer("./artifacts")
  scored = scorer.score_df(early_trades_features_df)

Notes:
- External APIs are stubbed; the pipeline remains functional without keys.
- Optional deps: xgboost, snorkel. If missing, pipeline auto-falls back.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import yaml  # type: ignore
except Exception:  # Fallback shim if PyYAML is unavailable
    class _YAMLShim:
        @staticmethod
        def safe_load(stream):
            if hasattr(stream, "read"):
                data = stream.read()
            else:
                data = stream
            try:
                return json.loads(data)
            except Exception:
                # Minimal parser: return empty dict on failure
                return {}

        @staticmethod
        def safe_dump(obj, stream=None, sort_keys=False):
            text = json.dumps(obj, indent=2)
            if stream is None:
                return text
            if hasattr(stream, "write"):
                stream.write(text)
            else:
                with open(stream, "w") as f:
                    f.write(text)

    yaml = _YAMLShim()  # type: ignore
try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except Exception:
    def retry(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco
    def stop_after_attempt(*args, **kwargs):  # type: ignore
        return None
    def wait_exponential(*args, **kwargs):  # type: ignore
        return None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

# Optional imports (graceful degrade)
try:  # xgboost is optional
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:  # ImportError or any other failure
    XGBOOST_AVAILABLE = False

try:  # snorkel is optional
    from snorkel.labeling.model import LabelModel  # type: ignore
    SNORKEL_AVAILABLE = True
except Exception:
    SNORKEL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False
try:
    # Optional local alerts (no Slack)
    from alerts import notify as local_notify  # type: ignore
    LOCAL_ALERTS_AVAILABLE = True
except Exception:
    LOCAL_ALERTS_AVAILABLE = False
try:
    # Prefer insiders' monitoring sink if available (to mirror insiders model)
    from insiders.monitoring import DefaultAlertSink as _InsidersDefaultSink  # type: ignore
    INSIDERS_ALERT_AVAILABLE = True
except Exception:
    INSIDERS_ALERT_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

    # Minimal metric fallbacks
    def precision_recall_curve(y_true, scores):  # type: ignore
        y = np.asarray(y_true).astype(int)
        s = np.asarray(scores)
        order = np.argsort(-s)
        y = y[order]
        s = s[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / np.maximum(tp[-1] if tp.size > 0 else 1, 1)
        thresholds = s[np.r_[np.where(np.diff(s) < 0)[0], len(s) - 1]] if len(s) else np.array([0.5])
        # Make lengths align like sklearn (precision, recall length = len(thr)+1). Simplify:
        return np.r_[precision, precision[-1] if len(precision) else 1.0], np.r_[recall, 0.0], thresholds

    def average_precision_score(y_true, scores):  # type: ignore
        # Trapezoidal area under PR curve (approx)
        p, r, _ = precision_recall_curve(y_true, scores)
        ap = 0.0
        for i in range(1, len(p)):
            ap += (r[i] - r[i - 1]) * p[i]
        return float(ap)

    def f1_score(y_true, y_pred):  # type: ignore
        y = np.asarray(y_true).astype(int)
        yp = np.asarray(y_pred).astype(int)
        tp = int(((y == 1) & (yp == 1)).sum())
        fp = int(((y == 0) & (yp == 1)).sum())
        fn = int(((y == 1) & (yp == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


# -----------------------------------------------------------------------------
# Defaults & Utilities
# -----------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DEFAULTS: Dict[str, Any] = {
    "discover": {
        "chains": ["ethereum", "bnb", "solana"],
        "timeframe_days": 90,
        "quote_symbols": ["USDC", "USDT", "DAI", "BUSD"],
        "t0_window_seconds": 300,
        "min_first_window_usd": 10000,
        "dedup_mode": "per_tx",
    },
    "weak_labels": {
        "early_threshold_sec": 20,
        "gas_percentile": 99,
        "same_block_swaps_min": 2,
        "known_bot_router": True,
    },
    "exclusions": {
        "buckets": [
            "burn",
            "lp_lock",
            "team_multisig",
            "treasury",
            "bridge",
            "staking_pool",
            "cex_hot",
        ]
    },
    "cohort": {
        "keys": ["chain", "tvl_quartile", "mcap_bin"],
        "mcap_bins": ["micro", "small", "mid"],
    },
    "training": {
        "model": "logreg_l1",
        "threshold_policy": "precision_at_k",
        "target_precision": 0.90,
        "stale_days": 30,
    },
    "paths": {
        "cache_dir": "./cache",
        "artifacts_dir": "./artifacts",
        "dataset_dir": "./dataset",
    },
}


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.array(x)))


def now_ts() -> int:
    return int(time.time())


def ts_to_dt(ts: int) -> datetime:
    return datetime.utcfromtimestamp(ts)


def dt_to_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Save DataFrame to parquet if possible, otherwise CSV fallback.

    Returns the actual written path.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_table(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet, or CSV fallback if parquet unavailable."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No table found at {path} or {csv_path}")


# -----------------------------------------------------------------------------
# Alerts (simple webhook)
# -----------------------------------------------------------------------------


def _notify(msg: str, level: str = "info") -> None:
    """Send a concise alert to a webhook if configured.

    Set SNIPER_ALERT_WEBHOOK_URL (or ALERT_WEBHOOK_URL) in the environment to enable.
    Designed to be best-effort: failures are silent.
    """
    # Prefer insiders' alert sink for parity with insiders model
    try:
        if INSIDERS_ALERT_AVAILABLE:
            _InsidersDefaultSink().send(f"[sniper][{level}] {msg}")
            # Do not return; also fan-out to local/file and webhook below
    except Exception:
        pass
    # Local alerts first (file/echo), then optional webhook
    try:
        if LOCAL_ALERTS_AVAILABLE:
            local_notify(msg, level=level, source="sniper")
    except Exception:
        pass
    url = os.getenv("SNIPER_ALERT_WEBHOOK_URL") or os.getenv("ALERT_WEBHOOK_URL")
    if url and REQUESTS_AVAILABLE:
        try:
            payload = {"text": f"[sniper][{level}] {msg}"}
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
    """Dict-like config with overrides.

    Use `from_yaml` to load overrides; falls back to DEFAULTS otherwise.
    Provides attribute and dict-style access.
    """

    data: Dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULTS)))

    @staticmethod
    def from_yaml(path: Optional[str]) -> "Config":
        base = json.loads(json.dumps(DEFAULTS))
        if path is None:
            return Config(base)
        with open(path, "r") as f:
            override = yaml.safe_load(f) or {}
        # deep merge
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(base, override)
        return Config(base)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    # Convenience paths
    @property
    def cache_dir(self) -> Path:
        return ensure_dir(self.data["paths"]["cache_dir"])

    @property
    def artifacts_dir(self) -> Path:
        return ensure_dir(self.data["paths"]["artifacts_dir"])

    @property
    def dataset_dir(self) -> Path:
        return ensure_dir(self.data["paths"]["dataset_dir"])


# -----------------------------------------------------------------------------
# API Clients (stubs with optional real calls)
# -----------------------------------------------------------------------------


class DuneClient:
    """Stubbed Dune client. Returns synthetic discovery results.

    Real calls are not implemented here; this is a structured stub
    capable of being extended by users.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("DUNE_API_KEY")
        self.logger = logging.getLogger(self.__class__.__name__)

    def discover_pairs(
        self,
        chains: List[str],
        timeframe_days: int,
        quote_symbols: List[str],
    ) -> pd.DataFrame:
        """Return synthetic token/pair discovery with t0 for demo.

        Columns: chain, token_address, pair_address, t0 (timestamp), dex_type,
                 tvl, mcap_bin
        """
        self.logger.info(
            "Discovering pairs (stub) for chains=%s timeframe_days=%s quotes=%s",
            chains,
            timeframe_days,
            quote_symbols,
        )
        # Create deterministic synthetic discovery
        rows = []
        now = datetime.utcnow()
        start = now - timedelta(days=timeframe_days)
        dex_types = ["uni_like", "amm_v2", "amm_v3"]
        mcap_bins = ["micro", "small", "mid"]
        random.seed(SEED)
        for chain in chains:
            for i in range(30):  # 30 tokens per chain for demo
                t0 = start + timedelta(days=random.random() * timeframe_days)
                tvl = float(max(1e4, np.random.lognormal(mean=11, sigma=1)))  # ~$50k-$1M+
                rows.append(
                    {
                        "chain": chain,
                        "token_address": f"{chain}_token_{i}",
                        "pair_address": f"{chain}_pair_{i}",
                        "t0": int(t0.timestamp()),
                        "dex_type": random.choice(dex_types),
                        "tvl": tvl,
                        "mcap_bin": random.choice(mcap_bins),
                    }
                )
        df = pd.DataFrame(rows)
        return df


class AlchemyClient:
    """Stubbed Alchemy/RPC client for EVM-like chains.

    In demo, returns synthetic gas/priority and EOA/contract flags.
    """

    def __init__(self, chain: str) -> None:
        self.chain = chain
        self.api_key = os.getenv("ALCHEMY_API_KEY_ETH") if chain == "ethereum" else os.getenv("ALCHEMY_API_KEY_BNB")
        self.logger = logging.getLogger(self.__class__.__name__)

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def get_tx_details(self, tx_hash: str) -> Dict[str, Any]:
        # Stub: return deterministic synthetic values
        h = abs(hash((self.chain, tx_hash)))
        gas_price_gwei = 10 + (h % 200)  # 10-209 gwei
        priority_fee_gwei = (h % 5)
        is_contract_caller = 1 if (h % 10) < 3 else 0
        return {
            "gas_price_gwei": float(gas_price_gwei),
            "priority_fee_gwei": float(priority_fee_gwei),
            "is_contract_caller": int(is_contract_caller),
            "router": random.choice(["default", "aggregator", "bot_router"]),
        }


class ExplorerClient:
    """Stubbed explorer client (Etherscan/BscScan/Solscan wrappers).

    Provides simple labels such as deployer, exclusions.
    """

    def __init__(self, chain: str) -> None:
        self.chain = chain
        self.api_key = (
            os.getenv("ETHERSCAN_API_KEY")
            if chain == "ethereum"
            else os.getenv("BSCSCAN_API_KEY") if chain == "bnb" else os.getenv("SOLSCAN_API_KEY")
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def get_address_labels(self, address: str) -> Dict[str, Any]:
        # Stub: random exclusion buckets
        buckets = DEFAULTS["exclusions"]["buckets"]
        h = abs(hash((self.chain, address)))
        chosen = None
        if (h % 50) == 0:
            chosen = random.choice(buckets)
        return {"exclusion_bucket": chosen}


class DexScreenerClient:
    """Optional price info; here we keep a stub to simulate USD conversion."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def to_usd(self, chain: str, token_address: str, amount_in: float) -> float:
        # Stub: convert via fake price per chain
        base = {"ethereum": 1.0, "bnb": 0.8, "solana": 0.6}.get(chain, 1.0)
        price = base * (0.8 + 0.4 * random.random())
        return amount_in * price


# -----------------------------------------------------------------------------
# Dataset Builder
# -----------------------------------------------------------------------------


class DatasetBuilder:
    """Build early trades dataset from discovery results (stubbed demo).

    - Discover pairs and t0
    - Collect early buy-side transactions in t0_window
    - Deduplicate per tx (collapse multi-hop)
    - USD conversion and quality guard
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dune = DuneClient()
        self.dexs = DexScreenerClient()

    def discover(self) -> pd.DataFrame:
        dcfg = self.config["discover"]
        df = self.dune.discover_pairs(
            chains=dcfg["chains"],
            timeframe_days=dcfg["timeframe_days"],
            quote_symbols=dcfg["quote_symbols"],
        )
        # Derive tvl quartiles per chain
        df["tvl_quartile"] = (
            df.groupby("chain")["tvl"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
        )
        self.logger.info("Discovered %d pairs/tokens", len(df))
        out = self.config.cache_dir / "discovered_pairs.parquet"
        written = save_table(df, out)
        self.logger.info("Saved discovery to %s", written)
        return df

    def _synth_trades_for_token(self, row: pd.Series, window_s: int) -> pd.DataFrame:
        chain = row["chain"]
        token = row["token_address"]
        pair = row["pair_address"]
        t0 = int(row["t0"])  # epoch seconds
        tvl = float(row["tvl"])  # in USD
        # Synthesize 50-200 trades in window
        n = random.randint(50, 200)
        base_ts = t0
        rows = []
        for i in range(n):
            dt_s = random.randint(0, window_s)
            ts = base_ts + dt_s
            block_number = (ts // 12)  # fake block number
            tx_hash = f"{pair}_{ts}_{i}"
            wallet = f"wallet_{abs(hash((token, i))) % 10000}"
            amount_in = abs(np.random.lognormal(mean=1.5, sigma=0.8))  # random size
            usd_value = min(amount_in * (tvl / 1e6), amount_in * 5)  # capped proxy
            # Gas details (EVM) or compute units (Solana)
            gas_price_gwei = 10 + (abs(hash((token, i))) % 180)
            priority_fee_gwei = (abs(hash((token, i, "p"))) % 4)
            router = random.choice(["default", "aggregator", "bot_router"])
            is_contract_caller = 1 if (abs(hash((wallet, i))) % 10) < 2 else 0
            rows.append(
                {
                    "chain": chain,
                    "token_address": token,
                    "pair_address": pair,
                    "t0": t0,
                    "block_time": ts,
                    "block_number": block_number,
                    "tx_hash": tx_hash,
                    "wallet": wallet,
                    "side": "buy",
                    "amount_in": float(amount_in),
                    "usd_value": float(usd_value),
                    "gas_price_gwei": float(gas_price_gwei),
                    "priority_fee_gwei": float(priority_fee_gwei),
                    "router": router,
                    "is_contract_caller": int(is_contract_caller),
                    "tvl": tvl,
                    "mcap_bin": row.get("mcap_bin", "micro"),
                    "tvl_quartile": row.get("tvl_quartile", "Q2"),
                    "dex_type": row.get("dex_type", "uni_like"),
                }
            )
        df = pd.DataFrame(rows)
        # Deduplicate per tx_hash (already unique in synth); compute same-block swaps
        same_block = df.groupby(["token_address", "block_number"]).tx_hash.transform("count")
        df["same_block_swaps"] = same_block.fillna(1).astype(int)
        # Random exclusions for some wallets
        buckets = DEFAULTS["exclusions"]["buckets"]
        h = df["wallet"].apply(lambda w: abs(hash((w, token))) % 60)
        df["exclusion_bucket"] = np.where(h == 0, np.random.choice(buckets), None)
        # keep only within window (already by construction)
        return df

    def build_early_trades(self, from_cache: bool = False) -> pd.DataFrame:
        dcfg = self.config["discover"]
        window_s = int(dcfg["t0_window_seconds"])
        min_usd = float(dcfg["min_first_window_usd"])
        if from_cache:
            disc_path = self.config.cache_dir / "discovered_pairs.parquet"
            if disc_path.exists() or disc_path.with_suffix(".csv").exists():
                discovered = load_table(disc_path)
            else:
                discovered = self.discover()
        else:
            discovered = self.discover()

        all_trades: List[pd.DataFrame] = []
        self.logger.info("Synthesizing early trades for %d tokens", len(discovered))
        for _, row in tqdm(discovered.iterrows(), total=len(discovered)):
            df = self._synth_trades_for_token(row, window_s)
            # Agg quality guard: total USD within window
            usd_sum = df["usd_value"].sum()
            if usd_sum < min_usd:
                continue
            all_trades.append(df)
        if not all_trades:
            self.logger.warning("No tokens passed min_first_window_usd; creating a tiny demo dataset.")
            # ensure at least some data
            if len(discovered) > 0:
                all_trades.append(self._synth_trades_for_token(discovered.iloc[0], window_s))
        early = pd.concat(all_trades, ignore_index=True)
        early["t_since_t0_sec"] = early["block_time"] - early["t0"]
        # Dedup per tx if required
        if dcfg.get("dedup_mode", "per_tx") == "per_tx":
            early = early.drop_duplicates(subset=["tx_hash"]).copy()
        out = self.config.dataset_dir / "early_trades.parquet"
        written = save_table(early, out)
        self.logger.info("Saved early trades to %s (%d rows)", written, len(early))
        return early


# -----------------------------------------------------------------------------
# Feature Builder
# -----------------------------------------------------------------------------


class FeatureBuilder:
    """Merge augmentation and compute features, plus cohort-aware normalization."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _compute_gas_percentile(self, df: pd.DataFrame) -> pd.Series:
        # Percentile rank per chain
        return (
            df.groupby("chain")["gas_price_gwei"].rank(pct=True).fillna(0.0) * 100.0
        )

    def build_features(self, early: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = early.copy()
        # Ensure required fields exist
        req_cols = [
            "t_since_t0_sec",
            "gas_price_gwei",
            "tvl",
            "same_block_swaps",
            "router",
            "is_contract_caller",
            "chain",
            "tvl_quartile",
            "mcap_bin",
        ]
        for c in req_cols:
            if c not in df.columns:
                df[c] = 0
        # Feature computations
        df["gas_price_pctile"] = self._compute_gas_percentile(df)
        df["buy_size_vs_liquidity"] = (df["usd_value"] / (df["tvl"].replace(0, np.nan))).fillna(0.0)
        df["is_contract_caller"] = df["is_contract_caller"].fillna(0).astype(int)
        # One-hot for router
        router_dummies = pd.get_dummies(df["router"].fillna("default"), prefix="router")
        df = pd.concat([df, router_dummies], axis=1)
        # Feature list
        base_features = [
            "t_since_t0_sec",
            "gas_price_pctile",
            "buy_size_vs_liquidity",
            "same_block_swaps",
            "is_contract_caller",
        ]
        one_hot_features = sorted(list(router_dummies.columns))
        feature_cols = base_features + one_hot_features

        # Cohort-aware robust scaling (median/IQR) for numerical features
        cohort_keys = self.config["cohort"]["keys"]
        scaling_params: Dict[str, Any] = {"cohort_keys": cohort_keys, "scalers": {}}
        df_norm = df.copy()

        num_cols = [c for c in feature_cols if c not in one_hot_features]
        if not cohort_keys:
            cohort_keys = []
        group = df.groupby(cohort_keys) if cohort_keys else [((), df)]
        # Ensure numeric columns are float to avoid dtype issues on assignment
        for col in num_cols:
            df_norm[col] = df_norm[col].astype(float)
        for key, sub in group:
            if isinstance(key, tuple):
                key_tuple = key
            elif key == ():
                key_tuple = ()
            else:
                key_tuple = (key,)
            mask = pd.Series(True, index=df.index)
            if cohort_keys:
                mask = np.logical_and.reduce(
                    [df[k] == v for k, v in zip(cohort_keys, key_tuple)]
                )
            scalers = {}
            for col in num_cols:
                med = float(sub[col].median()) if len(sub) > 0 else 0.0
                q1 = float(sub[col].quantile(0.25)) if len(sub) > 0 else 0.0
                q3 = float(sub[col].quantile(0.75)) if len(sub) > 0 else 1.0
                iqr = max(q3 - q1, 1e-12)
                scalers[col] = {"median": med, "iqr": iqr}
                # Apply transform
                df_norm.loc[mask, col] = (df.loc[mask, col] - med) / iqr
            scaling_params["scalers"][str(key_tuple)] = scalers

        feature_spec = {
            "feature_order": feature_cols,
            "one_hot_features": one_hot_features,
            "scaling": scaling_params,
        }
        return df_norm, feature_spec


# -----------------------------------------------------------------------------
# Weak Labeler
# -----------------------------------------------------------------------------


class WeakLabeler:
    """Implements labeling functions and combines them into weak_label_prob.

    If snorkel is available, uses LabelModel; otherwise, weighted voting with
    logistic squashing.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _lf_early_and_gas(self, df: pd.DataFrame) -> np.ndarray:
        wcfg = self.config["weak_labels"]
        early_s = df["t_since_t0_sec"] <= float(wcfg["early_threshold_sec"])
        gas_high = df["gas_price_pctile"] >= float(wcfg["gas_percentile"])
        y = np.where(early_s & gas_high, 1, 0)
        return y

    def _lf_same_block_burst(self, df: pd.DataFrame) -> np.ndarray:
        thr = int(self.config["weak_labels"]["same_block_swaps_min"])
        y = np.where(df["same_block_swaps"].fillna(0).astype(int) >= thr, 1, 0)
        return y

    def _lf_known_bot_router(self, df: pd.DataFrame) -> np.ndarray:
        use = bool(self.config["weak_labels"]["known_bot_router"])
        if not use:
            return np.zeros(len(df), dtype=int)
        # Ensure we use a vector aligned to df rows even if the one-hot column is missing
        cols = df.filter(like="router_")
        col_name = "router_bot_router"
        if col_name in cols.columns:
            s = cols[col_name].fillna(0)
        else:
            s = pd.Series(0, index=df.index)
        y = np.where(s > 0, 1, 0)
        return y

    def _lf_team_cex_lp_exclude(self, df: pd.DataFrame) -> np.ndarray:
        buckets = set(self.config["exclusions"]["buckets"])
        # 실데이터에는 exclusion_bucket 컬럼이 없을 수 있으므로 안전 가드
        if "exclusion_bucket" not in df.columns:
            return np.zeros(len(df), dtype=int)
        y = np.where(df["exclusion_bucket"].isin(buckets), -1, 0)
        return y

    def _lf_cohort_guard(self, df: pd.DataFrame) -> np.ndarray:
        # abstain if cohort fields missing
        cohort_keys = self.config["cohort"]["keys"]
        y = np.ones(len(df), dtype=int)
        for k in cohort_keys:
            y = y & df[k].notna().astype(int)
        return np.where(y == 1, 0, 0)  # keep 0 to abstain; guard handled implicitly

    def apply(self, df_features: pd.DataFrame) -> pd.DataFrame:
        # LFs
        L1 = self._lf_early_and_gas(df_features)
        L2 = self._lf_same_block_burst(df_features)
        L3 = self._lf_known_bot_router(df_features)
        L4 = self._lf_team_cex_lp_exclude(df_features)
        # Shape guards: ensure all are 1-D arrays with length == len(df)
        def _ensure_vec(x: np.ndarray) -> np.ndarray:
            a = np.asarray(x).reshape(-1)
            return a if a.shape[0] == len(df_features) else np.zeros(len(df_features), dtype=int)
        L1, L2, L3, L4 = map(_ensure_vec, [L1, L2, L3, L4])
        # Combine
        if SNORKEL_AVAILABLE:
            self.logger.info("Combining weak labels via snorkel LabelModel")
            # Map labels to {0,1} with -1 treated as 0 but can be handled via abstain in future
            # Construct label matrix: each LF outputs {1 (pos), 0/abstain, -1 (neg)}
            L = np.vstack([L1, L2, L3, L4]).T
            # Convert -1 to 0 for binary cardinality
            L_bin = L.copy()
            L_bin[L_bin == -1] = 0
            model = LabelModel(cardinality=2, verbose=False)
            model.fit(L_bin, n_epochs=100, log_freq=50)
            probs = model.predict_proba(L_bin)[:, 1]
            df_out = df_features.copy()
            df_out["weak_label_prob"] = probs
            return df_out
        else:
            self.logger.info("Combining weak labels via weighted voting fallback")
            # Weighted sum, logistic to [0,1]
            # Positive anchors high weight; negative exclusion negative weight
            w = np.array([2.0, 1.5, 2.5, -2.0])
            L = np.vstack([L1, L2, L3, L4]).T
            # Map 0 to 0, 1 to +1, -1 to -1
            s = (L * np.array([1, 1, 1, 1])).astype(float)
            z = s.dot(w)
            probs = sigmoid(z)
            df_out = df_features.copy()
            df_out["weak_label_prob"] = probs
            return df_out


# -----------------------------------------------------------------------------
# PU Learner (optional) and Supervised Trainer
# -----------------------------------------------------------------------------


class PULearner:
    """Two-stage PU learning with simple heuristics.

    Stage A: Train P vs U; select RN from U using alpha cutoff.
    Stage B: Train final model P vs RN.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def stage_train(
        self,
        X: pd.DataFrame,
        weak_prob: np.ndarray,
        feature_order: List[str],
        alpha: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # P: weak_prob >= 0.9; U: others
        P_idx = np.where(weak_prob >= 0.9)[0]
        U_idx = np.where(weak_prob < 0.9)[0]
        if len(P_idx) == 0 or len(U_idx) == 0:
            self.logger.warning("Insufficient P or U for PU; falling back to direct supervision")
            return P_idx, U_idx
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            random_state=SEED,
            class_weight="balanced",
        )
        Xmat = X[feature_order].values
        y = np.zeros(len(X), dtype=int)
        y[P_idx] = 1
        y[U_idx] = 0
        clf.fit(Xmat, y)
        scores = clf.predict_proba(Xmat)[:, 1]
        # RN selection from U: scores < alpha
        RN_idx = U_idx[scores[U_idx] < alpha]
        return P_idx, RN_idx


class Trainer:
    """Model trainer with threshold selection and metrics."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _split_timewise(self, df: pd.DataFrame, by_token: bool = True, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if by_token:
            # Sort tokens by t0, split by token
            token_t0 = df.groupby("token_address")["t0"].min().sort_values()
            tokens = token_t0.index.tolist()
            n_train = max(1, int(len(tokens) * ratio))
            train_tokens = set(tokens[:n_train])
            train_idx = df.index[df["token_address"].isin(train_tokens)].to_numpy()
            test_idx = df.index[~df["token_address"].isin(train_tokens)].to_numpy()
            return train_idx, test_idx
        else:
            # Fallback: time-based split
            thr = df["block_time"].quantile(ratio)
            train_idx = df.index[df["block_time"] <= thr].to_numpy()
            test_idx = df.index[df["block_time"] > thr].to_numpy()
            return train_idx, test_idx

    def _select_threshold(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        policy: str = "precision_at_k",
        target_precision: float = 0.9,
    ) -> float:
        # Guard: insufficient class variety -> return conservative default
        try:
            uniq = np.unique(y_true)
            if uniq.size < 2:
                return 0.5
        except Exception:
            return 0.5
        if policy == "precision_at_k":
            # Sweep thresholds to achieve >= target precision, choose with max recall
            precision, recall, thr = precision_recall_curve(y_true, scores)
            # thr has len-1 relative to precision/recall
            best_tau = 0.5
            best_recall = -1.0
            for p, r, t in zip(precision[:-1], recall[:-1], thr):
                if p >= target_precision and r > best_recall:
                    best_recall = r
                    best_tau = float(t)
            return float(best_tau)
        elif policy == "f1_max":
            precision, recall, thr = precision_recall_curve(y_true, scores)
            taus = list(thr)
            if not taus:
                return 0.5
            f1s = []
            for t in taus:
                y_pred = (scores >= t).astype(int)
                f1s.append(f1_score(y_true, y_pred))
            best_tau = float(taus[int(np.argmax(f1s))])
            return best_tau
        else:  # fixed_tau
            return 0.5

    def _train_logreg(self, X: pd.DataFrame, y: np.ndarray, feature_order: List[str]):
        if SKLEARN_AVAILABLE:
            clf = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                random_state=SEED,
                class_weight="balanced",
            )
            clf.fit(X[feature_order].values, y)
            coef = clf.coef_.reshape(-1).tolist()
            intercept = float(clf.intercept_.reshape(-1)[0])
            return clf, coef, intercept
        else:
            # Minimal numpy-based logistic regression (L2)
            Xmat = X[feature_order].values.astype(float)
            yvec = y.astype(float)
            n, d = Xmat.shape
            w = np.zeros(d)
            b = 0.0
            lr = 0.1
            l2 = 1.0 / max(n, 1)
            epochs = 400
            for _ in range(epochs):
                z = Xmat.dot(w) + b
                z = np.clip(z, -20, 20)
                p = 1.0 / (1.0 + np.exp(-z))
                grad_w = (Xmat.T.dot(p - yvec)) / n + l2 * w
                grad_b = float(np.mean(p - yvec))
                w -= lr * grad_w
                b -= lr * grad_b
            class _Dummy:
                def predict_proba(self, Xnew):
                    z = Xnew.dot(w) + b
                    z = np.clip(z, -20, 20)
                    s = 1.0 / (1.0 + np.exp(-z))
                    return np.vstack([1 - s, s]).T
            return _Dummy(), w.tolist(), float(b)

    def _train_xgb(self, X: pd.DataFrame, y: np.ndarray, feature_order: List[str]):
        assert XGBOOST_AVAILABLE
        clf = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=200,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=SEED,
            n_jobs=4,
            eval_metric="logloss",
        )
        clf.fit(X[feature_order].values, y)
        # Approximate linearization by feature importances; intercept 0
        importances = clf.feature_importances_.tolist()
        coef = importances
        intercept = 0.0
        return clf, coef, intercept

    def train(
        self,
        df_features: pd.DataFrame,
        feature_spec: Dict[str, Any],
        weak_prob: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        # Ensure required split key exists (t0); coalesce from merges if needed
        if "t0" not in df_features.columns:
            for cand in ["t0_x", "t0_y"]:
                if cand in df_features.columns:
                    df_features = df_features.copy()
                    df_features["t0"] = pd.to_numeric(df_features[cand], errors="coerce").fillna(0).astype(int)
                    break
        # Labels: prefer strong labels if available, otherwise weak labels
        y_all = (weak_prob >= 0.8).astype(int)
        try:
            labels_csv = self.config.dataset_dir / "scam_labels.csv"
            labels_parq = self.config.dataset_dir / "scam_labels.parquet"
            lab_df: Optional[pd.DataFrame] = None
            used_path: Optional[Path] = None
            if labels_csv.exists():
                lab_df = pd.read_csv(labels_csv)
                used_path = labels_csv
            elif labels_parq.exists():
                lab_df = pd.read_parquet(labels_parq)
                used_path = labels_parq
            if lab_df is not None:
                lab_df.columns = [c.strip().lower() for c in lab_df.columns]
                if "token" in lab_df.columns and "token_address" not in lab_df.columns:
                    lab_df = lab_df.rename(columns={"token": "token_address"})
                if "is_scam" in lab_df.columns and "label" not in lab_df.columns:
                    lab_df = lab_df.rename(columns={"is_scam": "label"})
                if "token_address" in lab_df.columns and "label" in lab_df.columns:
                    lab_df = lab_df[["token_address", "label"]].copy()
                    lab_df["label"] = pd.to_numeric(lab_df["label"], errors="coerce").astype("Int64")
                    join = df_features[["token_address"]].merge(
                        lab_df.dropna(subset=["label"]).drop_duplicates("token_address"),
                        on="token_address",
                        how="left",
                    )
                    if "label" in join.columns:
                        y_str = join["label"].fillna(-1).astype(int).to_numpy()
                        n_str = int((y_str >= 0).sum())
                        if n_str > 0:
                            self.logger.info(
                                "Applying strong labels from %s for %d rows (hybrid for others)",
                                used_path,
                                n_str,
                            )
                            y_all = np.where(y_str >= 0, y_str, y_all)
        except Exception as e:
            self.logger.warning("Strong label loading failed; using weak labels only. err=%s", e)
        feature_order: List[str] = feature_spec["feature_order"]

        # Split
        train_idx, test_idx = self._split_timewise(df_features)
        X_train, y_train = df_features.iloc[train_idx], y_all[train_idx]
        X_test, y_test = df_features.iloc[test_idx], y_all[test_idx]

        # Model choice
        model_choice = self.config["training"].get("model", "logreg_l1")
        if model_choice == "xgboost" and XGBOOST_AVAILABLE:
            clf, coef, intercept = self._train_xgb(X_train, y_train, feature_order)
            model_type = "xgboost"
        else:
            clf, coef, intercept = self._train_logreg(X_train, y_train, feature_order)
            model_type = "logreg_l1"

        # Scores for thresholding
        def predict_scores(df: pd.DataFrame) -> np.ndarray:
            if model_type == "xgboost":
                s = clf.predict_proba(df[feature_order].values.astype(float))[:, 1]
            else:
                z = df[feature_order].values.astype(float).dot(np.array(coef, dtype=float)) + float(intercept)
                z = np.clip(z, -20, 20)
                s = sigmoid(z)
            return np.array(s)

        train_scores = predict_scores(X_train)
        test_scores = predict_scores(X_test)

        # Metrics
        metrics = {
            "train_pr_auc": float(average_precision_score(y_train, train_scores)) if len(np.unique(y_train)) > 1 else None,
            "test_pr_auc": float(average_precision_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else None,
        }

        # Thresholds (global only for demo; cohort-specific can be added similarly)
        tau = self._select_threshold(
            y_test, test_scores, self.config["training"]["threshold_policy"], float(self.config["training"]["target_precision"])
        )
        thresholds = {"global": float(tau)}

        # Exportable model
        model_art = {
            "model_type": model_type,
            "feature_order": feature_order,
            "coef": coef,
            "intercept": intercept,
            "metrics": metrics,
        }
        return model_art, thresholds, metrics


# -----------------------------------------------------------------------------
# Exporter
# -----------------------------------------------------------------------------


class Exporter:
    """Saves artifacts for runtime scoring."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def export(self, model_art: Dict[str, Any], feature_spec: Dict[str, Any], thresholds: Dict[str, float]) -> None:
        art_dir = self.config.artifacts_dir
        ensure_dir(art_dir)
        with open(art_dir / "model_weights.json", "w") as f:
            json.dump(model_art, f, indent=2)
        with open(art_dir / "feature_spec.yaml", "w") as f:
            yaml.safe_dump(feature_spec, f, sort_keys=False)
        with open(art_dir / "thresholds.yaml", "w") as f:
            yaml.safe_dump(thresholds, f, sort_keys=False)
        self.logger.info("Artifacts exported to %s", art_dir)


# -----------------------------------------------------------------------------
# Runtime Scorer
# -----------------------------------------------------------------------------


class SniperScorer:
    """Runtime scorer using exported artifacts.

    score = sigmoid(w·x + b), and is_sniper = score >= tau
    """

    def __init__(self, artifacts_dir: str | Path = "./artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_art: Dict[str, Any] = {}
        self.feature_spec: Dict[str, Any] = {}
        self.thresholds: Dict[str, Any] = {}
        self._loaded = False
        self.load_artifacts(artifacts_dir)

    def load_artifacts(self, artifacts_dir: str | Path) -> None:
        art_dir = Path(artifacts_dir)
        with open(art_dir / "model_weights.json", "r") as f:
            self.model_art = json.load(f)
        with open(art_dir / "feature_spec.yaml", "r") as f:
            self.feature_spec = yaml.safe_load(f)
        with open(art_dir / "thresholds.yaml", "r") as f:
            self.thresholds = yaml.safe_load(f)
        self._loaded = True

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        spec = self.feature_spec.get("scaling", {})
        cohort_keys: List[str] = spec.get("cohort_keys", [])
        scalers: Dict[str, Any] = spec.get("scalers", {})
        df_out = df.copy()
        # Determine numerical columns (non one-hot)
        one_hot = set(self.feature_spec.get("one_hot_features", []))
        feature_order: List[str] = self.feature_spec.get("feature_order", [])
        num_cols = [c for c in feature_order if c not in one_hot]
        if not scalers:
            return df_out
        if cohort_keys:
            # Apply per cohort
            for key_tuple_str, sc in scalers.items():
                try:
                    key_tuple = eval(key_tuple_str)
                except Exception:
                    key_tuple = ()
                mask = np.logical_and.reduce(
                    [df_out[k] == v for k, v in zip(cohort_keys, key_tuple)]
                ) if cohort_keys else pd.Series(True, index=df_out.index)
                for col in num_cols:
                    par = sc.get(col, {"median": 0.0, "iqr": 1.0})
                    med, iqr = par.get("median", 0.0), max(par.get("iqr", 1.0), 1e-12)
                    df_out.loc[mask, col] = (df_out.loc[mask, col] - med) / iqr
        else:
            # Single scaler
            sc = scalers.get("()", {})
            for col in num_cols:
                par = sc.get(col, {"median": 0.0, "iqr": 1.0})
                med, iqr = par.get("median", 0.0), max(par.get("iqr", 1.0), 1e-12)
                df_out[col] = (df_out[col] - med) / iqr
        return df_out

    def score_df(self, df_features: pd.DataFrame) -> pd.DataFrame:
        assert self._loaded, "Artifacts not loaded"
        feature_order: List[str] = self.feature_spec["feature_order"]
        one_hot_features: List[str] = self.feature_spec.get("one_hot_features", [])
        # Ensure one-hot columns exist
        df = df_features.copy()
        for col in one_hot_features:
            if col not in df.columns:
                df[col] = 0
        # Ensure base features exist
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0.0
        # Apply scaling consistent with training
        df_scaled = self._apply_scaling(df)
        X = df_scaled[feature_order].astype(float).values
        coef = np.array(self.model_art["coef"]).reshape(-1)
        intercept = float(self.model_art["intercept"]) if self.model_art.get("model_type") != "xgboost" else 0.0
        z = X.dot(coef.astype(float)) + intercept
        z = np.clip(z, -20, 20)
        score = sigmoid(z)
        tau = float(self.thresholds.get("global", 0.5))
        is_sniper = (score >= tau).astype(int)
        out = df_features.copy()
        out["score"] = score
        out["is_sniper"] = is_sniper
        return out

    def maybe_update_weights(self, config: Config, freshness_days: Optional[int] = None) -> None:
        freshness = int(freshness_days if freshness_days is not None else DEFAULTS["training"]["stale_days"])
        art_dir = Path(config["paths"]["artifacts_dir"]) if config else self.artifacts_dir
        model_path = art_dir / "model_weights.json"
        dataset_path = Path(config["paths"]["dataset_dir"]) / "early_trades.parquet"
        stale = True
        if model_path.exists():
            age_days = (time.time() - model_path.stat().st_mtime) / 86400.0
            stale = age_days > freshness
        if dataset_path.exists() and model_path.exists():
            stale = stale or (dataset_path.stat().st_mtime > model_path.stat().st_mtime)
        if stale:
            self.logger.info("Artifacts stale or missing; retraining...")
            _notify(f"Artifacts stale/missing → retrain (freshness={freshness}d)")
            run_full_pipeline(config)
            # reload
            self.load_artifacts(art_dir)
            # Post-train metrics guard
            try:
                m = self.model_art.get("metrics", {}) if isinstance(self.model_art, dict) else {}
                test_ap = float(m.get("test_pr_auc")) if m.get("test_pr_auc") is not None else None
                tgt = float(config["training"].get("target_precision", 0.9))
                if test_ap is None or test_ap < 0.2:
                    _notify(f"Low test PR-AUC after retrain: {test_ap}. Check data/labels.", level="warn")
                else:
                    self.logger.info("Post-train test PR-AUC: %.3f (target precision %.2f)", test_ap, tgt)
            except Exception:
                pass
        else:
            self.logger.info("Artifacts are fresh; no update.")


# -----------------------------------------------------------------------------
# Pipeline Runners
# -----------------------------------------------------------------------------


def run_full_pipeline(config: Config) -> None:
    logger = logging.getLogger("Pipeline")
    # 1) Discover + Dataset
    dsb = DatasetBuilder(config)
    dsb.discover()
    early = dsb.build_early_trades(from_cache=True)
    # 2) Features
    fb = FeatureBuilder(config)
    feats, feat_spec = fb.build_features(early)
    # 3) Weak labels
    wl = WeakLabeler(config)
    weaked = wl.apply(feats)
    # 4) Train
    trainer = Trainer(config)
    model_art, thresholds, metrics = trainer.train(weaked, feat_spec, weaked["weak_label_prob"].values)
    # 5) Export
    exporter = Exporter(config)
    exporter.export(model_art, feat_spec, thresholds)
    logger.info("Pipeline completed. Metrics: %s", metrics)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sniper ML Pipeline CLI")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config to override defaults")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("discover", help="Discover pairs and t0")
    s1.add_argument("--days", type=int, default=DEFAULTS["discover"]["timeframe_days"]) 
    s1.add_argument("--min-first-window", type=float, default=DEFAULTS["discover"]["min_first_window_usd"]) 
    s1.add_argument("--chains", nargs="+", default=DEFAULTS["discover"]["chains"])

    s2 = sub.add_parser("build-dataset", help="Build early trades dataset")
    s2.add_argument("--from-cache", type=int, default=1)

    s3 = sub.add_parser("weak-label", help="Apply weak labeling")
    s3.add_argument("--snorkel", type=int, default=1)

    s4 = sub.add_parser("train", help="Train a model")
    s4.add_argument("--model", type=str, default=DEFAULTS["training"]["model"], choices=["logreg_l1", "xgboost"]) 

    s5 = sub.add_parser("export-config", help="Export current feature spec and thresholds")
    s5.add_argument("--out", type=str, default="./artifacts")

    s6 = sub.add_parser("score", help="Score a features parquet")
    s6.add_argument("--input", type=str, required=True)
    s6.add_argument("--out", type=str, required=True)

    s7 = sub.add_parser("maybe-update-weights", help="Retrain if artifacts are stale")
    s7.add_argument("--stale-days", type=int, default=DEFAULTS["training"]["stale_days"]) 

    return p


def main_cli(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = build_arg_parser().parse_args(argv)
    config = Config.from_yaml(args.config)
    logger = logging.getLogger("CLI")

    if args.cmd == "discover":
        # Override runtime params
        config.data["discover"]["timeframe_days"] = int(args.days)
        config.data["discover"]["min_first_window_usd"] = float(args.min_first_window)
        config.data["discover"]["chains"] = list(args.chains)
        logger.info("Running discover with %s", config["discover"])
        DatasetBuilder(config).discover()

    elif args.cmd == "build-dataset":
        from_cache = bool(int(args.from_cache))
        logger.info("Building dataset (from_cache=%s)", from_cache)
        DatasetBuilder(config).build_early_trades(from_cache=from_cache)

    elif args.cmd == "weak-label":
        if int(args.snorkel) == 0:
            global SNORKEL_AVAILABLE
            SNORKEL_AVAILABLE = False
        ds_path = config.dataset_dir / "early_trades.parquet"
        if not (ds_path.exists() or ds_path.with_suffix('.csv').exists()):
            logger.info("Dataset missing; building from cache discovery first.")
            DatasetBuilder(config).build_early_trades(from_cache=True)
        df = load_table(ds_path)
        feats, feat_spec = FeatureBuilder(config).build_features(df)
        weaked = WeakLabeler(config).apply(feats)
        out = config.dataset_dir / "early_trades_weak.parquet"
        out_written = save_table(weaked, out)
        with open(config.artifacts_dir / "feature_spec.yaml", "w") as f:
            yaml.safe_dump(feat_spec, f, sort_keys=False)
        logger.info("Weak labels saved to %s", out_written)

    elif args.cmd == "train":
        # optional model override
        config.data["training"]["model"] = args.model
        ds_path = config.dataset_dir / "early_trades.parquet"
        if not (ds_path.exists() or ds_path.with_suffix('.csv').exists()):
            logger.info("Dataset missing; building from cache discovery first.")
            DatasetBuilder(config).build_early_trades(from_cache=True)
        df = load_table(ds_path)
        feats, feat_spec = FeatureBuilder(config).build_features(df)
        weaked = WeakLabeler(config).apply(feats)
        trainer = Trainer(config)
        model_art, thresholds, metrics = trainer.train(weaked, feat_spec, weaked["weak_label_prob"].values)
        Exporter(config).export(model_art, feat_spec, thresholds)
        logger.info("Training done. thresholds=%s metrics=%s", thresholds, metrics)

    elif args.cmd == "export-config":
        outdir = Path(args.out)
        ensure_dir(outdir)
        # If artifacts exist, just confirm copy; else run minimal build to generate feature_spec
        fspec_path = config.artifacts_dir / "feature_spec.yaml"
        if not fspec_path.exists():
            # generate from dataset
            ds_path = config.dataset_dir / "early_trades.parquet"
            if not (ds_path.exists() or ds_path.with_suffix('.csv').exists()):
                DatasetBuilder(config).build_early_trades(from_cache=True)
            df = load_table(ds_path)
            _, feat_spec = FeatureBuilder(config).build_features(df)
            with open(fspec_path, "w") as f:
                yaml.safe_dump(feat_spec, f, sort_keys=False)
        # Copy
        for name in ["model_weights.json", "feature_spec.yaml", "thresholds.yaml"]:
            src = config.artifacts_dir / name
            if src.exists():
                dst = outdir / name
                dst.write_bytes(src.read_bytes())
        logger.info("Exported artifacts to %s", outdir)

    elif args.cmd == "score":
        scorer = SniperScorer(config.artifacts_dir)
        in_path = Path(args.input)
        df = load_table(in_path)
        # Ensure features exist; rebuild if input is early_trades
        if "gas_price_pctile" not in df.columns:
            df, _ = FeatureBuilder(config).build_features(df)
        out_df = scorer.score_df(df)
        out_path = Path(args.out)
        ensure_dir(out_path.parent)
        written = save_table(out_df, out_path)
        logger.info("Saved scored table to %s", written)

    elif args.cmd == "maybe-update-weights":
        stale_days = int(args.stale_days)
        scorer = SniperScorer(config.artifacts_dir)
        scorer.maybe_update_weights(config, freshness_days=stale_days)
        logger.info("maybe-update-weights complete.")


# -----------------------------------------------------------------------------
# Adapters for main_pipeline.py
# -----------------------------------------------------------------------------


def get_sniper_scorer(artifacts_dir: str = "./artifacts") -> SniperScorer:
    """Load artifacts and return a ready-to-use SniperScorer.

    Parameters:
    - artifacts_dir: path to artifacts directory
    """
    return SniperScorer(artifacts_dir)


def maybe_update_weights(config_path: str | None = None, freshness_days: int | None = None) -> None:
    """If artifacts are stale or missing, run discover→build-dataset→weak-label→train→export.

    Parameters:
    - config_path: optional path to a YAML config
    - freshness_days: override default staleness window
    """
    config = Config.from_yaml(config_path)
    scorer = SniperScorer(config.artifacts_dir)
    scorer.maybe_update_weights(config, freshness_days=freshness_days)


def score_transactions(transactions_df: "pd.DataFrame") -> "pd.DataFrame":
    """Score early-interval transactions and return with score/is_sniper.

    Required columns (or computed if missing):
    - chain, tvl_quartile, mcap_bin, router, is_contract_caller,
      gas_price_gwei, usd_value, t_since_t0_sec, same_block_swaps
    If raw early trades are provided, features will be derived automatically.
    """
    cfg = Config.from_yaml(None)
    df = transactions_df.copy()
    # If features not present, build them
    if "gas_price_pctile" not in df.columns:
        df, _ = FeatureBuilder(cfg).build_features(df)
    scorer = SniperScorer(cfg.artifacts_dir)
    # Try auto-update if artifacts are missing
    try:
        return scorer.score_df(df)
    except FileNotFoundError:
        run_full_pipeline(cfg)
        scorer = SniperScorer(cfg.artifacts_dir)
        return scorer.score_df(df)


# -----------------------------------------------------------------------------
# Simple Unit Tests (quick, local)
# -----------------------------------------------------------------------------


def _test_feature_order_deterministic() -> None:
    cfg = Config.from_yaml(None)
    dsb = DatasetBuilder(cfg)
    disc = dsb.discover()
    early = dsb.build_early_trades(from_cache=True)
    fb = FeatureBuilder(cfg)
    feats1, spec1 = fb.build_features(early)
    feats2, spec2 = fb.build_features(early)
    assert spec1["feature_order"] == spec2["feature_order"], "Feature order should be deterministic"


def _test_scoring_deterministic() -> None:
    cfg = Config.from_yaml(None)
    run_full_pipeline(cfg)
    df = load_table(cfg.dataset_dir / "early_trades.parquet")
    df_feat, _ = FeatureBuilder(cfg).build_features(df)
    scorer = SniperScorer(cfg.artifacts_dir)
    out1 = scorer.score_df(df_feat)
    out2 = scorer.score_df(df_feat)
    assert np.allclose(out1["score"].values, out2["score"].values), "Scores should be deterministic"


def _test_threshold_application() -> None:
    cfg = Config.from_yaml(None)
    scorer = SniperScorer(cfg.artifacts_dir)
    tau = float(scorer.thresholds.get("global", 0.5))
    # Create a tiny synthetic feature DF aligned to feature order
    fo = scorer.feature_spec["feature_order"]
    df = pd.DataFrame({c: 0.0 for c in fo})
    # Force scores above/below by adjusting sign via coef
    coef = np.array(scorer.model_art["coef"])[: len(fo)]
    if np.allclose(coef, 0):
        # If coef all zero (unlikely), set one feature to produce extremes
        df.loc[0, fo[0]] = 100.0
    out = scorer.score_df(df)
    assert (out["score"] >= 0).all() and (out["score"] <= 1).all(), "Scores must be in [0,1]"
    assert ((out["score"] >= tau) == (out["is_sniper"] == 1)).all(), "Threshold application mismatch"


if __name__ == "__main__":
    # If invoked directly without args, run CLI help
    if len(sys.argv) == 1:
        print("Usage examples:\n  python sniper_ml_pipeline.py discover --days 90 --chains ethereum bnb solana\n  python sniper_ml_pipeline.py build-dataset --from-cache 1\n  python sniper_ml_pipeline.py weak-label --snorkel 0\n  python sniper_ml_pipeline.py train --model logreg_l1\n  python sniper_ml_pipeline.py score --input ./dataset/early_trades.parquet --out ./dataset/scored.parquet\n  python sniper_ml_pipeline.py maybe-update-weights --stale-days 30\n")
        sys.exit(0)
    main_cli()
