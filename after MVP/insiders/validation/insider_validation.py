"""
Insider Risk Scoring – Validation Pipeline

This module automates validation for an existing Insider Score model:
- Ground truth generation (via Dune) for positives/negatives
- Component-level validation (feature distributions)
- Final score validation (distributions, thresholds, PR curve)

Notes:
- Dune SQL is provided as commented templates inside fetch_* functions.
- The dune-client calls are placeholders; adapt queries and result parsing.
- The pipeline writes plots to artifacts/validation/.

Python: 3.10+
Dependencies: pandas, matplotlib, seaborn, scikit-learn, dune-client
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    # dune-client >=1.7.0
    from dune_client.client import DuneClient  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    DuneClient = None  # type: ignore

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_sccore,
    precision_recall_curve,
)


ARTIFACT_DIR = Path("artifacts/validation")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Part 1: Ground Truth Dataset
# ----------------------------


def setup_dune_client() -> "DuneClient":
    """Initialize a Dune client using env var DUNE_API_KEY.

    Returns
    -------
    DuneClient
        An authenticated dune-client instance.

    Raises
    ------
    RuntimeError
        If dune-client is not installed or API key is missing.
    """
    if DuneClient is None:
        raise RuntimeError(
            "dune-client is not installed. Please `poetry add dune-client` and set DUNE_API_KEY."
        )
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DUNE_API_KEY in environment.")
    return DuneClient(api_key=api_key)


def fetch_pre_dex_receivers(token_address: str) -> list[str]:
    """Placeholder: wallets receiving token from deployer before first DEX liquidity.

    The real implementation should execute the below SQL template via Dune and
    return a list of addresses (as lowercased EVM checksum-insensitive strings).

    Dune SQL Template (adapt and parameterize appropriately):

    -- Find first addLiquidity event timestamp for major DEXes (Uniswap/Sushiswap/Balancer)
    WITH first_liquidity AS (
        -- Replace with actual decoded event tables and factory/router references
        -- Aim: earliest timestamp where the token is paired and liquidity is added
        SELECT
            MIN(evt_block_time) AS first_liq_ts
        FROM
            dex.trades
        WHERE
            (token_a = LOWER('{{token_address}}') OR token_b = LOWER('{{token_address}}'))
    ),
    deployer_transfers AS (
        -- Outbound token transfers from deployer before first liquidity timestamp
        SELECT
            t.to AS receiver,
            t.evt_block_time
        FROM erc20."ERC20_evt_Transfer" t
        JOIN first_liquidity f ON TRUE
        WHERE
            t.contract_address = LOWER('{{token_address}}')
            AND t.from = (SELECT deployer FROM tokens.metadata WHERE token_address = LOWER('{{token_address}}'))
            AND t.evt_block_time < f.first_liq_ts
    )
    SELECT DISTINCT LOWER(receiver) AS wallet
    FROM deployer_transfers;

    """
    # TODO: Execute query via dune client and parse results
    # For now, return empty to allow pipeline to run; main() provides fallback.
    return []


def fetch_deployer_funded_wallets(token_address: str) -> list[str]:
    """Placeholder: wallets funded in native token by deployer pre-launch, then first buyers.

    Dune SQL Template (adapt and parameterize):

    -- Identify deployer and launch window
    WITH deployer AS (
        SELECT deployer
        FROM tokens.metadata
        WHERE token_address = LOWER('{{token_address}}')
    ),
    first_liquidity AS (
        SELECT MIN(evt_block_time) AS first_liq_ts
        FROM dex.trades
        WHERE token_a = LOWER('{{token_address}}') OR token_b = LOWER('{{token_address}}')
    ),
    native_funding AS (
        -- Native coin transfers (e.g., ETH) from deployer to wallets within T hours pre-liquidity
        SELECT
            tx."to" AS wallet,
            tx.block_time AS ts,
            tx.value
        FROM ethereum.transactions tx
        JOIN deployer d ON tx."from" = d.deployer
        JOIN first_liquidity f ON TRUE
        WHERE tx.block_time BETWEEN f.first_liq_ts - INTERVAL '24 hours' AND f.first_liq_ts
    ),
    early_buys AS (
        -- First buyers on DEX within N blocks post-liquidity
        SELECT DISTINCT LOWER(trader) AS wallet
        FROM dex.trades
        JOIN first_liquidity f ON TRUE
        WHERE (token_b = LOWER('{{token_address}}') OR token_a = LOWER('{{token_address}}'))
          AND evt_block_time BETWEEN f.first_liq_ts AND f.first_liq_ts + INTERVAL '1 hour'
          AND side = 'buy'
    )
    SELECT DISTINCT LOWER(nf.wallet) AS wallet
    FROM native_funding nf
    JOIN early_buys eb ON LOWER(nf.wallet) = eb.wallet;

    """
    # TODO: Execute query via dune client and parse results
    return []


def fetch_obvious_non_insiders(token_address: str, sample_size: int = 100) -> list[str]:
    """Placeholder: wallets first acquiring ≥30 days post-launch with CEX-funded sources.

    Dune SQL Template (adapt and parameterize):

    -- Launch and CEX labeled addresses
    WITH first_liquidity AS (
        SELECT MIN(evt_block_time) AS first_liq_ts
        FROM dex.trades
        WHERE token_a = LOWER('{{token_address}}') OR token_b = LOWER('{{token_address}}')
    ),
    cex_labels AS (
        -- Known CEX hot/cold wallets from labels
        SELECT LOWER(address) AS cex_addr
        FROM labels.cex
    ),
    first_acquisition AS (
        -- Wallet's first inbound token transfer or first buy
        SELECT LOWER(t."to") AS wallet, MIN(t.evt_block_time) AS first_ts
        FROM erc20."ERC20_evt_Transfer" t
        WHERE t.contract_address = LOWER('{{token_address}}')
        GROUP BY 1
    ),
    late_adopters AS (
        SELECT fa.wallet
        FROM first_acquisition fa
        JOIN first_liquidity f ON TRUE
        WHERE fa.first_ts >= f.first_liq_ts + INTERVAL '30 days'
    ),
    funding_sources AS (
        -- Trace immediate funding parents prior to first acquisition
        SELECT DISTINCT LOWER(tx."from") AS parent, LOWER(tx."to") AS wallet
        FROM ethereum.transactions tx
        JOIN first_acquisition fa ON tx."to" = fa.wallet AND tx.block_time <= fa.first_ts
    )
    SELECT DISTINCT fs.wallet
    FROM funding_sources fs
    JOIN late_adopters la ON la.wallet = fs.wallet
    JOIN cex_labels cl ON cl.cex_addr = fs.parent
    LIMIT {{sample_size}};

    """
    # TODO: Execute query via dune client and parse results
    return []


def generate_ground_truth_dataset(
    token_addresses: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build ground truth for positives (insiders) and negatives (non-insiders).

    Parameters
    ----------
    token_addresses: list[str]
        Token contract addresses to analyze (chain-specific, lowercase preferred).

    Returns
    -------
    (positive_df, negative_df): tuple of DataFrames
        Each with columns: ['wallet_address', 'token_address', 'ground_truth_label']
    """
    positive_records: list[tuple[str, str, int]] = []
    negative_records: list[tuple[str, str, int]] = []

    for token in token_addresses:
        # Positive heuristics
        pre_dex = fetch_pre_dex_receivers(token)
        funded = fetch_deployer_funded_wallets(token)

        for w in set([*pre_dex, *funded]):
            positive_records.append((w.lower(), token.lower(), 1))

        # Negatives
        non_insiders = fetch_obvious_non_insiders(token, sample_size=100)
        for w in non_insiders:
            negative_records.append((w.lower(), token.lower(), 0))

    positive_df = pd.DataFrame(
        positive_records, columns=["wallet_address", "token_address", "ground_truth_label"]
    ).drop_duplicates()
    negative_df = pd.DataFrame(
        negative_records, columns=["wallet_address", "token_address", "ground_truth_label"]
    ).drop_duplicates()

    # Fallback: create a tiny synthetic dataset if Dune queries returned no rows
    if positive_df.empty and negative_df.empty:
        rng = random.Random(42)
        synth_tokens = token_addresses or ["0xsynthetic"]
        pos = [(f"0xpos{idx:03x}", synth_tokens[0], 1) for idx in range(40)]
        neg = [(f"0xneg{idx:03x}", synth_tokens[0], 0) for idx in range(60)]
        positive_df = pd.DataFrame(pos, columns=["wallet_address", "token_address", "ground_truth_label"])
        negative_df = pd.DataFrame(neg, columns=["wallet_address", "token_address", "ground_truth_label"])

    return positive_df, negative_df


# --------------------------------
# Part 2: Component-level Validation
# --------------------------------


def get_raw_metric_scores(wallets_df: pd.DataFrame, metrics_list: list[str]) -> pd.DataFrame:
    """Mock metric computation for a list of wallets.

    Produces un-normalized metric scores for each wallet; replace with real
    feature computation hooked into your SDK/core when available.
    """
    rng = np.random.default_rng(seed=123)
    out = pd.DataFrame({"wallet_address": wallets_df["wallet_address"].values})
    for m in metrics_list:
        # Skew metrics slightly by lexical group for demo stability
        base = rng.normal(loc=0.0, scale=1.0, size=len(out))
        bump = np.where(out["wallet_address"].str.startswith("0xpos"), 0.5, 0.0)
        out[m] = base + bump
    return out


def validate_model_components(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    raw_metrics: list[str],
) -> None:
    """Generate box plots comparing raw metric distributions across groups.

    Saves one PNG per metric under artifacts/validation/.
    """
    # Combine for metric computation
    combined = pd.concat([positive_df, negative_df], ignore_index=True)
    metric_df = get_raw_metric_scores(combined[["wallet_address"]], raw_metrics)
    combined = combined.merge(metric_df, on="wallet_address", how="left")

    combined["group"] = np.where(combined["ground_truth_label"] == 1, "Insider", "Non-Insider")

    for metric in raw_metrics:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=combined, x="group", y=metric)
        sns.stripplot(data=combined, x="group", y=metric, color="black", alpha=0.2, size=2)
        plt.title(f"Distribution of {metric} for Insider vs. Non-Insider Groups")
        plt.xlabel("")
        plt.tight_layout()
        out_path = ARTIFACT_DIR / f"component_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


# ---------------------------
# Part 3: Final Score Validation
# ---------------------------


def calculate_insider_score(address: str, token: str) -> float:
    """Mock InsiderScore in [0, 100]. Replace with real model call.

    For demonstration, returns a random score.
    """
    rng = random.Random(hash((address, token)) & 0xFFFFFFFF)
    return rng.uniform(0, 100)


def validate_final_score(positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> None:
    """Validate final InsiderScore via distributions, thresholds, and PR curve.

    Outputs plots and prints Precision/Recall/F1 at thresholds 60 and 80.
    """
    def score_df(df: pd.DataFrame) -> pd.DataFrame:
        scores = [
            calculate_insider_score(r.wallet_address, r.token_address)  # type: ignore[attr-defined]
            for r in df.itertuples(index=False)
        ]
        out = df.copy()
        out["score"] = scores
        return out

    pos_scored = score_df(positive_df)
    neg_scored = score_df(negative_df)

    # Distribution plot
    plt.figure(figsize=(8, 4))
    sns.kdeplot(pos_scored["score"], fill=True, label="Insider")
    sns.kdeplot(neg_scored["score"], fill=True, label="Non-Insider")
    plt.title("InsiderScore Distributions: Insiders vs. Non-Insiders")
    plt.xlabel("InsiderScore")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "final_score_distributions.png", dpi=150)
    plt.close()

    # Metrics at thresholds
    y_true = np.concatenate([
        np.ones(len(pos_scored), dtype=int),
        np.zeros(len(neg_scored), dtype=int),
    ])
    y_scores = np.concatenate([
        pos_scored["score"].to_numpy(),
        neg_scored["score"].to_numpy(),
    ])

    for thresh in (60, 80):
        y_pred = (y_scores >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(
            f"Threshold {thresh}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}"
        )

    # Precision-Recall curve
    # Normalize scores to [0, 1] for stability
    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.ptp() + 1e-9)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores_norm)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_arr, precision_arr, color="C0")
    plt.title("Precision-Recall Curve (InsiderScore)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()


# ------------------------------
# Part 4: Orchestration (main)
# ------------------------------


def main() -> None:
    """Run the full validation pipeline end-to-end."""
    # Define tokens to analyze (lowercase).
    token_addresses = [
        # Replace with actual tokens, e.g., USDC is a placeholder here
        "0x0000000000000000000000000000000000000000",
    ]

    print("[1/3] Generating ground truth datasets...")
    positive_df, negative_df = generate_ground_truth_dataset(token_addresses)
    print(
        f"Ground truth: {len(positive_df)} positives, {len(negative_df)} negatives across {len(token_addresses)} tokens."
    )

    # Choose raw metrics to validate (replace with real feature names)
    raw_metrics = ["FR_mkt", "SPI_VC", "EMR_team"]
    print("[2/3] Validating model components (features)...")
    validate_model_components(positive_df, negative_df, raw_metrics)
    print(f"Saved component plots to {ARTIFACT_DIR}")

    print("[3/3] Validating final InsiderScore...")
    validate_final_score(positive_df, negative_df)
    print(f"Saved score plots to {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

