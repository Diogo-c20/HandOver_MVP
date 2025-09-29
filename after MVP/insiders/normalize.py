from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config_schema import Config


@dataclass
class CohortBaseline:
    """Baseline samples for a cohort with update timestamp for time decay."""

    samples: Sequence[float]
    updated_at: datetime


class BaselineStore:
    """In-memory baseline store with hierarchical keys for backoff.

    Key format is a string composed of fields (e.g., CHAIN:FF_BIN:MCAP_BIN:EVENT_TYPE).
    This store is deliberately simple for unit testing and offline usage.
    """

    def __init__(self, data: Dict[str, CohortBaseline]):
        self._data = data

    def get(self, key: str) -> Optional[CohortBaseline]:
        return self._data.get(key)

    def has(self, key: str) -> bool:
        return key in self._data


def _winsorize_value(x: float, samples: Sequence[float], p_lo: float, p_hi: float) -> float:
    lo = np.percentile(samples, p_lo)
    hi = np.percentile(samples, p_hi)
    return float(min(max(x, lo), hi))


def _percentile_score(x: float, samples: Sequence[float]) -> float:
    n = len(samples)
    if n == 0:
        return 50.0
    rank = np.sum(np.array(samples) <= x)
    return 100.0 * (rank / (n + 1))


def _robust_score(x: float, samples: Sequence[float]) -> float:
    if len(samples) == 0:
        return 50.0
    med = float(np.median(samples))
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = float(q75 - q25)
    scale = (iqr / 1.349) if iqr > 0 else 1e-9
    z = (x - med) / scale
    score = 50.0 + 12.0 * z
    return float(max(0.0, min(100.0, score)))


def _alpha(n: int, k: int) -> float:
    return n / (n + k) if n >= 0 else 0.0


def _time_decay_weight(ts: datetime, half_life_days: int) -> float:
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    lam = math.log(2) / max(1.0, float(half_life_days))
    return math.exp(-lam * age_days)


def normalize_with_backoff(
    *,
    cfg: Config,
    value: Optional[float],
    cohort_keys: List[str],
    event_group: str,
    baselines: BaselineStore,
    label_trust: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Normalize a metric using percentile + robust-z mixture with hierarchical backoff.

    If value is None, replace with cohort median at the most specific available level.

    Args:
        cfg: Configuration.
        value: Raw metric value (ratio) or None for NA.
        cohort_keys: Candidate cohort keys ordered from most specific to least specific,
            corresponding to cfg.normalization.backoff semantics.
        event_group: Logical group label, e.g., "FR" or "outflow" (for backoff stages
            where EVENT_GROUP is used in the key).
        baselines: BaselineStore providing sample distributions.
        label_trust: Weight in [0, 1] applied as a multiplicative dampener.

    Returns:
        Tuple of (score in [0, 100], details dict with debug info).
    """

    used_levels: List[Tuple[str, float, int, float]] = []  # (key, score, n, w)
    wins_lo, wins_hi = cfg.normalization.winsorize

    # Find the first available cohort for centering NA and computing scores
    chosen_key: Optional[str] = None
    chosen_samples: Optional[Sequence[float]] = None
    for key in cohort_keys:
        b = baselines.get(key)
        if b and len(b.samples) > 0:
            chosen_key = key
            chosen_samples = b.samples
            break

    # If value is NA, replace with median of chosen cohort if available
    x: float
    replaced_with_median = False
    if value is None:
        if chosen_samples is not None and len(chosen_samples) > 0:
            x = float(np.median(chosen_samples))
            replaced_with_median = True
        else:
            x = 0.0
    else:
        x = float(value)

    # Accumulate scores across backoff levels with time decay and sample-size weights
    total_w = 0.0
    total_score = 0.0
    for key in cohort_keys:
        b = baselines.get(key)
        if not b or len(b.samples) == 0:
            continue
        n = len(b.samples)
        # Winsorize value against this cohort distribution
        x_w = _winsorize_value(x, b.samples, wins_lo, wins_hi)
        s_pctl = _percentile_score(x_w, b.samples)
        s_rob = _robust_score(x_w, b.samples)
        a = _alpha(n, cfg.normalization.alpha_k)
        s_mix = a * s_pctl + (1.0 - a) * s_rob
        w_time = _time_decay_weight(b.updated_at, cfg.normalization.half_life_days)
        w = n * w_time
        total_w += w
        total_score += s_mix * w
        used_levels.append((key, s_mix, n, w))

    if total_w == 0:
        score = 50.0
    else:
        score = total_score / total_w

    # Apply label trust dampening
    score *= float(label_trust)
    score = max(0.0, min(100.0, score))

    details: Dict[str, float] = {
        "score": float(score),
        "replaced_with_median": 1.0 if replaced_with_median else 0.0,
    }
    for i, (k, s, n, w) in enumerate(used_levels):
        details[f"level_{i}_key"] = hash(k) % 1e9  # anonymize key but keep deterministic
        details[f"level_{i}_score"] = float(s)
        details[f"level_{i}_n"] = float(n)
        details[f"level_{i}_w"] = float(w)
    return float(score), details
