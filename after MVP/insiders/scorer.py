from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config_schema import Config
from .events import EventSignal


def _weighted_mean(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    wsum = 0.0
    ssum = 0.0
    for k, s in scores.items():
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        wsum += w
        ssum += w * float(s)
    if wsum == 0.0:
        return 50.0
    return ssum / wsum


def _synergy_damped_sum(events: List[EventSignal]) -> float:
    if not events:
        return 0.0
    vals = [e.value for e in events]
    # Sort by absolute value descending
    vals_sorted = sorted(vals, key=lambda x: abs(x), reverse=True)
    total = 0.0
    if vals_sorted:
        total += vals_sorted[0]
    if len(vals_sorted) > 1:
        total += 0.6 * sum(vals_sorted[1:])
    return total


def grade_from_score(score: float, cfg: Config) -> str:
    th = cfg.thresholds.grade
    if score >= th.HIGH:
        return "HIGH"
    if score >= th.MED:
        return "MED"
    if score >= th.MID:
        return "MID"
    return "LOW"


def compose_insider_score(
    *,
    cfg: Config,
    norm_scores: Dict[str, float],
    events: List[EventSignal],
) -> Tuple[float, str, float]:
    """Compose final insider score and grade.

    Returns:
        Tuple of (insider_score, grade, events_contrib)
    """

    base = _weighted_mean(norm_scores, cfg.weights.dict())
    ev = _synergy_damped_sum(events)
    score = max(0.0, min(100.0, base + ev))
    return score, grade_from_score(score, cfg), ev

