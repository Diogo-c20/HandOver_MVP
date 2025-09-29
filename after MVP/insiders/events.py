from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config_schema import Config


@dataclass
class EventSignal:
    """Represents a detected event with contribution to the score.

    value is the signed score adjustment (positive for penalty, negative for relief).
    """

    key: str
    value: float
    reason: str


def detect_events(
    *,
    cfg: Config,
    metrics: Dict[str, Optional[float]],
    signals: Dict[str, bool],
) -> List[EventSignal]:
    """Detect events and return score adjustments based on config.

    Args:
        cfg: Configuration containing event scores.
        metrics: Raw or normalized metric values as needed for thresholds.
        signals: Precomputed boolean flags for qualitative events.

    Returns:
        List of EventSignal entries with reasons.
    """

    out: List[EventSignal] = []
    pen = cfg.events.penalties
    rel = cfg.events.reliefs

    team_ff = metrics.get("team_to_cex_over_ff")
    if team_ff is not None:
        if team_ff >= 0.01:
            out.append(
                EventSignal(
                    key="team_to_cex_ge_1pct_ff",
                    value=float(pen.get("team_to_cex_ge_1pct_ff", 25.0)),
                    reason=f"Team→CEX {team_ff:.2%} of free-float (≥1%)",
                )
            )
        elif team_ff >= 0.003:
            out.append(
                EventSignal(
                    key="team_to_cex_ge_0_3pct_ff",
                    value=float(pen.get("team_to_cex_ge_0_3pct_ff", 15.0)),
                    reason=f"Team→CEX {team_ff:.2%} of free-float (0.3–1%)",
                )
            )

    if signals.get("lockup_fanout", False):
        out.append(
            EventSignal(
                key="lockup_fanout",
                value=float(pen.get("lockup_fanout", 8.0)),
                reason="Post-lockup fan-out pattern detected",
            )
        )

    if signals.get("vc_sync_sell_24h", False):
        out.append(
            EventSignal(
                key="vc_sync_sell_24h",
                value=float(pen.get("vc_sync_sell_24h", 12.0)),
                reason="VC cohort synchronous selling ≤24h window",
            )
        )

    if signals.get("new_whale_from_cex_or_team", False):
        out.append(
            EventSignal(
                key="new_whale_from_cex_or_team",
                value=float(pen.get("new_whale_from_cex_or_team", 10.0)),
                reason="New whale sourced from CEX/team treasury",
            )
        )

    if signals.get("obfuscation_bridge_mixer", False):
        out.append(
            EventSignal(
                key="obfuscation_bridge_mixer",
                value=float(pen.get("obfuscation_bridge_mixer", 5.0)),
                reason="Bridging/mixer obfuscation detected",
            )
        )

    if signals.get("onchain_vesting_escrow", False):
        out.append(
            EventSignal(
                key="onchain_vesting_escrow",
                value=float(rel.get("onchain_vesting_escrow", -8.0)),
                reason="On-chain vesting/escrow in effect",
            )
        )

    if signals.get("vc_unlock_no_sell_30d", False):
        out.append(
            EventSignal(
                key="vc_unlock_no_sell_30d",
                value=float(rel.get("vc_unlock_no_sell_30d", -8.0)),
                reason="VC unlocked but no sell for 30 days",
            )
        )

    if signals.get("mm_internal_liquidity_transfer", False):
        out.append(
            EventSignal(
                key="mm_internal_liquidity_transfer",
                value=float(rel.get("mm_internal_liquidity_transfer", -6.0)),
                reason="Market-maker internal liquidity transfer",
            )
        )

    return out

