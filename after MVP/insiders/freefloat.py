from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Holding:
    """Represents a holding at t0.

    Attributes:
        address: Address string.
        balance: Token units at t0 (float for simplicity).
        label: Optional role/category label, e.g., 'burn', 'team', 'vc', 'bridge',
               'router', 'staking_pool', 'treasury', 'cex_custody'.
    """

    address: str
    balance: float
    label: Optional[str] = None


@dataclass
class FreeFloatInputs:
    total_supply_t0: float
    holdings_t0: Iterable[Holding]
    exclude_treasury: bool = True
    exclude_cex_custody: bool = True


EXCLUDE_LABELS_BASE = {
    "burn",
    "lock",
    "vesting",
    "bridge",
    "router",
    "staking_pool",
}


def compute_free_float_t0(inputs: FreeFloatInputs) -> float:
    """Compute free-float at t0 with exclusion rules.

    Exclusions:
    - burn, lock/vesting, bridge/router, staking pools (always)
    - treasury (conditional), cex custody (conditional)

    Args:
        inputs: FreeFloatInputs

    Returns:
        Free-float supply at t0.
    """

    exclude = set(EXCLUDE_LABELS_BASE)
    if inputs.exclude_treasury:
        exclude.add("treasury")
    if inputs.exclude_cex_custody:
        exclude.add("cex_custody")

    excluded_sum = 0.0
    for h in inputs.holdings_t0:
        if h.label and h.label in exclude:
            excluded_sum += float(h.balance)

    free_float = float(inputs.total_supply_t0) - excluded_sum
    if free_float < 0:
        free_float = 0.0
    return free_float


def sum_holdings_by_label(holdings: Iterable[Holding], label: str) -> float:
    """Sum balances for a given label at t0."""

    return float(sum(h.balance for h in holdings if h.label == label))


def team_holdings_t0(holdings: Iterable[Holding]) -> float:
    """Convenience to compute team/research foundation holdings at t0."""

    return sum_holdings_by_label(holdings, "team")


def insiders_holdings_t0(holdings: Iterable[Holding]) -> float:
    """Convenience to compute total insiders (team + vc + labeled whales) at t0."""

    return float(
        sum(h.balance for h in holdings if h.label in {"team", "vc", "labeled_whale"})
    )

