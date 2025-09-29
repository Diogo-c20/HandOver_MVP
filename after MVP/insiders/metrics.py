from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math


@dataclass
class RawMetrics:
    """Raw market-impact oriented metrics prior to normalization.

    Attributes are expressed as ratios relative to denominators fixed at t0
    where applicable.
    """

    FR_vol: Optional[float]
    FR_mkt: Optional[float]
    EMR_team: Optional[float]
    team_to_cex_over_ff: Optional[float]
    SPI_VC: Optional[float]
    SPI_mkt: Optional[float]


def safe_div(n: float, d: float) -> Optional[float]:
    """Divide with None on zero denominator."""

    if d is None or d == 0:
        return None
    return n / d


def compute_fr_vol(buy_vol_specific_group: float, total_vol: float) -> Optional[float]:
    """FR_vol = buy_vol(specific_group, [T-72h, T-12h]) / total_vol([T-72h, T-12h])"""

    return safe_div(buy_vol_specific_group, total_vol)


def compute_fr_mkt(buy_vol_specific_group: float, free_float_t0: float) -> Optional[float]:
    """FR_mkt = buy_vol(specific_group, [T-72h, T-12h]) / free_float_t0"""

    return safe_div(buy_vol_specific_group, free_float_t0)


def compute_emr_team(cex_in_team: float, team_holdings_t0: float) -> Optional[float]:
    """EMR_team = cex_in(team_wallets, N_days) / team_holdings_t0"""

    return safe_div(cex_in_team, team_holdings_t0)


def compute_team_to_cex_over_ff(
    cex_in_team: float, free_float_t0: float
) -> Optional[float]:
    """team_to_cex_over_ff = cex_in_team / free_float_t0

    Derived from EMR_team * (team_holdings_t0 / free_float_t0) simplifying out
    team_holdings_t0.
    """

    return safe_div(cex_in_team, free_float_t0)


def compute_spi_vc(vc_outflow: float, unlocked_in_period: float) -> Optional[float]:
    """SPI_VC = outflow(vc_group, N_days) / unlocked_in_period(N_days)

    Returns None (NA) when unlocked_in_period == 0.
    """

    return safe_div(vc_outflow, unlocked_in_period)


def compute_spi_mkt(insiders_outflow: float, free_float_t0: float) -> Optional[float]:
    """SPI_mkt = outflow(insiders_all, N_days) / free_float_t0"""

    return safe_div(insiders_outflow, free_float_t0)


def compute_raw_metrics(
    *,
    buy_vol_specific_group: float,
    total_vol_fr_window: float,
    free_float_t0: float,
    cex_in_team: float,
    team_holdings_t0: float,
    vc_outflow: float,
    unlocked_in_period: float,
    insiders_outflow: float,
) -> RawMetrics:
    """Compute all raw metrics using provided inputs.

    All denominators are anchored at t0 where ratios apply.
    """

    FR_vol = compute_fr_vol(buy_vol_specific_group, total_vol_fr_window)
    FR_mkt = compute_fr_mkt(buy_vol_specific_group, free_float_t0)
    EMR_team = compute_emr_team(cex_in_team, team_holdings_t0)
    team_ff = compute_team_to_cex_over_ff(cex_in_team, free_float_t0)
    SPI_VC = compute_spi_vc(vc_outflow, unlocked_in_period)
    SPI_mkt = compute_spi_mkt(insiders_outflow, free_float_t0)
    return RawMetrics(
        FR_vol=FR_vol,
        FR_mkt=FR_mkt,
        EMR_team=EMR_team,
        team_to_cex_over_ff=team_ff,
        SPI_VC=SPI_VC,
        SPI_mkt=SPI_mkt,
    )

