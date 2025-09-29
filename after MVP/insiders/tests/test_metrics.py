from __future__ import annotations

import math

from insiders.metrics import (
    compute_fr_mkt,
    compute_fr_vol,
    compute_raw_metrics,
    compute_spi_mkt,
    compute_spi_vc,
)


def test_spi_vc_na_when_unlocked_zero():
    assert compute_spi_vc(100.0, 0.0) is None


def test_fr_metrics_basic():
    fr_vol = compute_fr_vol(10.0, 1000.0)
    fr_mkt = compute_fr_mkt(10.0, 50000.0)
    assert fr_vol == 0.01
    assert fr_mkt == 10.0 / 50000.0


def test_compute_raw_metrics_all():
    raw = compute_raw_metrics(
        buy_vol_specific_group=100.0,
        total_vol_fr_window=10000.0,
        free_float_t0=1_000_000.0,
        cex_in_team=12_000.0,
        team_holdings_t0=200_000.0,
        vc_outflow=5000.0,
        unlocked_in_period=0.0,  # NA
        insiders_outflow=20_000.0,
    )
    assert raw.FR_vol == 0.01
    assert raw.FR_mkt == 0.0001
    assert raw.EMR_team == 0.06
    assert raw.team_to_cex_over_ff == 0.012
    assert raw.SPI_mkt == 0.02
    assert raw.SPI_VC is None

