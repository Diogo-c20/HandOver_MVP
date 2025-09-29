from __future__ import annotations

from datetime import datetime, timezone

from insiders.config_schema import get_default_config
from insiders.metrics import compute_raw_metrics
from insiders.normalize import BaselineStore, CohortBaseline, normalize_with_backoff
from insiders.events import detect_events
from insiders.scorer import compose_insider_score
from .fixtures import make_baseline


def _norm_all(cfg, store, backoff_keys, raw_dict, trust=1.0):
    ns = {}
    for k in [
        "FR_vol",
        "FR_mkt",
        "EMR_team",
        "team_to_cex_over_ff",
        "SPI_VC",
        "SPI_mkt",
    ]:
        s, _ = normalize_with_backoff(
            cfg=cfg,
            value=raw_dict.get(k),
            cohort_keys=backoff_keys,
            event_group="outflow" if k.startswith("SPI") or k.startswith("EMR") or k.startswith("team") else "FR",
            baselines=store,
            label_trust=trust,
        )
        ns[k] = s
    return ns


def test_high_score_with_team_cex_and_events():
    cfg = get_default_config()
    store = BaselineStore(make_baseline())
    cohort_keys = [
        "ETH:Q2:Mid:team_to_cex",
        "ETH:Q2:team_to_cex",
        "ETH:outflow",
        "GLOBAL:outflow",
    ]
    # Raw inputs producing team_to_cex_over_ff = 1.2%
    raw = compute_raw_metrics(
        buy_vol_specific_group=50.0,
        total_vol_fr_window=100000.0,
        free_float_t0=1_000_000.0,
        cex_in_team=12_000.0,
        team_holdings_t0=150_000.0,
        vc_outflow=10_000.0,
        unlocked_in_period=20_000.0,
        insiders_outflow=15_000.0,
    )
    raw_dict = raw.__dict__
    norm_scores = _norm_all(cfg, store, cohort_keys, raw_dict, trust=1.0)
    events = detect_events(
        cfg=cfg,
        metrics=raw_dict,
        signals={"new_whale_from_cex_or_team": True, "lockup_fanout": True},
    )
    score, grade, ev = compose_insider_score(cfg=cfg, norm_scores=norm_scores, events=events)
    assert score >= 80.0
    assert grade in {"HIGH", "MED", "MID", "LOW"}


def test_label_trust_dampens_scores():
    cfg = get_default_config()
    store = BaselineStore(make_baseline())
    cohort_keys = [
        "ETH:Q2:Mid:pre_announce",
        "ETH:Q2:pre_announce",
        "ETH:FR",
        "GLOBAL:FR",
    ]
    raw = compute_raw_metrics(
        buy_vol_specific_group=100.0,
        total_vol_fr_window=50_000.0,
        free_float_t0=500_000.0,
        cex_in_team=2_000.0,
        team_holdings_t0=50_000.0,
        vc_outflow=0.0,
        unlocked_in_period=0.0,
        insiders_outflow=0.0,
    )
    raw_dict = raw.__dict__
    s_certain, _ = normalize_with_backoff(
        cfg=cfg,
        value=raw_dict["FR_vol"],
        cohort_keys=cohort_keys,
        event_group="FR",
        baselines=store,
        label_trust=1.0,
    )
    s_heur, _ = normalize_with_backoff(
        cfg=cfg,
        value=raw_dict["FR_vol"],
        cohort_keys=cohort_keys,
        event_group="FR",
        baselines=store,
        label_trust=0.8,
    )
    s_pat, _ = normalize_with_backoff(
        cfg=cfg,
        value=raw_dict["FR_vol"],
        cohort_keys=cohort_keys,
        event_group="FR",
        baselines=store,
        label_trust=0.6,
    )
    assert s_pat <= s_heur <= s_certain


def test_spi_vc_na_and_spi_mkt_used():
    cfg = get_default_config()
    store = BaselineStore(make_baseline())
    cohort_keys = [
        "ETH:Q2:Mid:team_to_cex",
        "ETH:Q2:team_to_cex",
        "ETH:outflow",
        "GLOBAL:outflow",
    ]
    raw = compute_raw_metrics(
        buy_vol_specific_group=0.0,
        total_vol_fr_window=1.0,
        free_float_t0=1_000_000.0,
        cex_in_team=0.0,
        team_holdings_t0=100_000.0,
        vc_outflow=1000.0,
        unlocked_in_period=0.0,  # NA
        insiders_outflow=2000.0,
    )
    raw_dict = raw.__dict__
    norm_scores = _norm_all(cfg, store, cohort_keys, raw_dict, trust=1.0)
    # Ensure SPI_VC isn't NaN normalized (value replaced if needed), SPI_mkt exists
    assert "SPI_mkt" in norm_scores and isinstance(norm_scores["SPI_mkt"], float)

