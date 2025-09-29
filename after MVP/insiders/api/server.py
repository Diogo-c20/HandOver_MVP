from __future__ import annotations

from datetime import datetime
from typing import Dict

from fastapi import FastAPI

from ..config_schema import get_default_config
from ..metrics import compute_raw_metrics
from ..normalize import BaselineStore, CohortBaseline, normalize_with_backoff
from ..events import detect_events
from ..scorer import compose_insider_score
from .schemas import NormalizeRequest, ScoreRequest, ScoreResponse


app = FastAPI(title="Insiders Risk Scoring API")


@app.post("/score/insiders", response_model=ScoreResponse)
def score_insiders(req: ScoreRequest) -> ScoreResponse:
    cfg = get_default_config()

    # Compute raw metrics
    raw = compute_raw_metrics(
        buy_vol_specific_group=req.metrics.buy_vol_specific_group,
        total_vol_fr_window=req.metrics.total_vol_fr_window,
        free_float_t0=req.metrics.free_float_t0,
        cex_in_team=req.metrics.cex_in_team,
        team_holdings_t0=req.metrics.team_holdings_t0,
        vc_outflow=req.metrics.vc_outflow,
        unlocked_in_period=req.metrics.unlocked_in_period,
        insiders_outflow=req.metrics.insiders_outflow,
    )
    raw_dict = raw.__dict__

    # Build baseline store
    store = BaselineStore(
        data={
            b.key: CohortBaseline(samples=b.samples, updated_at=b.updated_at)
            for b in req.baseline_sets
        }
    )

    # Cohort keys per backoff policy
    cohort_specific = f"{req.chain}:{req.ff_bin}:{req.mcap_bin}:{req.event_type}"
    cohort_chain_ff_event = f"{req.chain}:{req.ff_bin}:{req.event_type}"
    cohort_chain_group = f"{req.chain}:{req.event_group}"
    cohort_global_group = f"GLOBAL:{req.event_group}"
    backoff_keys = [
        cohort_specific,
        cohort_chain_ff_event,
        cohort_chain_group,
        cohort_global_group,
    ]

    # Normalize each metric; apply label trust weight
    trust = float(cfg.label_trust.get(req.label_quality, 1.0))
    norm_scores: Dict[str, float] = {}
    explanations: list[str] = []
    for key in [
        "FR_vol",
        "FR_mkt",
        "EMR_team",
        "team_to_cex_over_ff",
        "SPI_VC",
        "SPI_mkt",
    ]:
        val = raw_dict.get(key)
        score, details = normalize_with_backoff(
            cfg=cfg,
            value=val,
            cohort_keys=backoff_keys,
            event_group=req.event_group,
            baselines=store,
            label_trust=trust,
        )
        norm_scores[key] = score
        if details.get("replaced_with_median", 0.0) == 1.0:
            explanations.append(f"{key}: NA replaced with cohort median")

    # Events
    events = detect_events(cfg=cfg, metrics=raw_dict, signals=req.signals)
    # Compose score
    insider_score, grade, ev_contrib = compose_insider_score(
        cfg=cfg, norm_scores=norm_scores, events=events
    )
    if ev_contrib != 0.0:
        explanations.append(f"Events contribution: {ev_contrib:+.1f}")

    return ScoreResponse(
        cohort_id=cohort_specific,
        raw_metrics=raw_dict,
        norm_scores=norm_scores,
        events=[{"key": e.key, "reason": e.reason} for e in events],
        insider_score=insider_score,
        grade=grade,
        explanations=explanations,
    )

