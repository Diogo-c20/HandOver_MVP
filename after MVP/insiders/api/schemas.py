from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field


class MetricInputs(BaseModel):
    buy_vol_specific_group: float
    total_vol_fr_window: float
    free_float_t0: float
    cex_in_team: float
    team_holdings_t0: float
    vc_outflow: float
    unlocked_in_period: float
    insiders_outflow: float


class BaselineCohortIn(BaseModel):
    key: str
    samples: Sequence[float] = Field(default_factory=list)
    updated_at: datetime


class NormalizeRequest(BaseModel):
    chain: Literal["ETH", "BNB", "SOL"]
    ff_bin: Literal["Q1", "Q2", "Q3", "Q4"]
    mcap_bin: Literal["Micro", "Small", "Mid"]
    event_type: Literal["pre_announce", "team_to_cex", "vc_outflow"]
    event_group: Literal["FR", "outflow"]
    metric_key: Literal[
        "FR_vol",
        "FR_mkt",
        "EMR_team",
        "team_to_cex_over_ff",
        "SPI_VC",
        "SPI_mkt",
    ]
    value: Optional[float]
    label_trust: float = 1.0
    baselines: List[BaselineCohortIn]


class ScoreRequest(BaseModel):
    chain: Literal["ETH", "BNB", "SOL"]
    ff_bin: Literal["Q1", "Q2", "Q3", "Q4"]
    mcap_bin: Literal["Micro", "Small", "Mid"]
    event_type: Literal["pre_announce", "team_to_cex", "vc_outflow"]
    event_group: Literal["FR", "outflow"]
    label_quality: Literal["certain", "heuristic", "pattern"] = "certain"
    metrics: MetricInputs
    baseline_sets: List[BaselineCohortIn]
    signals: Dict[str, bool] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    cohort_id: str
    raw_metrics: Dict[str, Optional[float]]
    norm_scores: Dict[str, float]
    events: List[Dict[str, str]]
    insider_score: float
    grade: str
    explanations: List[str]

