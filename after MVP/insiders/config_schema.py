from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, validator


class WindowConfig(BaseModel):
    """Window configuration parameters."""

    fr_hours: Tuple[int, int] = Field(default=(72, 12))
    outflow_days: Tuple[int, int] = Field(default=(7, 30))


class WeightConfig(BaseModel):
    """Weights for normalized metrics in final score combination."""

    FR_vol: float = 15.0
    FR_mkt: float = 10.0
    EMR_team: float = 10.0
    team_to_cex_over_ff: float = 5.0
    SPI_VC: float = 10.0
    SPI_mkt: float = 5.0


class GradeThresholds(BaseModel):
    HIGH: float = 80.0
    MED: float = 60.0
    MID: float = 40.0


class ThresholdConfig(BaseModel):
    grade: GradeThresholds = Field(default_factory=GradeThresholds)


class NormalizationConfig(BaseModel):
    """Normalization policy parameters."""

    winsorize: Tuple[float, float] = Field(default=(1.0, 99.0))
    denom_anchor: Literal["window_start"] = "window_start"
    min_samples: int = 30
    alpha_k: int = 20
    half_life_days: int = 90
    backoff: List[str] = Field(
        default_factory=lambda: [
            "CHAIN:FF_BIN:MCAP_BIN:EVENT_TYPE",
            "CHAIN:FF_BIN:EVENT_TYPE",
            "CHAIN:EVENT_GROUP",
            "GLOBAL:EVENT_GROUP",
        ]
    )


class EventConfig(BaseModel):
    penalties: Dict[str, float] = Field(default_factory=dict)
    reliefs: Dict[str, float] = Field(default_factory=dict)


class Config(BaseModel):
    """Top-level configuration model for the insiders scoring engine."""

    windows: WindowConfig = Field(default_factory=WindowConfig)
    weights: WeightConfig = Field(default_factory=WeightConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    label_trust: Dict[str, float] = Field(
        default_factory=lambda: {"certain": 1.0, "heuristic": 0.8, "pattern": 0.6}
    )
    baseline_table_uri: str = "warehouse.insiders_baseline_v1"
    events: EventConfig = Field(default_factory=EventConfig)

    @validator("label_trust")
    def validate_label_trust(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k in v:
            if k not in {"certain", "heuristic", "pattern"}:
                raise ValueError(f"Invalid label trust key: {k}")
        return v


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file, allowing environment overrides.

    Environment variable overrides (optional):
    - DUNE_API_KEY, GEMINI_API_KEY, ALCHEMY_API_KEYS, ETHERSCAN_MULTICHAIN_API_KEY
      are referenced elsewhere but not required here.

    Args:
        path: Optional custom path to the YAML config.

    Returns:
        Parsed and validated Config object.
    """

    if path is None:
        # Default to package-local config_defaults.yaml
        path = Path(__file__).with_name("config_defaults.yaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Direct mapping into Pydantic models
    cfg = Config(**data)
    return cfg


# Convenience singleton loader (not module-global singleton to ease testing)
def get_default_config() -> Config:
    """Return a Config loaded from the default YAML file shipped with the package."""

    return load_config()

