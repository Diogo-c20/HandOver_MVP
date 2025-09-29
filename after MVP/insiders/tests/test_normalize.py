from __future__ import annotations

from datetime import datetime, timezone

from insiders.config_schema import get_default_config
from insiders.normalize import BaselineStore, normalize_with_backoff
from .fixtures import make_baseline


def test_fr_na_replaced_with_cohort_median():
    cfg = get_default_config()
    store = BaselineStore(make_baseline())
    # FR metric NA should be replaced with cohort median, score around 50
    keys = [
        "ETH:Q2:Mid:pre_announce",
        "ETH:Q2:pre_announce",
        "ETH:FR",
        "GLOBAL:FR",
    ]
    score, details = normalize_with_backoff(
        cfg=cfg,
        value=None,
        cohort_keys=keys,
        event_group="FR",
        baselines=store,
        label_trust=1.0,
    )
    assert details["replaced_with_median"] == 1.0
    assert 35 <= score <= 65

