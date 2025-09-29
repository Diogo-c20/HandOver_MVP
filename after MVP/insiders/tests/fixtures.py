from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from ..normalize import CohortBaseline


def make_baseline(now: datetime | None = None) -> Dict[str, CohortBaseline]:
    """Create simple synthetic baselines for tests across backoff levels."""

    if now is None:
        now = datetime.now(timezone.utc)
    # Create distributions where median ~ 0.005 and moderate spread
    rng = np.random.default_rng(42)
    samples_specific = list(np.clip(rng.normal(0.006, 0.003, size=200), 0, 0.05))
    samples_chain_ff_event = list(np.clip(rng.normal(0.005, 0.004, size=150), 0, 0.05))
    samples_chain_group = list(np.clip(rng.normal(0.004, 0.004, size=120), 0, 0.05))
    samples_global_group = list(np.clip(rng.normal(0.004, 0.005, size=300), 0, 0.05))

    return {
        "ETH:Q2:Mid:team_to_cex": CohortBaseline(samples_specific, now - timedelta(days=5)),
        "ETH:Q2:team_to_cex": CohortBaseline(samples_chain_ff_event, now - timedelta(days=10)),
        "ETH:outflow": CohortBaseline(samples_chain_group, now - timedelta(days=20)),
        "GLOBAL:outflow": CohortBaseline(samples_global_group, now - timedelta(days=40)),
        # FR group separate
        "ETH:Q2:Mid:pre_announce": CohortBaseline(samples_specific, now - timedelta(days=5)),
        "ETH:Q2:pre_announce": CohortBaseline(samples_chain_ff_event, now - timedelta(days=10)),
        "ETH:FR": CohortBaseline(samples_chain_group, now - timedelta(days=20)),
        "GLOBAL:FR": CohortBaseline(samples_global_group, now - timedelta(days=40)),
    }

