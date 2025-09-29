"""DEX Screener helper (mock-friendly)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_pair_activity(pair_id: str) -> Dict[str, Any]:
    """Return activity stats for a DEX pair (mock)."""
    raise NotImplementedError("This is a mock interface.")

