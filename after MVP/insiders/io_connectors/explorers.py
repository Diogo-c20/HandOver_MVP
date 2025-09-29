"""Explorer parsers for labels/vesting/timelocks (mock-friendly)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_address_labels(address: str) -> Dict[str, str]:
    """Return label metadata for an address (mock)."""
    raise NotImplementedError("This is a mock interface.")


def get_vesting_schedules(token_address: str) -> List[Dict[str, Any]]:
    """Return vesting schedules for a token (mock)."""
    raise NotImplementedError("This is a mock interface.")

