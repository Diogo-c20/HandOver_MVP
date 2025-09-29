"""Alchemy EVM data (mock-friendly)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_transfers(address: str, start_block: Optional[int] = None, end_block: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return token transfers for an address (mock)."""
    raise NotImplementedError("This is a mock interface.")

