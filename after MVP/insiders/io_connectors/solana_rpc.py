"""Solana RPC (mock-friendly)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_token_accounts_by_owner(owner: str) -> List[Dict[str, Any]]:
    """Return token accounts owned by an address (mock)."""
    raise NotImplementedError("This is a mock interface.")

