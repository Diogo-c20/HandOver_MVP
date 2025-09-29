"""Dune query skeletons (mock-friendly).

These functions expose signatures and docstrings for unit tests to mock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def search_tokens_by_keyword(keyword: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Mockable function to search tokens by keyword in Dune.

    Args:
        keyword: Search keyword.
        api_key: Optional API key; unused in mock.

    Returns:
        List of token metadata dicts.
    """

    raise NotImplementedError("This is a mock interface.")

