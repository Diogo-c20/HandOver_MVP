import requests
from typing import Optional, List, Dict, Any

from config import SOLSCAN_API_KEY, HELIUS_API_KEY

SOLSCAN_BASE = "https://pro-api.solscan.io"
HELIUS_BASE = "https://api.helius.xyz"


def solscan_get_token_holders(mint: str, limit: int = 50, offset: int = 0, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    if not SOLSCAN_API_KEY:
        return []
    sess = session or requests.Session()
    url = f"{SOLSCAN_BASE}/v1.0/token/holders"
    headers = {"token": SOLSCAN_API_KEY}
    try:
        resp = sess.get(url, headers=headers, params={"tokenAddress": mint, "offset": offset, "limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        items = data.get("data") or data.get("holders") or []
        out = []
        for it in items:
            addr = it.get("owner") or it.get("address")
            amt = it.get("amount") or it.get("balance")
            out.append({"address": addr, "balance": amt})
        return out
    except requests.exceptions.RequestException:
        return []


def helius_get_token_holders(mint: str, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    if not HELIUS_API_KEY:
        return []
    sess = session or requests.Session()
    url = f"{HELIUS_BASE}/v0/token-holders?api-key={HELIUS_API_KEY}&mint={mint}"
    try:
        resp = sess.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        items = data.get("tokenHolders") or data.get("result") or data or []
        out = []
        for it in items:
            addr = it.get("owner") or it.get("address")
            amt = it.get("amount") or it.get("balance")
            out.append({"address": addr, "balance": amt})
        return out
    except requests.exceptions.RequestException:
        return []
