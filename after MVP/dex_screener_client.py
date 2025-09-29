import time
import requests
import pandas as pd
from typing import Dict, Any, Optional, List

DEX_API_BASE = "https://api.dexscreener.com"
_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SEC = 600  # 10분


def _map_chain(chain_id: str) -> Optional[str]:
    mapping = {
        "ethereum": "ethereum",
        "bsc": "bnb",
        "bnb": "bnb",
        "solana": "solana",
        # extend as needed: polygon, arbitrum, base, etc.
    }
    return mapping.get((chain_id or "").lower())


def search_pairs(keyword: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Search pairs with ≤10s timeout and ≤3 retries."""
    sess = session or requests.Session()
    url = f"{DEX_API_BASE}/latest/dex/search"
    last_err = None
    for i, to in enumerate((5, 7, 10), start=1):
        try:
            resp = sess.get(url, params={"q": keyword}, timeout=to)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.3 * i)
            continue
    if last_err:
        raise last_err
    return {"pairs": []}


def search_pairs_cached(keyword: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    key = (keyword or "").strip().lower()
    now = time.time()
    entry = _SEARCH_CACHE.get(key)
    if entry and (now - entry.get("ts", 0)) < _CACHE_TTL_SEC:
        return entry["data"]
    data = search_pairs(keyword, session)
    _SEARCH_CACHE[key] = {"ts": now, "data": data}
    return data


def tokens_from_pairs_json(data: Dict[str, Any]) -> pd.DataFrame:
    pairs = (data or {}).get("pairs", []) or []
    rows: List[Dict[str, Any]] = []
    for p in pairs:
        chain = _map_chain(p.get("chainId"))
        if not chain:
            continue
        base = p.get("baseToken") or {}
        quote = p.get("quoteToken") or {}
        if base.get("address"):
            rows.append({
                "blockchain": chain,
                "address": base.get("address"),
                "name": base.get("name"),
                "symbol": base.get("symbol"),
            })
        if quote.get("address"):
            rows.append({
                "blockchain": chain,
                "address": quote.get("address"),
                "name": quote.get("name"),
                "symbol": quote.get("symbol"),
            })
    if not rows:
        return pd.DataFrame(columns=["blockchain", "address", "name", "symbol"])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["address"]).reset_index(drop=True)
    return df[["blockchain", "address", "name", "symbol"]]


def _to_ds_chain(chain: str) -> Optional[str]:
    mapping = {"ethereum": "ethereum", "bnb": "bsc", "solana": "solana"}
    return mapping.get((chain or "").lower())


def _chunks(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def enrich_tokens(df: pd.DataFrame, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """DexScreener tokens API로 name/symbol 보강(최대 30개 배치).
    기존 name/symbol이 비어 있을 때만 채움. 오류 시 무시.
    """
    if df is None or df.empty:
        return df
    if "blockchain" not in df.columns or "address" not in df.columns:
        return df
    sess = session or requests.Session()
    out = df.copy()
    for chain, group in out.groupby(out["blockchain"].str.lower()):
        ds_chain = _to_ds_chain(chain)
        if not ds_chain:
            continue
        addrs = [a for a in group["address"].astype(str).tolist() if a]
        if not addrs:
            continue
        for chunk in _chunks(addrs, 30):
            try:
                url = f"{DEX_API_BASE}/tokens/v1/{ds_chain}/" + ",".join(chunk)
                resp = sess.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json() or {}
                items = data.get("tokens") or data.get("pairs") or data.get("data") or []
                mapping: Dict[str, Dict[str, Optional[str]]] = {}
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    addr = it.get("address") or (it.get("baseToken") or {}).get("address")
                    nm = it.get("name") or (it.get("baseToken") or {}).get("name")
                    sym = it.get("symbol") or (it.get("baseToken") or {}).get("symbol")
                    if addr:
                        mapping[addr] = {"name": nm, "symbol": sym}
                if not mapping:
                    continue
                mask = out["blockchain"].str.lower().eq(chain)
                for i in out[mask].index:
                    addr = out.at[i, "address"]
                    meta = mapping.get(addr)
                    if not meta:
                        continue
                    if (pd.isna(out.at[i, "name"]) or not str(out.at[i, "name"]).strip()) and meta.get("name"):
                        out.at[i, "name"] = meta["name"]
                    if (pd.isna(out.at[i, "symbol"]) or not str(out.at[i, "symbol"]).strip()) and meta.get("symbol"):
                        out.at[i, "symbol"] = meta["symbol"]
            except requests.exceptions.RequestException:
                continue
            except Exception:
                continue
    return out
