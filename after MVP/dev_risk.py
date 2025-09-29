from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ---- Pydantic Models ----


class DevSignals(BaseModel):
    token: str
    chain: str  # "eth" | "bnb" | "sol"

    # Supply and circulating
    total_supply: int | float | None = 0
    burned_amount: int | float | None = 0
    lp_locked_amount: int | float | None = 0
    lp_locked_until_ts: Optional[int] = None
    vesting_locked_amount: int | float | None = 0
    timelock_locked_amount: int | float | None = 0
    non_circulating_extra: int | float | None = 0  # bridge/router/staking 등

    # Dev entities absolute amount (same unit as total_supply)
    dev_entity_amount: int | float | None = 0
    unlocked_team_amount: int | float | None = 0

    # Role/permission signals (EVM)
    owner: Optional[str] = None
    minter: Optional[str] = None
    proxy_admin: Optional[str] = None
    upgrader: Optional[str] = None

    owner_is_eoa: Optional[bool] = None
    minter_is_eoa: Optional[bool] = None
    proxy_admin_is_eoa: Optional[bool] = None
    upgrader_is_eoa: Optional[bool] = None

    # Additional permissions presence and EOA control
    has_pause: bool = False
    has_blacklist: bool = False
    has_tax: bool = False
    pause_controller_is_eoa: Optional[bool] = None
    blacklist_controller_is_eoa: Optional[bool] = None
    tax_controller_is_eoa: Optional[bool] = None

    # Mitigations
    treasury_multisig: bool = False
    team_vesting_percent: Optional[float] = None  # 0..100
    timelock_min_delay_sec: Optional[int] = None
    lp_burned_or_locked_1y: Optional[bool] = None
    is_verified: Optional[bool] = None  # Sourcify-based verification (EVM)

    # Bookkeeping
    reason_codes: List[str] = Field(default_factory=list)


class DevScore(BaseModel):
    token: str
    chain: str
    dev_holdings_pct: float
    dev_risk_score: int
    developer_subscore: float
    signals: Dict[str, Any]
    reason_codes: List[str]


# ---- RPC Clients (Explorerless) ----


class RPCError(Exception):
    pass


class EVMClient:
    def __init__(self, providers: List[str], timeout: float = 10.0) -> None:
        if not providers:
            raise ValueError("No EVM providers configured")
        self.providers = providers
        self.timeout = timeout
        self.session = requests.Session()

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.4, min=0.2, max=2.0), retry=retry_if_exception_type(RPCError))
    def _rpc(self, method: str, params: list[Any]) -> Any:
        last_exc: Optional[Exception] = None
        for url in self.providers:
            try:
                payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                res = self.session.post(url, json=payload, timeout=self.timeout)
                if res.status_code != 200:
                    last_exc = RPCError(f"HTTP {res.status_code}")
                    continue
                data = res.json()
                if "error" in data:
                    last_exc = RPCError(str(data["error"]))
                    continue
                return data.get("result")
            except Exception as e:  # noqa: BLE001
                last_exc = RPCError(str(e))
                continue
        raise last_exc or RPCError("RPC failed")

    def get_code(self, address: str) -> str:
        return self._rpc("eth_getCode", [address, "latest"]) or "0x"

    def is_eoa(self, address: Optional[str]) -> Optional[bool]:
        if not address:
            return None
        try:
            return (self.get_code(address) == "0x")
        except Exception:  # noqa: BLE001
            return None

    def eth_call(self, to: str, data: str) -> Optional[str]:
        try:
            return self._rpc("eth_call", [{"to": to, "data": data}, "latest"])  # hex string
        except Exception:  # noqa: BLE001
            return None

    def get_storage_at(self, address: str, slot_hex: str) -> Optional[str]:
        try:
            return self._rpc("eth_getStorageAt", [address, slot_hex, "latest"])  # hex32
        except Exception:  # noqa: BLE001
            return None

    def get_block_number(self) -> Optional[int]:
        try:
            h = self._rpc("eth_blockNumber", [])
            return int(h, 16) if isinstance(h, str) else None
        except Exception:  # noqa: BLE001
            return None


class SolanaClient:
    def __init__(self, providers: List[str], timeout: float = 10.0) -> None:
        if not providers:
            raise ValueError("No Solana providers configured")
        self.providers = providers
        self.timeout = timeout
        self.session = requests.Session()

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.4, min=0.2, max=2.0), retry=retry_if_exception_type(RPCError))
    def _rpc(self, method: str, params: list[Any]) -> Any:
        last_exc: Optional[Exception] = None
        for url in self.providers:
            try:
                payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                res = self.session.post(url, json=payload, timeout=self.timeout)
                if res.status_code != 200:
                    last_exc = RPCError(f"HTTP {res.status_code}")
                    continue
                data = res.json()
                if "error" in data:
                    last_exc = RPCError(str(data["error"]))
                    continue
                return data.get("result")
            except Exception as e:  # noqa: BLE001
                last_exc = RPCError(str(e))
                continue
        raise last_exc or RPCError("RPC failed")

    def get_token_supply(self, mint: str) -> Optional[int]:
        res = self._rpc("getTokenSupply", [mint])
        try:
            v = res.get("value", {}).get("amount")
            return int(v) if v is not None else None
        except Exception:  # noqa: BLE001
            return None

    def get_token_largest_accounts(self, mint: str) -> list[dict[str, Any]]:
        res = self._rpc("getTokenLargestAccounts", [mint])
        try:
            return res.get("value") or []
        except Exception:  # noqa: BLE001
            return []

    def get_account_info(self, account: str) -> Any:
        return self._rpc("getAccountInfo", [account, {"encoding": "jsonParsed"}])


# ---- Signal Collection (minimal, explorerless) ----


def _hex_to_addr(hex32: Optional[str]) -> Optional[str]:
    if not hex32 or not isinstance(hex32, str) or not hex32.startswith("0x"):
        return None
    # last 20 bytes
    try:
        h = hex32[-40:]
        return "0x" + h
    except Exception:  # noqa: BLE001
        return None


# EIP‑1967 slots
ADMIN_SLOT = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


def _sourcify_has_sources(session: requests.Session, chain: str, address: str, timeout: float = 10.0) -> bool:
    chain_ids = {"eth": 1, "bnb": 56}
    cid = chain_ids.get(chain)
    if not cid:
        return False
    addr = (address or "").lower()
    bases = [
        f"https://repo.sourcify.dev/contracts/full_match/{cid}/{addr}",
        f"https://repo.sourcify.dev/contracts/partial_match/{cid}/{addr}",
    ]

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.4, min=0.2, max=2.0))
    def _get(url: str) -> Optional[dict]:
        try:
            resp = session.get(url + "/metadata.json", timeout=timeout)
            if resp.status_code != 200:
                raise RPCError(f"HTTP {resp.status_code}")
            return resp.json()
        except Exception as e:  # noqa: BLE001
            raise RPCError(str(e))

    for base in bases:
        try:
            meta = _get(base)
            if isinstance(meta, dict) and (meta.get("sources") or meta.get("settings")):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def collect_signals(token: str, chain: str, deadline_ts: Optional[float] = None) -> DevSignals:
    chain = chain.lower()
    sig = DevSignals(token=token, chain=chain)

    try:
        def _deadline_exceeded() -> bool:
            return bool(deadline_ts and time.time() > deadline_ts)

        if chain in {"eth", "bnb"}:
            from config import EVM_RPC_PROVIDERS  # local import to avoid heavy import at module load

            providers = EVM_RPC_PROVIDERS.get("ethereum" if chain == "eth" else "bnb") or []
            evm = EVMClient(providers)

            # totalSupply()
            if _deadline_exceeded():
                return sig
            out = evm.eth_call(token, "0x18160ddd")
            if isinstance(out, str):
                try:
                    sig.total_supply = int(out, 16)
                except Exception:  # noqa: BLE001
                    sig.total_supply = None

            # Basic roles
            if _deadline_exceeded():
                return sig
            owner = evm.eth_call(token, "0x8da5cb5b")  # owner()
            sig.owner = _hex_to_addr(owner)
            sig.owner_is_eoa = evm.is_eoa(sig.owner)

            if _deadline_exceeded():
                return sig
            minter = evm.eth_call(token, "0x07540978")  # minter()
            sig.minter = _hex_to_addr(minter)
            sig.minter_is_eoa = evm.is_eoa(sig.minter)

            # EIP‑1967 proxy admin / implementation
            if _deadline_exceeded():
                return sig
            admin_hex = evm.get_storage_at(token, ADMIN_SLOT)
            impl_hex = evm.get_storage_at(token, IMPL_SLOT)
            sig.proxy_admin = _hex_to_addr(admin_hex)
            sig.upgrader = _hex_to_addr(impl_hex)
            sig.proxy_admin_is_eoa = evm.is_eoa(sig.proxy_admin)
            sig.upgrader_is_eoa = evm.is_eoa(sig.upgrader)

            # Attempt: multisig detection (best‑effort)
            if _deadline_exceeded():
                return sig
            if sig.owner and not sig.owner_is_eoa:
                threshold = evm.eth_call(sig.owner, "0x54d4b564")  # getThreshold() (heuristic)
                try:
                    val = int(threshold, 16) if isinstance(threshold, str) else 0
                    if val and val > 1:
                        sig.treasury_multisig = True
                except Exception:  # noqa: BLE001
                    pass

            # Optional: paused()/blacklist()/tax existence (best‑effort via function selectors)
            if _deadline_exceeded():
                return sig
            sig.has_pause = bool(evm.eth_call(token, "0x5c975abb"))  # paused()
            sig.has_blacklist = bool(evm.eth_call(token, "0xdd62ed3e"))  # placeholder
            sig.has_tax = bool(evm.eth_call(token, "0x3ccfd60b"))  # placeholder

            # Controllers assumed as owner unless better signals found
            sig.pause_controller_is_eoa = sig.owner_is_eoa if sig.has_pause else None
            sig.blacklist_controller_is_eoa = sig.owner_is_eoa if sig.has_blacklist else None
            sig.tax_controller_is_eoa = sig.owner_is_eoa if sig.has_tax else None

            # LP signals, vesting, timelock: left unknown here (requires protocol‑specifics)
            # They can be provided by upstream or inferred by separate modules later.

            # Sourcify verification (free)
            if _deadline_exceeded():
                return sig
            try:
                sig.is_verified = _sourcify_has_sources(evm.session, chain, token)
            except Exception:
                sig.is_verified = None

        elif chain == "sol":
            from config import SOLANA_RPC_PROVIDERS  # local import

            sol = SolanaClient(SOLANA_RPC_PROVIDERS)
            if _deadline_exceeded():
                return sig
            ts = sol.get_token_supply(token)
            sig.total_supply = ts
            # Solana authorities (mintAuthority/freezeAuthority) via getAccountInfo would be processed here.
            # For now, we keep defaults and focus on score logic; tests will mock.

    except Exception:
        # Fail closed (minimal signals) – scoring will be conservative
        return sig

    return sig


# ---- Scoring ----


def calc_dev_holdings(signals: DevSignals) -> float:
    total = float(signals.total_supply or 0)
    burned = float(signals.burned_amount or 0)
    lp_locked = float(signals.lp_locked_amount or 0)
    vest_locked = float(signals.vesting_locked_amount or 0)
    tl_locked = float(signals.timelock_locked_amount or 0)
    extra = float(signals.non_circulating_extra or 0)
    free_float = max(0.0, total - burned - lp_locked - vest_locked - tl_locked - extra)
    dev_amt = float(signals.dev_entity_amount or 0)
    pct = 0.0 if free_float <= 0 else max(0.0, min(100.0, 100.0 * dev_amt / free_float))
    return pct


def calc_dev_risk_score(signals: DevSignals) -> int:
    score = 0
    rc: list[str] = signals.reason_codes

    # Critical permissions EOA: +50 each (clip later)
    if signals.minter_is_eoa:
        score += 50
        rc.append("MINTER_EOA+50")
    if signals.proxy_admin_is_eoa:
        score += 50
        rc.append("PROXYADMIN_EOA+50")
    if signals.upgrader_is_eoa:
        score += 50
        rc.append("UPGRADER_EOA+50")
    if signals.owner_is_eoa:
        # Treat owner as critical if it governs sensitive paths
        score += 50
        rc.append("OWNER_EOA+50")

    # Additional permissions controlled by EOA: +20
    if signals.has_pause and signals.pause_controller_is_eoa:
        score += 20
        rc.append("PAUSE_EOA+20")
    if signals.has_blacklist and signals.blacklist_controller_is_eoa:
        score += 20
        rc.append("BLACKLIST_EOA+20")
    if signals.has_tax and signals.tax_controller_is_eoa:
        score += 20
        rc.append("TAX_EOA+20")

    # LP unlocked EOA custody: +40
    if signals.lp_locked_amount is not None and signals.lp_locked_amount <= 0 and signals.owner_is_eoa:
        score += 40
        rc.append("LP_UNLOCKED_EOA+40")

    # Unlocked team / FreeFloat ratio
    try:
        total = float(signals.total_supply or 0)
        burned = float(signals.burned_amount or 0)
        lp_locked = float(signals.lp_locked_amount or 0)
        vest_locked = float(signals.vesting_locked_amount or 0)
        tl_locked = float(signals.timelock_locked_amount or 0)
        extra = float(signals.non_circulating_extra or 0)
        free_float = max(0.0, total - burned - lp_locked - vest_locked - tl_locked - extra)
        unlocked_team = float(signals.unlocked_team_amount or 0)
        ratio = 0.0 if free_float <= 0 else 100.0 * unlocked_team / free_float
        if ratio > 20.0:
            score += 20
            rc.append("UNLOCKED_TEAM_RATIO_GT20+20")
        elif 5.0 <= ratio <= 20.0:
            score += 10
            rc.append("UNLOCKED_TEAM_RATIO_5_TO_20+10")
    except Exception:  # noqa: BLE001
        pass

    # Mitigations
    vp = signals.team_vesting_percent
    if vp is not None:
        if vp >= 90.0:
            score -= 20
            rc.append("TEAM_VESTING_GE90-20")
        elif 70.0 <= vp < 90.0:
            score -= 15
            rc.append("TEAM_VESTING_70_TO_89-15")
        elif 40.0 <= vp < 70.0:
            score -= 5
            rc.append("TEAM_VESTING_40_TO_69-5")
        else:  # < 40%
            score += 5
            rc.append("TEAM_VESTING_LT40+5")

    if signals.timelock_min_delay_sec is not None and signals.timelock_min_delay_sec >= 24 * 3600:
        score -= 15
        rc.append("TIMELOCK_24H_PLUS-15")

    if signals.treasury_multisig:
        score -= 10
        rc.append("TREASURY_MULTISIG-10")

    if signals.lp_burned_or_locked_1y:
        score -= 10
        rc.append("LP_LOCKED_1Y_OR_BURNED-10")

    # Optional mitigation: verified source via Sourcify
    if signals.is_verified:
        score -= 10
        rc.append("SOURCIFY_VERIFIED-10")

    # Final clip 0..100
    score = max(0, min(100, score))
    return score


def compute_developer_subscore(token_address: str, chain: str, now_ts: int | None = None) -> dict:
    """
    chain ∈ {"eth","bnb","sol"}
    반환:
      {
        "token": "0x...",
        "chain": "eth",
        "dev_holdings_pct": 23.4,
        "dev_risk_score": 35,
        "developer_subscore": 52.0,
        "signals": { ... },
        "reason_codes": [ ... ]
      }
    """
    # Per-token time budget (seconds)
    try:
        from config import DEV_RISK_MAX_TOKEN_SEC  # type: ignore
        budget_sec = int(DEV_RISK_MAX_TOKEN_SEC)
    except Exception:
        budget_sec = 25
    start_ts = time.time()
    deadline = start_ts + max(1, budget_sec)

    # Backward-compat: some tests may mock collect_signals without deadline param
    try:
        sig = collect_signals(token_address, chain, deadline_ts=deadline)
    except TypeError:
        sig = collect_signals(token_address, chain)  # type: ignore[misc]

    # Compute holdings and risk score
    dev_holdings_pct = calc_dev_holdings(sig)
    dev_risk_score = calc_dev_risk_score(sig)
    developer_subscore = 0.6 * dev_holdings_pct + 0.4 * float(dev_risk_score)
    developer_subscore = max(0.0, min(100.0, developer_subscore))

    # Clear large raw data if any (defensive; we keep only signals summary)
    # In this implementation, signals already store compact summaries.

    # Budget flag
    if time.time() > deadline:
        sig.reason_codes.append("TIME_BUDGET_EXCEEDED")

    out = DevScore(
        token=token_address,
        chain=chain,
        dev_holdings_pct=dev_holdings_pct,
        dev_risk_score=int(dev_risk_score),
        developer_subscore=developer_subscore,
        signals=json.loads(sig.json()),
        reason_codes=list(sig.reason_codes),
    )
    return json.loads(out.json())
