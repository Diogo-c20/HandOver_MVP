# 최종 자동화 파이프라인 스크립트 (v16: 총 발행량(Total Supply) 수집)

# 1. 라이브러리를 설치합니다.
# pip install dune-spice pandas requests

import os
import sys
import json
import time
import argparse
import pandas as pd
try:
    from dev_risk import compute_developer_subscore as dr_compute_developer_subscore  # type: ignore
except Exception:
    dr_compute_developer_subscore = None  # will fallback to local
import requests
try:
    import spice  # dune-spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
import dex_screener_client as ds
from decimal import Decimal
from datetime import datetime, timezone
from collections import defaultdict
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from solana_providers import helius_get_token_holders

# Sniper ML integration (optional)
SNIPER_DIR = os.path.expanduser("~/Desktop/Hyperindex/snipers")
if os.path.isdir(SNIPER_DIR) and SNIPER_DIR not in sys.path:
    sys.path.insert(0, SNIPER_DIR)
try:
    # Lazy imports inside function to avoid hard dependency at import-time
    from sniper_ml_pipeline import SniperScorer as _SniperScorer  # type: ignore
    from sniper_ml_pipeline import FeatureBuilder as _FeatureBuilder  # type: ignore
    from sniper_ml_pipeline import Config as _SniperConfig  # type: ignore
    _SNIPER_AVAILABLE = True
except Exception:
    _SNIPER_AVAILABLE = False

# --- ⚙️ 통합 설정 ⚙️ ---
from config import (
    ALCHEMY_API_KEYS,
    SEARCH_KEYWORD,
    SOLANA_RPC_PROVIDERS,
    EVM_RPC_PROVIDERS,
    RISK_WEIGHTS,
    SCAM_THRESHOLD,
)
DUNE_API_KEY_ENV = "DUNE_API_KEY"
CHAIN_RPC_URLS = {
    "ethereum": "https://eth-mainnet.g.alchemy.com/v2/",
    "bnb": "https://bsc-mainnet.g.alchemy.com/v2/",
}
SOLANA_RPC_BASE = "https://solana-mainnet.g.alchemy.com/v2/"  # legacy; prefer config.SOLANA_RPC_PROVIDERS
# Explorerless mode enforced: no Etherscan/BscScan/Solscan usage

# Chain IDs for Sourcify
CHAIN_IDS = {
    "ethereum": 1,
    "bnb": 56,
}

# --- 글로벌 변수 및 캐시 ---
FUNCTION_SIGNATURE_CACHE = {}
_BLOCK_TS_CACHE: dict[str, str] = {}

# --- 간단한 메트릭 및 레이트리미터 ---
METRICS = {
    "alchemy": {"calls": 0, "by_method": defaultdict(int), "estimated_cu": 0},
    "fourbyte": {"calls": 0},
    "dune": {"queries": 0},
    "errors": {"count": 0},
}
ALCHEMY_CU_MAP = {  # 추정치: 최신 정책에 맞춰 조정 필요
    "alchemy_getAssetTransfers": 10,
    "alchemy_getTokenHolders": 10,
    "eth_getTransactionByHash": 2,
    "eth_getTransactionReceipt": 2,
    "eth_getStorageAt": 2,
    "eth_call": 2,
}
_LAST_REQ_TS = 0.0
_MIN_REQ_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL_SEC", "0"))
_HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))
# Fallback tuning (env overrides)
_LOG_WINDOW_BLOCKS = int(os.getenv("LOG_FALLBACK_WINDOW_BLOCKS", "50000"))
_LOG_MAX_EVENTS = int(os.getenv("LOG_FALLBACK_MAX_EVENTS", "1000"))
_HOLDERS_FALLBACK_MAX_EVENTS = int(os.getenv("HOLDERS_FALLBACK_MAX_EVENTS", "5000"))
def _rate_limit():
    global _LAST_REQ_TS
    if _MIN_REQ_INTERVAL <= 0:
        return
    now = time.time()
    diff = now - _LAST_REQ_TS
    if diff < _MIN_REQ_INTERVAL:
        time.sleep(_MIN_REQ_INTERVAL - diff)
    _LAST_REQ_TS = time.time()

# --- 알려진 락커 컨트랙트 주소 (수동 관리) ---
# 여기에 TeamFinance, Unicrypt 등 주요 유동성 락커 컨트랙트 주소를 추가합니다.
# 주소는 소문자로 저장하는 것이 좋습니다.
# 실제 주소는 각 락커 서비스의 공식 문서를 참조하세요。
KNOWN_LOCKER_CONTRACTS = {
    "ethereum": [
        "0x0000000000000000000000000000000000000000", # 예시: TeamFinance (실제 주소 아님)
        "0x0000000000000000000000000000000000000000", # 예시: Unicrypt (실제 주소 아님)
    ],
    "bnb": [
        "0x0000000000000000000000000000000000000000", # 예시
    ],
}

from config import LABELS as LABELED_ADDRESSES

# ==============================================================================
# SECTION 0: HTTP 요청 헬퍼 함수
# ==============================================================================
def http_post(session, url, json_payload):
    try:
        _rate_limit()
        response = session.post(url, json=json_payload, timeout=_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json().get('result')
        # 메트릭 (Alchemy RPC)
        if "alchemy.com/v2/" in url:
            method = (json_payload or {}).get("method")
            METRICS["alchemy"]["calls"] += 1
            if method:
                METRICS["alchemy"]["by_method"][method] += 1
                METRICS["alchemy"]["estimated_cu"] += ALCHEMY_CU_MAP.get(method, 1)
        return data
    except requests.exceptions.RequestException:
        METRICS["errors"]["count"] += 1
        return None

# --- Provider-aware JSON-RPC helper with failover ---
def _supports_method(endpoint_url: str, method: str) -> bool:
    if method and method.startswith("alchemy_"):
        return "alchemy.com" in (endpoint_url or "")
    return True

def rpc_post_any(session, providers: list[str], payload: dict):
    method = (payload or {}).get("method")
    for idx, url in enumerate(providers or []):
        if not _supports_method(url, method):
            continue
        res = http_post(session, url, payload)
        if res is not None:
            if idx > 0:
                try:
                    print(f"    ↩️ RPC 페일오버 동작: {method} → {url}")
                except Exception:
                    pass
            return res
    return None

def http_get(session, url):
    try:
        _rate_limit()
        response = session.get(url, timeout=_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        # 메트릭 (탐색기/4byte)
        if "etherscan" in url or "bscscan" in url:
            METRICS["explorer"]["calls"] += 1
        if "4byte.directory" in url:
            METRICS["fourbyte"]["calls"] += 1
        return response.json()
    except requests.exceptions.RequestException:
        METRICS["errors"]["count"] += 1
        return None

def http_get_text(session, url):
    try:
        _rate_limit()
        response = session.get(url, timeout=_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        METRICS["errors"]["count"] += 1
        return None

def http_get_text(session, url):
    try:
        _rate_limit()
        response = session.get(url, timeout=_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        METRICS["errors"]["count"] += 1
        return None

# ==============================================================================
# SECTION 1: Dune에서 토큰 목록 검색
# ==============================================================================
DUNE_SQL_QUERY_TEMPLATE = """
SELECT
    'ethereum' AS blockchain,
    CAST(contract_address AS VARCHAR) AS address,
    name,
    symbol
FROM tokens.erc20
WHERE
    (LOWER(name) LIKE LOWER(CONCAT('%', '{keyword}', '%')) OR LOWER(symbol) LIKE LOWER(CONCAT('%', '{keyword}', '%')))
    AND blockchain = 'ethereum'

UNION ALL

SELECT
    'bnb' AS blockchain,
    CAST(contract_address AS VARCHAR) AS address,
    name,
    symbol
FROM tokens.erc20
WHERE
    (LOWER(name) LIKE LOWER(CONCAT('%', '{keyword}', '%')) OR LOWER(symbol) LIKE LOWER(CONCAT('%', '{keyword}', '%')))
    AND blockchain = 'bnb'

UNION ALL

SELECT
    'solana' AS blockchain,
    CAST(token_mint_address AS VARCHAR) AS address,
    name,
    symbol
FROM tokens_solana.fungible
WHERE
    LOWER(name) LIKE LOWER(CONCAT('%', '{keyword}', '%')) OR LOWER(symbol) LIKE LOWER(CONCAT('%', '{keyword}', '%'))

LIMIT 10
"""

def find_tokens_by_keyword(sql_template: str, keyword: str) -> pd.DataFrame | None:
    print(f"\n[단계 1] Dune에서 '{keyword}' 키워드로 토큰 목록 검색 시작 (dune-spice 사용)....")
    final_sql = sql_template.format(keyword=keyword.replace("'", "''"))
    try:
        results_df = pd.DataFrame(spice.query(final_sql))
        METRICS["dune"]["queries"] += 1
        results_df.columns = ['blockchain', 'address', 'name', 'symbol']
        print(f"✅ 총 {len(results_df)}개의 토큰을 찾았습니다.")
        return results_df
    except Exception as e:
        print(f"❌ Dune API 호출 중 오류: {e}")
        # Dune 오류 시에는 더미를 반환하지 않고, 상위 로직에서 DexScreener로 페일오버합니다.
        return None

def build_token_list_from_args(addresses_arg: str) -> pd.DataFrame:
    """Parse comma-separated addresses with optional chain prefix like 'ethereum:0x..'"""
    rows = []
    skipped = 0
    for raw in [x.strip() for x in addresses_arg.split(',') if x.strip()]:
        if ':' in raw:
            chain, addr = raw.split(':', 1)
        else:
            chain, addr = "ethereum", raw
        addr = addr.strip()
        if addr.endswith("...") or len(addr) != 42 or not addr.lower().startswith("0x"):
            print(f"    ⚠️ 잘못된 주소 형식 감지로 건너뜀: {raw}")
            skipped += 1
            continue
        rows.append({"blockchain": chain.lower(), "address": addr, "name": None, "symbol": None})
    df = pd.DataFrame(rows)
    if skipped:
        print(f"    ⚠️ 입력 주소 중 {skipped}개가 무시되었습니다. 전체 42자리 0x 주소를 사용하세요.")
    return df

# ==============================================================================
# SECTION 2: Alchemy에서 Raw Data 수집
# ==============================================================================
def get_function_signature(session, selector):
    if selector in FUNCTION_SIGNATURE_CACHE: return FUNCTION_SIGNATURE_CACHE[selector]
    if len(selector) != 10 or not selector.startswith('0x'): return None
    url = f"https://www.4byte.directory/api/v1/signatures/?hex_signature={selector}"
    response_data = http_get(session, url)
    if response_data and response_data.get('count', 0) > 0:
        signature = response_data['results'][0]['text_signature']
        FUNCTION_SIGNATURE_CACHE[selector] = signature
        return signature
    return None

def get_enriched_transfers(session, providers, contract_address, blockchain):
    print(f"    📥 전송 기록 및 함수 시그니처 수집 중...")
    enriched_transfers = []
    page_key = None
    item_count = 0
    # 1st: Alchemy Transfers (강화 API)
    while True:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [{"contractAddresses": [contract_address], "category": ["erc20"], "withMetadata": True, "maxCount": "0x3e8", "pageKey": page_key if page_key else None}]}
        response_data = rpc_post_any(session, providers, payload)
        if not response_data or not response_data.get('transfers'):
            break
        transfers = response_data.get('transfers', [])
        for transfer in transfers:
            item_count += 1
            tx_hash = transfer.get('hash')
            tx_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionByHash", "params": [tx_hash]}
            tx_info = rpc_post_any(session, providers, tx_payload)
            function_signature = None
            if tx_info and tx_info.get('input'):
                selector = tx_info['input'][:10]
                function_signature = get_function_signature(session, selector)
            receipt_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionReceipt", "params": [tx_hash]}
            receipt = rpc_post_any(session, providers, receipt_payload)
            gas_fee_in_eth = None
            if receipt:
                gas_used = Decimal(int(receipt.get('gasUsed'), 16))
                effective_gas_price = Decimal(int(receipt.get('effectiveGasPrice'), 16))
                gas_fee_in_eth = (gas_used * effective_gas_price) / Decimal(10**18)
            enriched_transfers.append({'block_timestamp': transfer.get('metadata', {}).get('blockTimestamp'), 'hash': tx_hash, 'function_signature': function_signature, 'from_address': transfer.get('from'), 'to_address': transfer.get('to'), 'value': transfer.get('value'), 'asset': transfer.get('asset'), 'gas_fee_eth': gas_fee_in_eth})
        print(f"    ... 전송 기록 {item_count}건 처리 중 ...")
        page_key = response_data.get('pageKey')
        if not page_key:
            break
    if enriched_transfers:
        return enriched_transfers
    # 2nd: eth_getLogs 폴백 (탐색기 없이)
    print("    ↪️ Alchemy Transfers 실패/빈결과 → eth_getLogs 폴백")
    return enriched_transfers_from_logs(session, providers, contract_address, window_blocks=_LOG_WINDOW_BLOCKS, max_events=_LOG_MAX_EVENTS)

def get_all_token_holders(session, providers, contract_address, blockchain):
    print(f"    📥 보유자 목록 수집 중...")
    all_holders = []
    page_key = None
    while True:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getTokenHolders", "params": [contract_address, {"pageKey": page_key if page_key else None}]}
        response_data = rpc_post_any(session, providers, payload)
        if not response_data or not response_data.get('holders'):
            break
        all_holders.extend(response_data.get('holders', []))
        page_key = response_data.get('pageKey')
        if not page_key:
            break
    if all_holders and len(all_holders) > 0:
        print(f"    ... 총 {len(all_holders)}명의 보유자 발견 ...")
        return all_holders
    # 폴백: 최근 로그 기반 재구성
    print("    ↪️ TokenHolders 실패/빈결과 → 로그 기반 보유자 재구성")
    holders = reconstruct_holders_from_logs(session, providers, contract_address, window_blocks=_LOG_WINDOW_BLOCKS, max_events=_HOLDERS_FALLBACK_MAX_EVENTS)
    print(f"    ... 재구성된 보유자 수: {len(holders)}")
    return holders

# ==============================================================================
# SECTION 2-S: Solana 최소 수집 (getSignaturesForAddress / getTransaction 일부)
# ==============================================================================
def solana_get_signatures(session, full_url: str, address: str, limit: int = 20):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress", "params": [address, {"limit": limit}]}
    return http_post(session, full_url, payload) or []

def solana_get_transaction(session, full_url: str, signature: str):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getTransaction", "params": [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]}
    return http_post(session, full_url, payload)

def solana_get_token_supply(session, full_url: str, mint: str):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [mint]}
    res = http_post(session, full_url, payload)
    try:
        if isinstance(res, dict) and res.get('value'):
            v = res['value']
            amount = v.get('amount')
            decimals = v.get('decimals', 0)
            if amount is not None:
                from decimal import Decimal as D
                return (D(amount) / (D(10) ** int(decimals)))
    except Exception:
        pass
    return Decimal(0)

# RPC fallback: top holders via getTokenLargestAccounts
def solana_get_largest_holders(session, full_url: str, mint: str, limit: int = 20):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getTokenLargestAccounts", "params": [mint]}
    res = http_post(session, full_url, payload) or {}
    holders = []
    try:
        vals = res.get('value') or []
        for it in vals[:limit]:
            if not isinstance(it, dict):
                continue
            addr = it.get('address')
            ui = it.get('uiAmount')
            if addr:
                holders.append({"address": addr, "balance": ui})
    except Exception:
        return []
    return holders

# Pick first working Solana RPC from configured providers
def select_solana_rpc(session) -> str | None:
    for url in SOLANA_RPC_PROVIDERS:
        # Lightweight health check: getEpochInfo
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getEpochInfo", "params": []}
        res = http_post(session, url, payload)
        if res is not None:
            return url
    return None

# --- EVM 도우미: 주소가 컨트랙트인지 판별 ---
def is_contract_address(session, providers, address: str) -> bool:
    try:
        if not address:
            return False
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getCode", "params": [address, "latest"]}
        code = rpc_post_any(session, providers if isinstance(providers, list) else [providers], payload)
        return bool(code and isinstance(code, str) and code != "0x")
    except Exception:
        return False

def find_contract_creation_info(session, providers, contract_address):
    print("    🔍 컨트랙트 배포자 및 생성시점 조회 중...")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [{"contractAddresses": [contract_address], "category": ["erc20"], "order": "asc", "maxCount": "0x1"}]}
    response_data = rpc_post_any(session, providers, payload)
    if not response_data or not response_data.get('transfers'): return {"deployer": None, "creation_timestamp": None}
    first_transfer = response_data.get('transfers')[0]
    tx_hash = first_transfer.get('hash')
    tx_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionByHash", "params": [tx_hash]}
    tx_info = rpc_post_any(session, providers, tx_payload)
    return {"deployer": tx_info.get('from') if tx_info else None, "creation_timestamp": first_transfer.get('metadata', {}).get('blockTimestamp')}

def explorer_creation_fallback(session: requests.Session, blockchain: str, address: str, api_key: str) -> dict:
    # Explorer usage removed; rely on on-chain transfer crawl upstream
    return {"deployer": None, "creation_timestamp": None}

def get_contract_role(session, providers, contract_address, function_selector):
    """owner(), minter() 등 간단한 읽기 함수를 호출하여 주소를 반환합니다."""
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": contract_address, "data": function_selector}, "latest"]}
        hex_result = rpc_post_any(session, providers, payload)
        if not hex_result or hex_result == '0x' or len(hex_result) < 42: return None
        return "0x" + hex_result[-40:]
    except Exception: return None

def analyze_proxy_info(session, providers, contract_address):
    print("    🔍 프록시 정보 조회 중...")
    try:
        admin_slot = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
        impl_slot = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
        admin_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getStorageAt", "params": [contract_address, admin_slot, "latest"]}
        admin_hex = rpc_post_any(session, providers, admin_payload)
        proxy_admin = ("0x" + admin_hex[-40:]) if admin_hex and int(admin_hex, 16) != 0 else None
        impl_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getStorageAt", "params": [contract_address, impl_slot, "latest"]}
        impl_hex = rpc_post_any(session, providers, impl_payload)
        implementation = ("0x" + impl_hex[-40:]) if impl_hex and int(impl_hex, 16) != 0 else None
        if proxy_admin or implementation:
            print(f"    ✅ 프록시 정보 발견: Admin({proxy_admin}), Impl({implementation})")
        return {"proxy_admin": proxy_admin, "implementation": implementation}
    except Exception: return {"proxy_admin": None, "implementation": None}

# --- [NEW] 총 발행량(Total Supply) 조회 ---
def get_total_supply(session, providers, contract_address):
    print("    🔍 총 발행량 조회 중...")
    try:
        # 1. decimals() 호출
        decimals_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": contract_address, "data": "0x313ce567"}, "latest"]}
        decimals_hex = rpc_post_any(session, providers, decimals_payload)
        decimals = int(decimals_hex, 16) if decimals_hex and decimals_hex != '0x' else 0

        # 2. totalSupply() 호출
        total_supply_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": contract_address, "data": "0x18160ddd"}, "latest"]}
        total_supply_hex = rpc_post_any(session, providers, total_supply_payload)
        
        if total_supply_hex and total_supply_hex != '0x':
            total_supply_wei = Decimal(int(total_supply_hex, 16))
            total_supply = total_supply_wei / (Decimal(10) ** decimals)
            print(f"    ✅ 총 발행량: {total_supply}")
            return total_supply
        else:
            return Decimal(0)
    except Exception as e:
        print(f"    ❌ 총 발행량 조회 실패: {e}")
        return Decimal(0)

# --- [NEW] 지갑 분석 유틸리티 함수 ---
def get_wallet_first_seen(session, providers, wallet_address):
    """특정 지갑 주소의 첫 활동(첫 송금 또는 첫 수신) 시간을 찾습니다."""
    try:
        # 보낸 첫 거래 조회
        from_payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [{"fromAddress": wallet_address, "maxCount": "0x1", "order": "asc"}]}
        from_tx_data = rpc_post_any(session, providers, from_payload)
        from_ts = from_tx_data['transfers'][0]['metadata']['blockTimestamp'] if from_tx_data and from_tx_data.get('transfers') else None

        # 받은 첫 거래 조회
        to_payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [{"toAddress": wallet_address, "maxCount": "0x1", "order": "asc"}]}
        to_tx_data = rpc_post_any(session, providers, to_payload)
        to_ts = to_tx_data['transfers'][0]['metadata']['blockTimestamp'] if to_tx_data and to_tx_data.get('transfers') else None

        # 두 타임스탬프를 파싱하여 비교
        from_datetime = datetime.fromisoformat(from_ts.replace('Z', '+00:00')) if from_ts else None
        to_datetime = datetime.fromisoformat(to_ts.replace('Z', '+00:00')) if to_ts else None

        if from_datetime and to_datetime:
            return min(from_datetime, to_datetime).isoformat()
        elif from_datetime:
            return from_datetime.isoformat()
        elif to_datetime:
            return to_datetime.isoformat()
        else:
            return None
    except Exception as e:
        # print(f"    - 지갑({wallet_address}) 첫 활동 조회 중 오류: {e}")
        return None

# --- [NEW] 소스코드 가져오기 기능 ---
def _sourcify_fetch_sources(session, blockchain: str, address: str) -> str | None:
    cid = CHAIN_IDS.get(blockchain)
    if not cid:
        return None
    addr = address.lower()
    bases = [
        f"https://repo.sourcify.dev/contracts/full_match/{cid}/{addr}",
        f"https://repo.sourcify.dev/contracts/partial_match/{cid}/{addr}",
    ]
    for base in bases:
        meta_url = f"{base}/metadata.json"
        meta = http_get(session, meta_url)
        if not meta or not isinstance(meta, dict):
            continue
        sources = meta.get("sources") or {}
        if not sources:
            continue
        combined = []
        for rel in list(sources.keys())[:20]:  # cap files
            raw_url = f"{base}/{rel}"
            text = http_get_text(session, raw_url)
            if text:
                combined.append(text)
        if combined:
            return "\n\n".join(combined)
    return None

def get_contract_source_code(session, blockchain, api_key, contract_address):
    # Use Sourcify only (free, explorerless)
    return _sourcify_fetch_sources(session, blockchain, contract_address)

# --- [NEW] 검증 여부 확인 ---
def is_contract_verified(session, blockchain, api_key, contract_address) -> bool:
    # Verification via Sourcify only (free)
    src = _sourcify_fetch_sources(session, blockchain, contract_address)
    return bool(src and str(src).strip())

# --- [NEW] 온체인 로그/블록 헬퍼 (탐색기 없이) ---
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

def _hex(n: int) -> str:
    return hex(n)

def get_latest_block_number(session, providers) -> int:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
    res = rpc_post_any(session, providers, payload)
    try:
        return int(res, 16) if isinstance(res, str) else 0
    except Exception:
        return 0

def get_block_timestamp_iso(session, providers, block_number: int) -> str | None:
    key = str(block_number)
    if key in _BLOCK_TS_CACHE:
        return _BLOCK_TS_CACHE[key]
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [hex(block_number), False]}
    res = rpc_post_any(session, providers, payload) or {}
    try:
        ts = int(res.get("timestamp"), 16)
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        _BLOCK_TS_CACHE[key] = iso
        return iso
    except Exception:
        return None

def get_token_decimals(session, providers, contract_address: str) -> int:
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": contract_address, "data": "0x313ce567"}, "latest"]}
        res = rpc_post_any(session, providers, payload)
        return int(res, 16) if isinstance(res, str) and res not in ("0x", None) else 0
    except Exception:
        return 0

def get_logs_transfers(session, providers, contract_address: str, from_block: int, to_block: int) -> list[dict]:
    logs: list[dict] = []
    step = 2000
    blk = from_block
    while blk <= to_block:
        sub_to = min(to_block, blk + step)
        params = [{
            "fromBlock": _hex(blk),
            "toBlock": _hex(sub_to),
            "address": contract_address,
            "topics": [TRANSFER_TOPIC0],
        }]
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": params}
        res = rpc_post_any(session, providers, payload) or []
        if isinstance(res, list):
            logs.extend(res)
        blk = sub_to + 1
    return logs

def reconstruct_holders_from_logs(session, providers, contract_address: str, window_blocks: int = 50000, max_events: int = 5000) -> list[dict]:
    latest = get_latest_block_number(session, providers)
    if latest <= 0:
        return []
    from_blk = max(0, latest - window_blocks)
    logs = get_logs_transfers(session, providers, contract_address, from_blk, latest)
    if not logs:
        return []
    dec = get_token_decimals(session, providers, contract_address)
    balances: dict[str, Decimal] = {}
    for lg in logs[:max_events]:
        try:
            topics = lg.get("topics") or []
            if len(topics) < 3:
                continue
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]
            data = lg.get("data")
            if not isinstance(data, str):
                continue
            raw = int(data, 16)
            amt = Decimal(raw) / (Decimal(10) ** dec) if dec else Decimal(raw)
            if from_addr.lower() != "0x0000000000000000000000000000000000000000":
                balances[from_addr] = balances.get(from_addr, Decimal(0)) - amt
            balances[to_addr] = balances.get(to_addr, Decimal(0)) + amt
        except Exception:
            continue
    holders = []
    for addr, bal in balances.items():
        if bal <= 0:
            continue
        holders.append({"address": addr, "balance": bal})
    holders.sort(key=lambda x: x["balance"], reverse=True)
    return holders

def enriched_transfers_from_logs(session, providers, contract_address: str, window_blocks: int = 50000, max_events: int = 1000) -> list[dict]:
    latest = get_latest_block_number(session, providers)
    if latest <= 0:
        return []
    from_blk = max(0, latest - window_blocks)
    logs = get_logs_transfers(session, providers, contract_address, from_blk, latest)
    out: list[dict] = []
    seen = 0
    for lg in logs:
        if seen >= max_events:
            break
        try:
            tx_hash = lg.get("transactionHash")
            blk_hex = lg.get("blockNumber")
            blk_num = int(blk_hex, 16) if isinstance(blk_hex, str) else None
            ts_iso = get_block_timestamp_iso(session, providers, blk_num) if blk_num is not None else None
            topics = lg.get("topics") or []
            from_addr = "0x" + topics[1][-40:] if len(topics) > 1 else None
            to_addr = "0x" + topics[2][-40:] if len(topics) > 2 else None
            data = lg.get("data")
            val = int(data, 16) if isinstance(data, str) else None
            out.append({
                'block_timestamp': ts_iso,
                'hash': tx_hash,
                'function_signature': None,
                'from_address': from_addr,
                'to_address': to_addr,
                'value': val,
                'asset': None,
                'gas_fee_eth': None,
            })
            seen += 1
        except Exception:
            continue
    return out

# ==============================================================================
# SECTION 2.5: 지표 계산 유틸 (NHHI 등)
# ==============================================================================
def _to_decimal(val) -> Decimal:
    try:
        if val is None:
            return Decimal(0)
        # hex string
        if isinstance(val, str) and val.startswith("0x"):
            return Decimal(int(val, 16))
        # numeric string or number
        return Decimal(str(val))
    except Exception:
        return Decimal(0)

def _extract_balance(row: dict) -> Decimal:
    """Best‑effort balance extractor across different holder schemas."""
    for key in ("balance", "tokenBalance", "token_balance", "rawBalance", "tokenBalanceRaw"):
        if key in row and row[key] is not None:
            return _to_decimal(row[key])
    return Decimal(0)

def compute_nhhi(holders_df, circulating: Decimal, exclude_labels: bool = True) -> float:
    """Normalized HHI in [0,1]. Uses balances over circulating.

    - Excludes known non‑circulating labels if available (burn/exchange/locker) when exclude_labels=True
    - Falls back gracefully when columns are missing
    """
    try:
        if holders_df is None or holders_df.empty:
            return 0.0

        df = holders_df.copy()
        # Optional exclusions by label
        if exclude_labels and "label" in df.columns:
            mask = ~df["label"].fillna("").str.contains(
                "Burn Address|Hot Wallet|Bridge|Router|Treasury|Locker",
                case=False,
                regex=True,
            )
            df = df[mask]

        if len(df) == 0 or circulating <= 0:
            return 0.0

        # Ensure a 'balance' column as Decimal
        if "balance" not in df.columns:
            df["balance"] = df.apply(lambda r: _extract_balance(r.to_dict()), axis=1)
        else:
            df["balance"] = df["balance"].apply(_to_decimal)

        # Only positive balances
        df = df[df["balance"] > 0]
        if len(df) == 0:
            return 0.0

        shares = (df["balance"] / circulating).clip(lower=Decimal(0))
        # Convert to float for HHI (sum of squares)
        shares_float = shares.astype(float)
        hhi = float((shares_float ** 2).sum())
        n = len(shares_float)
        # Normalize to [0,1]
        min_hhi = 1.0 / n
        nhhi = 0.0 if n <= 1 else (hhi - min_hhi) / (1.0 - min_hhi)
        return max(0.0, min(1.0, nhhi))
    except Exception:
        return 0.0

def compute_top_holder_share(holders_df, circulating: Decimal) -> float:
    try:
        if holders_df is None or holders_df.empty or circulating <= 0:
            return 0.0
        df = holders_df.copy()
        if "balance" not in df.columns:
            df["balance"] = df.apply(lambda r: _extract_balance(r.to_dict()), axis=1)
        else:
            df["balance"] = df["balance"].apply(_to_decimal)
        df = df[df["balance"] > 0]
        if len(df) == 0:
            return 0.0
        top = df["balance"].max()
        return float((top / circulating))
    except Exception:
        return 0.0

def compute_fresh_wallet_fraction(session, providers, top_addresses: list[str], creation_ts_iso: str, days: int = 7) -> float:
    """Among given addresses, fraction whose first activity is within N days of token creation.
    EVM‑only; returns 0.0 if dates unavailable.
    """
    try:
        if not creation_ts_iso or not top_addresses:
            return 0.0
        try:
            created = datetime.fromisoformat(creation_ts_iso.replace("Z", "+00:00"))
        except Exception:
            return 0.0
        fresh = 0
        checked = 0
        providers_list = providers if isinstance(providers, list) else [providers]
        for addr in top_addresses:
            ts = get_wallet_first_seen(session, providers_list, addr)
            if not ts:
                continue
            try:
                first_seen = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            checked += 1
            if (first_seen - created).days <= days:
                fresh += 1
        if checked == 0:
            return 0.0
        return fresh / checked
    except Exception:
        return 0.0

# Subscores (0..100)
def compute_concentration_subscore(holders_df, circulating: Decimal) -> int:
    nhhi = compute_nhhi(holders_df, circulating)
    return int(round(max(0.0, min(1.0, nhhi)) * 100))

def _build_balance_map(holders_df) -> dict:
    bm = {}
    if holders_df is None or holders_df.empty:
        return bm
    df = holders_df.copy()
    if 'address' not in df.columns:
        return bm
    if 'balance' not in df.columns:
        df['balance'] = df.apply(lambda r: _extract_balance(r.to_dict()), axis=1)
    else:
        df['balance'] = df['balance'].apply(_to_decimal)
    for _, row in df.iterrows():
        addr = str(row.get('address') or '').lower()
        if not addr:
            continue
        bal = row.get('balance')
        try:
            bal_d = _to_decimal(bal)
        except Exception:
            bal_d = Decimal(0)
        bm[addr] = bm.get(addr, Decimal(0)) + bal_d
    return bm

def compute_developer_subscore(holders_df, contract_info: dict, session, providers, circulating: Decimal, is_evm: bool) -> int:
    """개발자 리스크 서브스코어(0..100)를 계산합니다.

    구성 요소
    - 개발자 보유비중(0..100): 배포자/소유자/민터/프록시관리자 주소가 보유한 유통량 대비 비중
    - 권한 리스크 점수(가산): EOA가 민감 권한을 보유, 업그레이드 가능성 등
    - 완화 요인(감산): 소스 검증, 소유권 포기(renounce) 등
    """
    try:
        if circulating <= 0:
            return 0

        # 1) 개발자 관련 주소 수집
        dev_addrs = set()
        for key in ('deployer', 'owner', 'minter', 'proxy_admin'):
            v = contract_info.get(key)
            if isinstance(v, str) and v:
                dev_addrs.add(v.lower())

        # 2) 개발자 보유 비중 계산 (유통량 대비 %)
        bal_map = _build_balance_map(holders_df)
        dev_bal = sum((bal_map.get(a, Decimal(0)) for a in dev_addrs), Decimal(0))
        dev_holdings_ratio = float((dev_bal / circulating) * 100) if circulating > 0 else 0.0
        dev_holdings_ratio = max(0.0, min(100.0, dev_holdings_ratio))

        # 3) 권한 리스크 및 완화 요인 산정
        risk_points = 0

        def _is_zero(addr: str | None) -> bool:
            if not addr or not isinstance(addr, str):
                return False
            return addr.lower() == "0x0000000000000000000000000000000000000000"

        if is_evm:
            owner = contract_info.get('owner')
            minter = contract_info.get('minter')
            proxy_admin = contract_info.get('proxy_admin')

            # (a) 민감 권한을 EOA가 보유하면 가산점(리스크 증가)
            if minter and not is_contract_address(session, providers, minter):
                risk_points += 25
            if owner and not is_contract_address(session, providers, owner):
                risk_points += 20
            if proxy_admin and not is_contract_address(session, providers, proxy_admin):
                risk_points += 20

            # (b) 업그레이드 경로 존재 시 가산
            if contract_info.get('implementation'):
                risk_points += 10

            # (c) 배포자와 권한 주소가 동일한 EOA인 경우 소폭 가산
            deployer = contract_info.get('deployer')
            if isinstance(deployer, str) and deployer:
                dl = deployer.lower()
                for role_addr in (owner, minter, proxy_admin):
                    if isinstance(role_addr, str) and role_addr and role_addr.lower() == dl and not is_contract_address(session, providers, role_addr):
                        risk_points += 5
                        break

            # (d) 완화: 소유권 포기(renounce) 또는 권한이 컨트랙트(멀티시그/타임락 가능성)
            if _is_zero(owner):
                risk_points -= 15
            else:
                # 컨트랙트가 소유하면 소폭 감산
                if owner and is_contract_address(session, providers, owner):
                    risk_points -= 5
            if minter and is_contract_address(session, providers, minter):
                risk_points -= 3
            if proxy_admin and is_contract_address(session, providers, proxy_admin):
                risk_points -= 3

        # (e) 소스 검증 여부는 체인 공통 완화 요인으로 반영
        if contract_info.get('is_verified'):
            risk_points -= 10

        # 정규화
        risk_points = max(0, min(100, risk_points))

        # 4) 최종 스코어 결합 (기존 가중치 유지)
        dev_score = 0.6 * dev_holdings_ratio + 0.4 * risk_points
        return int(round(max(0.0, min(100.0, dev_score))))
    except Exception:
        return 0

def _parse_iso(ts: str):
    try:
        return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
    except Exception:
        return None

def _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso: str, circulating: Decimal) -> int:
    try:
        if transfers_df is None or transfers_df.empty or circulating <= 0 or not creation_ts_iso:
            return 0
        created = _parse_iso(creation_ts_iso)
        if not created:
            return 0
        window_sec = 600
        early_receivers = set()
        for _, row in transfers_df.iterrows():
            ts = _parse_iso(row.get('block_timestamp')) if row.get('block_timestamp') else None
            if not ts:
                continue
            if 0 <= (ts - created).total_seconds() <= window_sec:
                to_addr = str(row.get('to_address') or '').lower()
                if to_addr:
                    early_receivers.add(to_addr)
        if not early_receivers:
            return 0
        bal_map = _build_balance_map(holders_df)
        sniper_bal = sum((bal_map.get(a, Decimal(0)) for a in early_receivers), Decimal(0))
        ratio = float((sniper_bal / circulating) * 100)
        return int(round(max(0.0, min(100.0, ratio))))
    except Exception:
        return 0


def compute_sniper_subscore(transfers_df, holders_df, creation_ts_iso: str, circulating: Decimal, contract_info: dict | None = None) -> int:
    """Compute sniper subscore using trained ML if artifacts/data available, else fallback.

    ML path:
    - Load early_trades dataset filtered by token_address
    - Build features with FeatureBuilder (consistent scaling)
    - Score with SniperScorer; subscore = 100 * weighted (by usd_value) average score
    """
    # Heuristic fallback if ML not available
    if not _SNIPER_AVAILABLE or contract_info is None:
        return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)
    try:
        token_addr = str(contract_info.get('address') or '').lower()
        if not token_addr:
            return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)
        # Load config and dataset
        cfg_path = os.path.expanduser("~/Desktop/Hyperindex/snipers/config.yaml")
        cfg = _SniperConfig.from_yaml(cfg_path)
        ds_path = cfg.dataset_dir / "early_trades.parquet"
        if not (ds_path.exists() or ds_path.with_suffix('.csv').exists()):
            return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)
        import pandas as _pd  # local import
        # Load and filter
        try:
            df = _pd.read_parquet(ds_path)
        except Exception:
            df = _pd.read_csv(ds_path.with_suffix('.csv'))
        if 'token_address' not in df.columns:
            return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)
        mask = df['token_address'].astype(str).str.lower() == token_addr
        df_tok = df[mask].copy()
        if df_tok.empty:
            return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)
        # Build features and score
        feats, _spec = _FeatureBuilder(cfg).build_features(df_tok)
        scorer = _SniperScorer(cfg.artifacts_dir)
        out = scorer.score_df(feats)
        if 'score' not in out.columns:
            return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)
        scores = out['score'].astype(float)
        if 'usd_value' in out.columns:
            w = out['usd_value'].astype(float).clip(lower=0.0)
            if w.sum() > 0:
                s = float((scores * w).sum() / w.sum())
            else:
                s = float(scores.mean())
        else:
            s = float(scores.mean())
        return int(round(max(0.0, min(1.0, s)) * 100))
    except Exception:
        return _compute_sniper_subscore_heuristic(transfers_df, holders_df, creation_ts_iso, circulating)

def compute_insider_subscore(transfers_df, holders_df, contract_info: dict, circulating: Decimal) -> int:
    try:
        if transfers_df is None or transfers_df.empty or circulating <= 0:
            return 0
        # Define insider set
        insiders = set()
        for key in ('deployer', 'owner', 'minter', 'proxy_admin'):
            v = contract_info.get(key)
            if isinstance(v, str) and v:
                insiders.add(v.lower())
        if not insiders:
            return 0
        # Determine recent window baseline: last 7 days from max timestamp seen
        ts_vals = [_parse_iso(x) for x in transfers_df.get('block_timestamp', []) if x]
        ts_vals = [t for t in ts_vals if t]
        if not ts_vals:
            return 0
        end_ts = max(ts_vals)
        start_ts = end_ts - pd.Timedelta(days=7)
        outflow = Decimal(0)
        for _, row in transfers_df.iterrows():
            ts = _parse_iso(row.get('block_timestamp')) if row.get('block_timestamp') else None
            if not ts or ts < start_ts or ts > end_ts:
                continue
            from_addr = str(row.get('from_address') or '').lower()
            if from_addr in insiders:
                outflow += _to_decimal(row.get('value'))
        ratio = float((outflow / circulating) * 100)
        return int(round(max(0.0, min(100.0, ratio))))
    except Exception:
        return 0

# ==============================================================================
# SECTION 3.5: 엔티티 클러스터링 (탐색기 없이 동작하는 1차 휴리스틱)
# ==============================================================================
class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}
        self.sz: dict[str, int] = {}
    def find(self, x: str) -> str:
        if x not in self.p:
            self.p[x] = x
            self.sz[x] = 1
            return x
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] = self.sz.get(ra, 1) + self.sz.get(rb, 1)

def _is_addr(x) -> bool:
    return isinstance(x, str) and x.startswith("0x") and len(x) == 42

def _guess_pair_address(transfers_df) -> str | None:
    if transfers_df is None or transfers_df.empty:
        return None
    counts = defaultdict(int)
    for col in ("from_address", "to_address"):
        if col in transfers_df.columns:
            for v in transfers_df[col].astype(str).tolist():
                if _is_addr(v):
                    counts[v.lower()] += 1
    if not counts:
        return None
    # The most frequent counterparty often corresponds to the pair (buys: from=pair, sells: to=pair)
    return max(counts.items(), key=lambda kv: kv[1])[0]

def cluster_entities(transfers_df, *, window_sec: int = 10, min_group: int = 2, max_events: int = 1000):
    """Build clusters via co-trade and sink heuristics based on token transfers only.
    - co-trade: recipients of transfers from the guessed pair within same block or <=window_sec grouped
    - sink: multiple distinct senders repeatedly sending to the same sink (not the pair)
    """
    try:
        if transfers_df is None or transfers_df.empty:
            return []
        df = transfers_df.copy()
        # normalize ts
        if "block_timestamp" in df.columns:
            df["_ts"] = df["block_timestamp"].apply(lambda x: _parse_iso(x) if x else None)
        else:
            df["_ts"] = None
        df = df.sort_values([c for c in ["_ts"] if c in df.columns])
        pair = _guess_pair_address(df)
        uf = _UF()

        # co-trade edges (buys within same block or small window, from=pair)
        if pair:
            batch = []
            prev_ts = None
            for _, r in df.iterrows():
                if r.get("from_address") and str(r.get("from_address")).lower() == pair:
                    ts = r.get("_ts")
                    if prev_ts is None:
                        prev_ts = ts
                    # flush batch if ts gap exceeds window
                    if prev_ts and ts and (ts - prev_ts).total_seconds() > window_sec:
                        # connect all recipients in batch
                        recips = [str(x).lower() for x in batch if _is_addr(x)]
                        if len(recips) >= min_group:
                            root = recips[0]
                            for addr in recips[1:]:
                                uf.union(root, addr)
                        batch = []
                    prev_ts = ts
                    batch.append(r.get("to_address"))
            # flush remainder
            recips = [str(x).lower() for x in batch if _is_addr(x)]
            if len(recips) >= min_group:
                root = recips[0]
                for addr in recips[1:]:
                    uf.union(root, addr)

        # sink edges (multiple senders to same sink, excluding pair addr)
        sink_counts = defaultdict(int)
        for _, r in df.head(max_events).iterrows():
            to_addr = str(r.get("to_address") or "").lower()
            if _is_addr(to_addr) and (not pair or to_addr != pair):
                sink_counts[to_addr] += 1
        frequent_sinks = {s for s, c in sink_counts.items() if c >= max(3, min_group)}
        for sink in frequent_sinks:
            senders = set()
            for _, r in df.head(max_events).iterrows():
                if str(r.get("to_address") or "").lower() == sink:
                    frm = str(r.get("from_address") or "").lower()
                    if _is_addr(frm):
                        senders.add(frm)
            if len(senders) >= min_group:
                senders = list(senders)
                root = senders[0]
                for addr in senders[1:]:
                    uf.union(root, addr)

        # materialize clusters
        comps: dict[str, set[str]] = defaultdict(set)
        for addr in set([str(x).lower() for x in df.get("from_address", []) if _is_addr(x)] + [str(x).lower() for x in df.get("to_address", []) if _is_addr(x)]):
            rep = uf.find(addr)
            comps[rep].add(addr)
        clusters = [sorted(list(members)) for members in comps.values() if len(members) >= min_group]
        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters
    except Exception:
        return []

# ==============================================================================
# SECTION 3: 스캠 분석 로직 (사용자가 직접 채워야 할 부분)
# ==============================================================================
def analyze_scam_score(transfers_df, holders_df, contract_info, session, providers_or_url, token_blockchain):
    print("    🔬 스캠 위험 스코어 산정(0~100, 높을수록 위험)...")

    # --- Free Float 계산 ---
    total_supply = contract_info.get('total_supply', Decimal(0))
    burned_amount = Decimal(0)
    if 'to_label' in transfers_df.columns:
        burn_txs = transfers_df[(transfers_df['to_label'] == 'Burn Address')]
        if not burn_txs.empty:
            burned_amount = burn_txs['value'].apply(lambda x: Decimal(str(x)) if x is not None else Decimal(0)).sum()

    locked_amount_sum = Decimal(0)
    if 'is_known_locker' in holders_df.columns and 'locked_amount' in holders_df.columns:
        locked_holders = holders_df[(holders_df['is_known_locker'] == True)]
        if not locked_holders.empty:
            locked_amount_sum = locked_holders['locked_amount'].apply(lambda x: Decimal(str(x)) if x is not None else Decimal(0)).sum()

    cex_custody_amount = Decimal(0)
    if 'label' in holders_df.columns:
        cex_holders = holders_df[(holders_df['label'].str.contains('Hot Wallet', na=False))]
        if not cex_holders.empty:
            cex_custody_amount = cex_holders['balance'].apply(lambda x: Decimal(str(x)) if x is not None else Decimal(0)).sum()

    bridge_router_amount = Decimal(0)
    staking_pool_amount = Decimal(0)
    treasury_amount = Decimal(0)

    total_excluded_amount = burned_amount + locked_amount_sum + cex_custody_amount + \
                            bridge_router_amount + staking_pool_amount + treasury_amount

    free_float = total_supply - total_excluded_amount
    print(f"    🌊 Free Float: {free_float}")

    # [NEW] 라벨링된 주소 정보를 활용하는 예시:
    # -------------------------------------------------------------
    # 예: 전송 기록에 소각 주소(Burn Address)가 포함되어 있는지 확인
    if 'from_label' in transfers_df.columns and 'to_label' in transfers_df.columns:
        burn_tx_count = transfers_df[(transfers_df['to_label'] == 'Burn Address')].shape[0]
        if burn_tx_count > 0:
            print(f"    🔥 소각 트랜잭션 {burn_tx_count}건 발견.")
            # 소각 트랜잭션이 많으면 긍정적인 신호일 수 있으므로 점수 조정
            # score += 5 # 예시

    # 예: 홀더 목록에 거래소 주소가 포함되어 있는지 확인
    if 'label' in holders_df.columns:
        exchange_holders = holders_df[(holders_df['label'].str.contains('Hot Wallet', na=False))].shape[0]
        if exchange_holders > 0:
            print(f"    🏦 거래소 홀더 {exchange_holders}명 발견.")
            # 거래소 홀더가 많으면 집중도 분석 시 제외하거나 특별 처리
            # score -= 5 # 예시
    # -------------------------------------------------------------

    # --- Subscores (0..100 each) ---
    circulating = free_float if free_float > 0 else total_supply
    is_evm = token_blockchain != 'solana'

    sub_concentration = compute_concentration_subscore(holders_df, circulating) if circulating > 0 else 0
    # Dev risk: prefer external engine if available; otherwise local heuristic
    sub_developer = 0
    chain_code = {"solana": "sol", "ethereum": "eth", "bnb": "bnb"}.get(token_blockchain, "eth")
    if dr_compute_developer_subscore is not None:
        try:
            dr = dr_compute_developer_subscore(contract_info.get('address') or token_address, chain_code)
            sub_developer = int(round(float(dr.get('developer_subscore', 0))))
            print(f"    🧑‍💻 Dev: hold={dr.get('dev_holdings_pct', 0):.1f}% risk={dr.get('dev_risk_score', 0)} sub={sub_developer}")
        except Exception:
            sub_developer = 0
    else:
        try:
            sub_developer = compute_developer_subscore(holders_df, contract_info, session, providers_or_url, circulating, is_evm)
        except Exception:
            sub_developer = 0
    sub_sniper = compute_sniper_subscore(transfers_df, holders_df, contract_info.get('creation_timestamp'), circulating, contract_info) if is_evm else 0
    sub_insider = compute_insider_subscore(transfers_df, holders_df, contract_info, circulating) if is_evm else 0

    print(f"    📊 Subscores → Concentration:{sub_concentration} | Developer:{sub_developer} | Sniper:{sub_sniper} | Insider:{sub_insider}")

    # Weights normalization (sum to 1)
    w = RISK_WEIGHTS.copy()
    s = sum(w.values()) or 1.0
    for k in w:
        w[k] = w[k] / s
    final_score = (
        sub_concentration * w.get('concentration', 0)
        + sub_developer * w.get('developer', 0)
        + sub_sniper * w.get('sniper', 0)
        + sub_insider * w.get('insider', 0)
    )
    final_score = float(max(0.0, min(100.0, final_score)))
    print(f"    ✅ 최종 위험 점수: {final_score:.2f} (임계 {SCAM_THRESHOLD})")
    verdict = "SCAM" if final_score >= SCAM_THRESHOLD else "OK"
    print(f"    🔎 판정: {verdict}")
    return final_score

    # [NEW] 지갑 생성일 분석 함수 사용 예시:
    # -------------------------------------------------------------
    # 예: 상위 홀더 중 한 명이 의심스러워 생성일을 조회하고 싶을 때
    if not holders_df.empty:
        suspicious_holder = holders_df.iloc[0]['address'] # 첫번째 홀더를 예시로
        print(f"    🔍 의심 지갑({suspicious_holder})의 첫 활동일 조회...")
        first_seen_timestamp = None
        if token_blockchain != 'solana':
            # EVM 전용: providers 리스트를 사용한 페일오버 호출
            providers = providers_or_url if isinstance(providers_or_url, list) else [providers_or_url]
            first_seen_timestamp = get_wallet_first_seen(session, providers, suspicious_holder)
        if first_seen_timestamp:
            print(f"    ✅ 의심 지갑의 첫 활동일: {first_seen_timestamp}")
            # 이제 이 시간을 바탕으로 '새 지갑' 여부 등을 판단하는 로직을 추가할 수 있습니다。
        else:
            print("    ⚠️ 의심 지갑의 활동일을 찾지 못했습니다.")
    # -------------------------------------------------------------

    # [NEW] 소스코드 가져오기 함수 사용 예시:
    # -------------------------------------------------------------
    # 예: 컨트랙트의 소스코드를 가져와서 특정 키워드(예: 'mintable')가 있는지 확인
    contract_address = contract_info.get('address')
    token_blockchain = contract_info.get('blockchain')
    if contract_address and token_blockchain:
        print(f"    🔍 컨트랙트({contract_address}) 소스코드 분석 중...")
        # Explorerless: skip source fetching
        source_code = None
        if source_code:
            if "mintable" in source_code.lower():
                print("    ⚠️ 소스코드에서 'mintable' 키워드 발견! 추가 발행 가능성.")
                score -= 20 # 예시 감점
            else:
                print("    ✅ 소스코드에서 'mintable' 키워드 없음.")
        else:
            print("    ⚠️ 소스코드를 가져올 수 없어 분석 건너뜁니다.")
    # -------------------------------------------------------------

    # dead code path removed above

# ==============================================================================
# SECTION 4: 메인 파이프라인 실행
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperIndex token risk pipeline")
    parser.add_argument("--chains", type=str, default="ethereum,bnb", help="Comma-separated chains to analyze (e.g., ethereum,bnb)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum number of tokens to analyze")
    parser.add_argument("--min-interval", type=float, default=None, help="Minimum seconds between HTTP requests (rate limit)")
    parser.add_argument("--keyword", type=str, default=None, help="Override search keyword for Dune query")
    parser.add_argument("--dex-fallback-only", action="store_true", help="Use DexScreener when Dune fails or skip Dune")
    parser.add_argument("--dex-csv-out", type=str, default=None, help="Save DexScreener tokens CSV to this path")
    parser.add_argument("--log-window-blocks", type=int, default=None, help="Fallback eth_getLogs window size (blocks)")
    parser.add_argument("--log-max-events", type=int, default=None, help="Fallback max events to parse from logs")
    parser.add_argument("--token-addresses", type=str, default=None, help="Comma-separated addresses, optionally prefixed with chain (e.g., ethereum:0x...,bnb:0x...). Skips Dune.")
    parser.add_argument("--cluster", action="store_true", help="Run entity clustering heuristics and print summary")
    parser.add_argument("--cluster-out", type=str, default=None, help="Write clusters as JSON to this path")
    parser.add_argument("--cluster-min-group", type=int, default=2, help="Minimum addresses to form a cluster")
    parser.add_argument("--cluster-window-sec", type=int, default=10, help="Seconds window for co-trade grouping")
    parser.add_argument("--cluster-top-k", type=int, default=3, help="Print top-K clusters in detail")
    args = parser.parse_args()

    # Setup overrides if provided
    if args.min_interval is not None:
        try:
            _globals = globals()
            _globals["_MIN_REQUEST_INTERVAL_SEC"] = float(args.min_interval)
        except Exception:
            pass
    if args.log_window_blocks is not None:
        try:
            globals()["_LOG_WINDOW_BLOCKS"] = int(args.log_window_blocks)
        except Exception:
            pass
    if args.log_max_events is not None:
        try:
            globals()["_LOG_MAX_EVENTS"] = int(args.log_max_events)
            globals()["_HOLDERS_FALLBACK_MAX_EVENTS"] = int(max(args.log_max_events, 400))
        except Exception:
            pass

    selected_chains = set([c.strip().lower() for c in args.chains.split(",") if c.strip()])

    start_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    search_keyword = args.keyword if args.keyword is not None else SEARCH_KEYWORD
    print(f"🚀 자동화 파이프라인을 시작합니다. (v16) | 시작: {start_ts} | 체인: {','.join(sorted(selected_chains))} | 키워드: {search_keyword}")

    if not args.dex_fallback_only and DUNE_API_KEY_ENV not in os.environ:
        exit(f"🛑 오류: Dune API 키 환경 변수({DUNE_API_KEY_ENV})가 설정되지 않았습니다.")
    
    if args.token_addresses:
        print("[단계 1] Dune 건너뜀: 직접 입력된 주소 사용")
        token_list = build_token_list_from_args(args.token_addresses)
    else:
        http_session = requests.Session()
        token_list = None
        if not args.dex_fallback_only and SPICE_AVAILABLE:
            token_list = find_tokens_by_keyword(DUNE_SQL_QUERY_TEMPLATE, search_keyword)
        if token_list is None or (hasattr(token_list, "empty") and token_list.empty) or (hasattr(token_list, "__len__") and len(token_list) == 0):
            print("[단계 1] Dune 실패/빈 결과 → DexScreener로 대체")
            raw = ds.search_pairs_cached(search_keyword, http_session)
            token_list = ds.tokens_from_pairs_json(raw)
            try:
                token_list = ds.enrich_tokens(token_list, http_session)
            except Exception:
                pass
            if args.dex_csv_out:
                token_list[["blockchain","address","name","symbol"]].to_csv(args.dex_csv_out, index=False)
    # Chain filter and max-tokens cap
    if token_list is not None and not token_list.empty:
        token_list['blockchain'] = token_list['blockchain'].str.lower()
        token_list = token_list[token_list['blockchain'].isin(selected_chains)]
        if args.max_tokens:
            token_list = token_list.head(args.max_tokens)

    if token_list is not None and len(token_list) > 0:
        print("\n[단계 2] 개별 토큰 분석 및 스코어링 시작...")
        final_results = []
        http_session = requests.Session()
        # Session retry for transient errors (mainly GETs)
        try:
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            http_session.mount('http://', adapter)
            http_session.mount('https://', adapter)
        except Exception:
            pass

        for index, token in token_list.iterrows():
            token_address = token['address']
            token_blockchain = token['blockchain'].lower()
            print(f"\n[{index + 1}/{len(token_list)}] 토큰 처리 시작: {token_address} on {token_blockchain}")

            # Solana 분기 (최소 수집)
            if token_blockchain == "solana":
                if not SOLANA_RPC_PROVIDERS:
                    print("    ⚠️ solana에 대한 설정이 없어 건너뜁니다.")
                    continue
                sol_full = select_solana_rpc(http_session)
                if not sol_full:
                    print("    ⚠️ 사용 가능한 Solana RPC 엔드포인트를 찾지 못했습니다.")
                    continue
                sigs = solana_get_signatures(http_session, sol_full, token_address, limit=20)
                transfers = []
                if isinstance(sigs, list):
                    for s in sigs[:3]:
                        sig = s.get('signature') if isinstance(s, dict) else None
                        if not sig:
                            continue
                        _ = solana_get_transaction(http_session, sol_full, sig)
                        transfers.append({
                            'block_timestamp': None,
                            'hash': sig,
                            'function_signature': None,
                            'from_address': None,
                            'to_address': None,
                            'value': None,
                            'asset': None,
                            'gas_fee_eth': None,
                        })
                transfers_df = pd.DataFrame(transfers)
                # Solscan 제거: Helius REST → RPC 순으로 홀더 수집
                holders = helius_get_token_holders(token_address, session=http_session)
                if not holders:
                    # RPC 폴백: 상위 홀더 확보
                    holders = solana_get_largest_holders(http_session, sol_full, token_address, limit=20)
                holders_df = pd.DataFrame(holders)
                try:
                    apply_labels(holders_df, transfers_df, token_blockchain)
                except Exception:
                    pass
                print(f"    📊 Solana 홀더 수집: {len(holders_df)}명")
                total_supply = solana_get_token_supply(http_session, sol_full, token_address)
                contract_info = {"address": token_address, "blockchain": token_blockchain, "total_supply": total_supply}
                scam_score = analyze_scam_score(transfers_df, holders_df, contract_info, http_session, sol_full, token_blockchain)
                result = {
                    'address': token_address, 'blockchain': token_blockchain, 'name': token.get('name'),
                    'symbol': token.get('symbol'), 'scam_score': scam_score, 'deployer': None,
                    'creation_timestamp': None, 'owner': None,
                    'minter': None, 'proxy_admin': None, 'implementation': None,
                    'is_verified': False,
                    'verdict': ('SCAM' if scam_score >= SCAM_THRESHOLD else 'OK')
                }
                final_results.append(result)
                print(f"    🗑️ Raw Data 메모리 정리 완료.")
                continue

            providers = EVM_RPC_PROVIDERS.get(token_blockchain) or []
            if not providers:
                print(f"    ⚠️ {token_blockchain}에 대한 RPC 설정이 없어 건너뜁니다.")
                continue

            # Use first provider for alchemy-enhanced calls in downstream functions
            full_url = providers[0]

            creation_info = find_contract_creation_info(http_session, providers, token_address)
            owner = get_contract_role(http_session, providers, token_address, "0x8da5cb5b") # owner()
            minter = get_contract_role(http_session, providers, token_address, "0x07540978") # minter()
            proxy_info = analyze_proxy_info(http_session, providers, token_address)
            total_supply = get_total_supply(http_session, providers, token_address) # [NEW] 총 발행량 수집
            transfers = get_enriched_transfers(http_session, providers, token_address, token_blockchain)
            holders = get_all_token_holders(http_session, providers, token_address, token_blockchain)
            
            transfers_df = pd.DataFrame(transfers)
            holders_df = pd.DataFrame(holders)
            try:
                apply_labels(holders_df, transfers_df, token_blockchain)
            except Exception:
                pass
            
            contract_info = {
                **creation_info,
                "owner": owner,
                "minter": minter,
                **proxy_info,
                "total_supply": total_supply,
                "address": token_address,
                "blockchain": token_blockchain,
            }
            # Use Sourcify for verification (free)
            verified = is_contract_verified(http_session, token_blockchain, None, token_address)
            contract_info["is_verified"] = verified
            scam_score = analyze_scam_score(transfers_df, holders_df, contract_info, http_session, providers, token_blockchain)

            result = {
                'address': token_address, 'blockchain': token_blockchain, 'name': token.get('name'),
                'symbol': token.get('symbol'), 'scam_score': scam_score, 'deployer': creation_info.get('deployer'),
                'creation_timestamp': creation_info.get('creation_timestamp'), 'owner': owner, 
                'minter': minter, 'proxy_admin': proxy_info.get('proxy_admin'), 'implementation': proxy_info.get('implementation'),
                'is_verified': False,
                'verdict': ('SCAM' if scam_score >= SCAM_THRESHOLD else 'OK')
            }
            final_results.append(result)

            # Optional: Entity clustering (EVM only)
            if args.cluster:
                try:
                    clusters = cluster_entities(
                        transfers_df,
                        window_sec=int(getattr(args, 'cluster_window_sec', 10) or 10),
                        min_group=int(getattr(args, 'cluster_min_group', 2) or 2),
                        max_events=_LOG_MAX_EVENTS,
                    )
                    if clusters:
                        print(f"    🧩 클러스터 수: {len(clusters)} | 상위 클러스터 크기: {len(clusters[0])}")
                        topk = int(getattr(args, 'cluster_top_k', 3) or 3)
                        for i, c in enumerate(clusters[:topk]):
                            print(f"    ▶ C{i} (size={len(c)}): {', '.join(c[:10])}{', ...' if len(c)>10 else ''}")
                        out = getattr(args, 'cluster_out', None)
                        if out:
                            try:
                                with open(out, 'w', encoding='utf-8') as f:
                                    import json as _json
                                    _json.dump({
                                        'token': token_address,
                                        'chain': token_blockchain,
                                        'params': {
                                            'window_sec': int(getattr(args, 'cluster_window_sec', 10) or 10),
                                            'min_group': int(getattr(args, 'cluster_min_group', 2) or 2),
                                            'max_events': _LOG_MAX_EVENTS,
                                        },
                                        'clusters': clusters,
                                    }, f, ensure_ascii=False, indent=2)
                                print(f"    💾 클러스터 JSON 저장: {out}")
                            except Exception as e:
                                print(f"    ⚠️ 클러스터 저장 실패: {e}")
                    else:
                        print("    🧩 클러스터 없음(규칙 미충족).")
                except Exception as e:
                    print(f"    🧩 클러스터링 중 오류: {e}")

            del transfers, holders, transfers_df, holders_df, creation_info, owner, minter, proxy_info, total_supply
            print(f"    🗑️ Raw Data 메모리 정리 완료.")

        print("\n\n[🎉 파이프라인 완료] 최종 분석 결과:")
        final_df = pd.DataFrame(final_results)
        print(final_df)
        out_csv = os.getenv("EXPORT_RESULTS_CSV")
        if out_csv:
            try:
                final_df.to_csv(out_csv, index=False)
                print(f"[CSV] Saved final results to: {out_csv}")
            except Exception as e:
                print(f"[CSV] Save failed: {e}")

        # 메트릭 요약 출력
        print("\n[메트릭 요약]")
        print(f"- Dune 쿼리 수: {METRICS['dune']['queries']}")
    # Explorer calls removed (Etherscan/BscScan/Solscan)
        print(f"- 4byte 호출 수: {METRICS['fourbyte']['calls']}")
        alchemy_calls = METRICS['alchemy']['calls']
        print(f"- Alchemy 호출 수: {alchemy_calls}")
        if alchemy_calls:
            by_method = METRICS['alchemy']['by_method']
            print("  * 메서드별 호출: " + ", ".join(f"{m}={c}" for m, c in by_method.items()))
            print(f"  * 추정 CU: {METRICS['alchemy']['estimated_cu']} (정책 확인 후 보정 필요)")
        if METRICS['errors']['count']:
            print(f"- 네트워크 오류 수: {METRICS['errors']['count']}")

        end_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        print(f"[타임라인] 시작: {start_ts} | 종료: {end_ts}")

    else:
        print("\n분석할 토큰을 찾지 못했습니다. 파이프라인을 종료합니다.")
