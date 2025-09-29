"""
환경 변수 기반 설정 모듈
- 민감정보(API 키)는 코드에 하드코딩하지 않고 환경 변수에서 로드합니다.
- 로컬 개발에서는 .env 파일을 사용하세요 (.env.example 참고).
"""

import os

# 선택: .env 자동 로드 (설치된 경우에만)
try:
    from dotenv import load_dotenv
    # Ensure .env values take precedence over any stale shell exports
    load_dotenv(override=True)
except Exception:
    pass

# 1) Alchemy 설정 (체인별 API 키)
# 각 체인에 대해 환경변수로 세팅하세요.
# ALCHEMY_API_KEY_ETHEREUM, ALCHEMY_API_KEY_BNB, ALCHEMY_API_KEY_SOLANA
ALCHEMY_API_KEYS = {
    "ethereum": os.getenv("ALCHEMY_API_KEY_ETHEREUM"),
    "bnb": os.getenv("ALCHEMY_API_KEY_BNB"),
    "solana": os.getenv("ALCHEMY_API_KEY_SOLANA"),
}

# 2) Etherscan 멀티체인 API 키 (여러 체인 커버)
# ETHERSCAN_MULTICHAIN_API_KEY 또는 ETHERSCAN_API_KEY 사용
ETHERSCAN_MULTICHAIN_API_KEY = (
    os.getenv("ETHERSCAN_MULTICHAIN_API_KEY") or os.getenv("ETHERSCAN_API_KEY")
)

# 3) 검색 설정 (기본값 유지 가능)
SEARCH_KEYWORD = os.getenv("SEARCH_KEYWORD", "pepe")

# 참고) 라벨링된 주소 데이터는 필요 시 외부에서 주입하세요.
# 사용하는 곳이 제한적이므로 기본 샘플만 남깁니다.
# 기본 내장 라벨 (부족 시 외부 labels.yaml로 확장)
LABELED_ADDRESSES = {
    "ethereum": {
        "0x000000000000000000000000000000000000dead": "Burn Address",
    },
    "bnb": {
        "0x000000000000000000000000000000000000dead": "Burn Address",
    },
}

# 외부 라벨 파일 읽기 (선택: PyYAML 필요). 파일이 없거나 PyYAML 미설치면 무시.
def _load_external_labels() -> dict:
    path = os.getenv("LABELS_FILE") or os.path.join(os.path.dirname(__file__), "labels.yaml")
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # normalize lowercasing
        out: dict[str, dict[str, str]] = {}
        for chain, mapping in (data or {}).items():
            cm = {}
            if isinstance(mapping, dict):
                for addr, label in mapping.items():
                    if isinstance(addr, str) and isinstance(label, str):
                        cm[addr.lower()] = label
            out[str(chain).lower()] = cm
        return out
    except Exception:
        return {}

def _merge_labels(a: dict, b: dict) -> dict:
    res = {k: {kk.lower(): vv for kk, vv in v.items()} for k, v in a.items()}
    for chain, mapping in (b or {}).items():
        chain_l = str(chain).lower()
        res.setdefault(chain_l, {})
        for addr, label in (mapping or {}).items():
            res[chain_l][addr.lower()] = label
    return res

LABELS = _merge_labels(LABELED_ADDRESSES, _load_external_labels())

# --- Risk scoring weights (sum to 1). Override via env vars. ---
def _fenv(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v is not None and v.strip() != "" else default
    except Exception:
        return default

RISK_WEIGHTS = {
    "concentration": _fenv("W_CONCENTRATION", 0.30),
    "developer": _fenv("W_DEVELOPER", 0.40),
    "sniper": _fenv("W_SNIPER", 0.10),
    "insider": _fenv("W_INSIDER", 0.20),
}

# Optional: risk threshold for scam classification (0..100)
SCAM_THRESHOLD = _fenv("SCAM_THRESHOLD", 70.0)

# --- Optional: Multi‑provider RPC configuration ---
# These are built from environment variables if present.
# EVM providers: standard JSON‑RPC work everywhere; Alchemy enhanced methods require Alchemy endpoints.

def _env(name: str) -> str | None:
    v = os.getenv(name)
    return v if v and v.strip() else None

# Ethereum
_INFURA_KEY = _env("INFURA_API_KEY")
_ANKR_ETH_URL = _env("ANKR_ETH_RPC_URL")  # full URL
_PUBLIC_ETH_URL = _env("PUBLIC_ETH_RPC_URL")

# BNB
_ANKR_BNB_URL = _env("ANKR_BNB_RPC_URL")
_PUBLIC_BNB_URL = _env("PUBLIC_BNB_RPC_URL")

# Build Alchemy endpoints (if keys exist)
def _alchemy_url(base: str, key: str | None) -> str | None:
    return f"{base}{key}" if key else None

ALCHEMY_URLS = {
    "ethereum": _alchemy_url("https://eth-mainnet.g.alchemy.com/v2/", ALCHEMY_API_KEYS.get("ethereum")),
    "bnb": _alchemy_url("https://bsc-mainnet.g.alchemy.com/v2/", ALCHEMY_API_KEYS.get("bnb")),
    "solana": _alchemy_url("https://solana-mainnet.g.alchemy.com/v2/", ALCHEMY_API_KEYS.get("solana")),
}

_QUICKNODE_ETH = _env("QUICKNODE_ETH_RPC_URL")
_QUICKNODE_BNB = _env("QUICKNODE_BNB_RPC_URL")

EVM_RPC_PROVIDERS = {
    "ethereum": [
        url for url in [
            ALCHEMY_URLS.get("ethereum"),
            (f"https://mainnet.infura.io/v3/{_INFURA_KEY}" if _INFURA_KEY else None),
            _ANKR_ETH_URL,
            _QUICKNODE_ETH,
            _PUBLIC_ETH_URL,
        ] if url
    ],
    "bnb": [
        # Prefer QuickNode first on BNB for stability, then Alchemy, then Ankr/public
        url for url in [
            _QUICKNODE_BNB,
            ALCHEMY_URLS.get("bnb"),
            _ANKR_BNB_URL,
            _PUBLIC_BNB_URL,
        ] if url
    ],
}

SOLSCAN_API_KEY = _env("SOLSCAN_API_KEY")
HELIUS_API_KEY = _env("HELIUS_API_KEY")

# Derive Helius RPC URL from API key if explicit URL is not provided
_HELIUS_RPC_URL = _env("HELIUS_RPC_URL") or (f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}" if HELIUS_API_KEY else None)

SOLANA_RPC_PROVIDERS = [
    url for url in [
        _HELIUS_RPC_URL,                 # Primary: Helius
        _env("QUICKNODE_SOL_RPC_URL"),  # Failover: QuickNode
        ALCHEMY_URLS.get("solana"),     # Additional: Alchemy
        _env("PUBLIC_SOLANA_RPC_URL"),  # Public fallback
    ] if url
]

# Optional: external providers newly added (not yet used in pipeline)
GOLD_RUSH_KEY = os.getenv("GOLD_RUSH_KEY")
MORALIS_KEY = os.getenv("MORALIS_KEY")

# Feature flags
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

# Explorerless mode: avoid Etherscan/BscScan; prefer Sourcify + on-chain heuristics
EXPLORERLESS = _env_bool("EXPLORERLESS", False)

# --- Dev risk per-token time budget (seconds) ---
def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default

DEV_RISK_MAX_TOKEN_SEC = _env_int("DEV_RISK_MAX_TOKEN_SEC", 25)
