"""
엔티티 클러스터링(탐색기 없이 동작하는 1차 휴리스틱)

- 입력: 토큰 전송 기록 DataFrame (columns: block_timestamp, from_address, to_address, value ...)
- 출력: 주소 리스트들의 리스트 (각 리스트가 하나의 클러스터)

휴리스틱 개요
- co-trade: 추정 페어 주소로부터 같은 블록/±window_sec 내 다수 주소가 동시에 토큰을 받은 경우 묶음
- sink: 동일 목적지 주소(페어 제외)에 여러 발신자가 반복적으로 송금한 경우 발신자들을 묶음

주의사항
- 이 1차 버전은 탐색기/API 없이도 동작하도록 설계된 근사치입니다.
- 정확도를 높이려면 거래소/브릿지/락커 라벨 제외, 금액/가스 유사도 필터 추가가 권장됩니다.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _parse_iso(ts: str):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


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


def cluster_entities(
    transfers_df,
    *,
    window_sec: int = 10,
    min_group: int = 2,
    max_events: int = 1000,
):
    """전송 기록 기반 엔티티 클러스터링(1차 휴리스틱)

    - co-trade: 추정 페어(from=pair)로부터 같은 블록/±window_sec 내 수신자들을 묶음
    - sink: 동일 목적지(페어 제외)에 여러 발신자가 반복 송금하면 발신자를 묶음

    returns: List[List[str]]  # 주소들의 클러스터 목록(내림차순: 큰 클러스터 우선)
    """
    try:
        if transfers_df is None or transfers_df.empty:
            return []
        df = transfers_df.copy()
        # 타임스탬프 정규화
        if "block_timestamp" in df.columns:
            df["_ts"] = df["block_timestamp"].apply(lambda x: _parse_iso(x) if x else None)
        else:
            df["_ts"] = None
        df = df.sort_values([c for c in ["_ts"] if c in df.columns])
        pair = _guess_pair_address(df)
        uf = _UF()

        # 1) co-trade 엣지
        if pair:
            batch: list[str | None] = []
            prev_ts = None
            for _, r in df.iterrows():
                if r.get("from_address") and str(r.get("from_address")).lower() == pair:
                    ts = r.get("_ts")
                    if prev_ts is None:
                        prev_ts = ts
                    # 시간차가 window를 넘으면 묶음 처리 후 초기화
                    if prev_ts and ts and (ts - prev_ts).total_seconds() > window_sec:
                        recips = [str(x).lower() for x in batch if _is_addr(x)]
                        if len(recips) >= min_group:
                            root = recips[0]
                            for addr in recips[1:]:
                                uf.union(root, addr)
                        batch = []
                    prev_ts = ts
                    batch.append(r.get("to_address"))
            # 마지막 배치 flush
            recips = [str(x).lower() for x in batch if _is_addr(x)]
            if len(recips) >= min_group:
                root = recips[0]
                for addr in recips[1:]:
                    uf.union(root, addr)

        # 2) sink 엣지 (동일 목적지로 반복 송금하는 다수 발신자)
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

        # 3) 연결 성분으로 클러스터 산출
        comps: dict[str, set[str]] = defaultdict(set)
        all_addrs = set()
        if "from_address" in df.columns:
            all_addrs |= {str(x).lower() for x in df["from_address"].astype(str).tolist() if _is_addr(x)}
        if "to_address" in df.columns:
            all_addrs |= {str(x).lower() for x in df["to_address"].astype(str).tolist() if _is_addr(x)}
        for addr in all_addrs:
            rep = uf.find(addr)
            comps[rep].add(addr)
        clusters = [sorted(list(members)) for members in comps.values() if len(members) >= min_group]
        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters
    except Exception:
        return []

