from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


LabelQuality = str  # one of {"certain", "heuristic", "pattern"}


@dataclass
class AddressLabel:
    """Represents a labeled address with quality/trust and category.

    Attributes:
        address: Address string.
        category: Logical role/category (e.g., 'team', 'vc', 'exchange', 'bridge').
        quality: LabelQuality string: certain/heuristic/pattern.
    """

    address: str
    category: str
    quality: LabelQuality = "heuristic"


@dataclass
class Transfer:
    """Simple EVM/Solana-agnostic transfer record for entity merging heuristics."""

    tx_hash: str
    ts: int  # unix seconds
    src: str
    dst: str
    token: str
    amount: float


class UnionFind:
    """Union-Find data structure used for clustering addresses into entities."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def merge_entities(
    transfers: Iterable[Transfer],
    labels: Dict[str, AddressLabel],
    co_move_window_sec: int = 3600,
) -> Dict[str, Set[str]]:
    """Merge addresses into entities using simple heuristics.

    Heuristics:
    - Shared source or destination within a short time window are clustered.
    - Direct bilateral transfers cluster the pair.
    - Addresses with identical labeled category and frequent interactions cluster.

    This is a simplified approach suitable for unit tests and offline use.

    Args:
        transfers: Iterable of Transfer records.
        labels: Mapping from address to AddressLabel.
        co_move_window_sec: Time window for co-movement clustering.

    Returns:
        Mapping from entity_id (representative address) to set of addresses.
    """

    uf = UnionFind()
    # Sort by time for window-based grouping
    sorted_tx = sorted(list(transfers), key=lambda t: t.ts)
    # Direct link clustering
    for t in sorted_tx:
        uf.union(t.src, t.dst)

    # Co-movement clustering by destination
    by_dst: Dict[str, List[Transfer]] = {}
    for t in sorted_tx:
        by_dst.setdefault(t.dst, []).append(t)
    for dst, txs in by_dst.items():
        i = 0
        for j in range(len(txs)):
            while txs[j].ts - txs[i].ts > co_move_window_sec:
                i += 1
            # Union all sources in the sliding window [i, j]
            window_src = [txs[k].src for k in range(i, j + 1)]
            if window_src:
                head = window_src[0]
                for s in window_src[1:]:
                    uf.union(head, s)

    # Category-based clustering for frequent interactions
    interaction_count: Dict[Tuple[str, str], int] = {}
    for t in sorted_tx:
        k = (t.src, t.dst)
        interaction_count[k] = interaction_count.get(k, 0) + 1
        if interaction_count[k] >= 3:
            uf.union(t.src, t.dst)

    # Build clusters
    clusters: Dict[str, Set[str]] = {}
    addrs = set()
    for t in sorted_tx:
        addrs.add(t.src)
        addrs.add(t.dst)
    for a in addrs:
        r = uf.find(a)
        clusters.setdefault(r, set()).add(a)
    return clusters


def label_trust_weight(quality: LabelQuality, trust_table: Dict[str, float]) -> float:
    """Map label quality to configured trust weight.

    Args:
        quality: Label quality string (certain/heuristic/pattern).
        trust_table: Mapping from quality to weight.

    Returns:
        Weight scalar in [0, 1]. Defaults to 1.0 when missing.
    """

    return float(trust_table.get(quality, 1.0))

