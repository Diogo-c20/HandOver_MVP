from __future__ import annotations

"""
재조정(Recalibration) 모니터링 가이드 및 알림 유틸리티

이 모듈은 가중치/정규화 정책의 재조정 필요 조건과 운영 가이드를 문서화하고,
동시에 간단한 모니터링/알림 함수를 제공합니다. 외부 네트워크 호출은 사용처에서
주입 가능한 방식(웹훅 등)으로 분리되어 있으며, 기본 구현은 콘솔 출력(AlertSink)입니다.

재조정 필요 조건(권장)
- 데이터 드리프트: 체인·FF 분위·시총 분포가 바뀌거나 이벤트 빈도가 달라짐
  - 모니터링: PSI(Population Stability Index), KL-divergence 등 분포 비교
  - 트리거 예시: PSI > 0.2 경고/재캘리브레이션 후보
- 성능 저하: 스코어 상위 그룹의 사후 7/30일 수익률·유출지표 예측력이 약화
  - 모니터링: RankIC(Spearman), 상·하위 분위 수익률/유출량 스프레드
  - 트리거 예시: |RankIC_current| < (1 - 0.2) * |RankIC_prev|
- 정책 변화: 이벤트 패널티/완화 강도 조정 필요(팀→CEX 급증 등)
  - 모니터링: 이벤트 비중/강도 시계열, 점수 기여도 변화

운영 권장사항
- 정기 점검: 분기별 재캘리브레이션 시도
- 트리거 대응: PSI 경고 지속, RankIC 하락 지속 시 가중치 그리드 탐색 → 검증 → 적용
- 가드레일: 이벤트 합성 감쇠(0.6) 유지, 단일 이벤트 상한 옵션 검토

사용 예시
>>> from insiders.monitoring import (
...     MonitoringThresholds, evaluate_recalibration, DefaultAlertSink, monitor_and_alert
... )
>>> thresholds = MonitoringThresholds()
>>> results = evaluate_recalibration(
...     base_scores=[...], cur_scores=[...],
...     base_target=[...], cur_target=[...],
...     thresholds=thresholds,
... )
>>> monitor_and_alert(results, DefaultAlertSink())

참고: RankIC 계산은 numpy 기반(간이 스피어만). PSI는 분위(bin) 방식으로 계산합니다.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple

import math
import numpy as np


@dataclass
class MonitoringThresholds:
    """모니터링 임계치 설정.

    psi_warn: 경고 수준 PSI
    psi_alert: 재캘리브레이션 후보로 간주할 PSI
    rankic_drop_pct: RankIC 절대값 하락률 임계(예: 0.2 = 20% 하락)
    min_sample: 통계량 신뢰를 위한 최소 표본 수
    bins: PSI 계산시 사용할 분위 bin 수
    """

    psi_warn: float = 0.1
    psi_alert: float = 0.2
    rankic_drop_pct: float = 0.2
    min_sample: int = 100
    bins: int = 10


def _rankdata(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    # 평균 순위(동순위 처리) 간소화: 같은 값 처리 미세오차 허용
    # 정밀한 tie-handling이 필요하면 scipy.stats.rankdata 사용을 권장
    return ranks


def spearman_rank_correlation(x: Iterable[float], y: Iterable[float]) -> Optional[float]:
    """간이 Spearman 상관. 표본 부족 시 None 반환."""

    x = list(x)
    y = list(y)
    n = min(len(x), len(y))
    if n < 3:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def _quantile_bins(values: Iterable[float], bins: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return arr
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(arr, qs)
    # 가장자리 중복 시 소량 확장
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def psi_from_values(ref: Iterable[float], cur: Iterable[float], bins: int = 10, eps: float = 1e-6) -> Optional[float]:
    """연속값 기반 PSI 계산. ref(기준) vs cur(현재).

    PSI = sum((p_i - q_i) * ln(p_i / q_i)) with smoothing.
    표본이 부족하면 None 반환.
    """

    ref = np.asarray(list(ref), dtype=float)
    cur = np.asarray(list(cur), dtype=float)
    n_ref, n_cur = len(ref), len(cur)
    if n_ref < 10 or n_cur < 10:
        return None
    edges = _quantile_bins(ref, bins)
    # 히스토그램 계산
    r_hist, _ = np.histogram(ref, bins=edges)
    c_hist, _ = np.histogram(cur, bins=edges)
    r_prob = (r_hist + eps) / (r_hist.sum() + eps * len(r_hist))
    c_prob = (c_hist + eps) / (c_hist.sum() + eps * len(c_hist))
    psi = np.sum((r_prob - c_prob) * np.log(r_prob / c_prob))
    return float(psi)


@dataclass
class RecalibrationResults:
    psi: Optional[float]
    rankic_prev: Optional[float]
    rankic_cur: Optional[float]
    flags: dict


class AlertSink(Protocol):
    def send(self, message: str) -> None:  # pragma: no cover - interface only
        ...


class DefaultAlertSink:
    """기본 알림 싱크: 콘솔 출력."""

    def send(self, message: str) -> None:  # pragma: no cover - side-effect only
        print(message)


def evaluate_recalibration(
    *,
    base_scores: Iterable[float],
    cur_scores: Iterable[float],
    base_target: Iterable[float],
    cur_target: Iterable[float],
    thresholds: MonitoringThresholds = MonitoringThresholds(),
) -> RecalibrationResults:
    """재조정 필요 조건을 평가.

    - PSI(ref=base_scores, cur=cur_scores)
    - RankIC(prev=base_scores vs base_target, cur=cur_scores vs cur_target)
    """

    base_scores = list(base_scores)
    cur_scores = list(cur_scores)
    base_target = list(base_target)
    cur_target = list(cur_target)

    psi = psi_from_values(base_scores, cur_scores, bins=thresholds.bins)
    ic_prev = spearman_rank_correlation(base_scores, base_target)
    ic_cur = spearman_rank_correlation(cur_scores, cur_target)

    flags = {}
    # PSI 플래그
    if psi is not None:
        flags["psi_warn"] = psi >= thresholds.psi_warn
        flags["psi_alert"] = psi >= thresholds.psi_alert
    else:
        flags["psi_insufficient"] = True

    # RankIC 하락 플래그
    if ic_prev is not None and ic_cur is not None and abs(ic_prev) > 1e-9:
        flags["rankic_drop"] = abs(ic_cur) < (1.0 - thresholds.rankic_drop_pct) * abs(ic_prev)
    else:
        flags["rankic_insufficient"] = True

    return RecalibrationResults(psi=psi, rankic_prev=ic_prev, rankic_cur=ic_cur, flags=flags)


def should_alert(results: RecalibrationResults) -> bool:
    """하나라도 강한 트리거가 있으면 알림."""

    f = results.flags
    return bool(f.get("psi_alert") or f.get("rankic_drop"))


def format_alert_message(results: RecalibrationResults) -> str:
    psi_s = f"{results.psi:.3f}" if results.psi is not None else "NA"
    icp = f"{results.rankic_prev:.3f}" if results.rankic_prev is not None else "NA"
    icc = f"{results.rankic_cur:.3f}" if results.rankic_cur is not None else "NA"
    parts = [
        "[Insiders] 재조정 모니터링 알림",
        f"PSI={psi_s}, RankIC(prev={icp}, cur={icc})",
    ]
    on = [k for k, v in results.flags.items() if v is True]
    if on:
        parts.append("Flags: " + ", ".join(sorted(on)))
    return " | ".join(parts)


def monitor_and_alert(results: RecalibrationResults, sink: AlertSink) -> None:
    """강한 트리거 발생 시 알림 싱크로 메시지 전송."""

    if should_alert(results):  # pragma: no cover - side-effect only
        sink.send(format_alert_message(results))

