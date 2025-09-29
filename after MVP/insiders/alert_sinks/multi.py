from __future__ import annotations

"""여러 AlertSink로 동시 전송하는 멀티 싱크."""

from typing import Iterable

from ..monitoring import AlertSink


class MultiSink(AlertSink):
    def __init__(self, *sinks: AlertSink) -> None:
        self.sinks = list(sinks)

    def send(self, message: str) -> None:  # pragma: no cover - 사이드이펙트
        for s in self.sinks:
            try:
                s.send(message)
            except Exception:
                # 일부 싱크 실패해도 나머지는 계속 전송
                pass

