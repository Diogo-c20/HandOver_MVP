from __future__ import annotations

"""Slack 알림 싱크 구현.

사용법:
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    from insiders.alert_sinks.slack import SlackSink
    sink = SlackSink()  # 또는 SlackSink(webhook_url="...")
    sink.send("테스트 메시지")

외부 의존성: requests (선택 설치: pip install requests)
"""

import os
from typing import Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - 선택 의존성 미존재시 import 실패 허용
    requests = None  # type: ignore

from ..monitoring import AlertSink


class SlackSink(AlertSink):
    """Slack 웹훅 기반 AlertSink.

    환경변수 SLACK_WEBHOOK_URL 또는 인자로 전달된 webhook_url을 사용합니다.
    requests 미설치 또는 전송 실패 시 예외를 발생시키지 않고 조용히 실패합니다.
    """

    def __init__(self, webhook_url: Optional[str] = None, timeout_sec: float = 5.0) -> None:
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
        self.timeout_sec = timeout_sec

    def send(self, message: str) -> None:  # pragma: no cover - 외부 요청 부문
        if not self.webhook_url or requests is None:
            # requests 미설치 또는 웹훅 미설정 시 무시
            return
        try:
            requests.post(self.webhook_url, json={"text": message}, timeout=self.timeout_sec)
        except Exception:
            # 알림 경로 실패는 엔진 동작을 막지 않도록 무시
            pass

