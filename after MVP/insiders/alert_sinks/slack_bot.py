from __future__ import annotations

"""Slack Bot(웹 API) 기반 알림 싱크.

필수 설정:
- SLACK_BOT_TOKEN: xoxb- 로 시작하는 봇 토큰
- SLACK_CHANNEL_ID: 채널 ID (예: C0123ABCD). 이름(#channel) 대신 ID 권장

선택 설정:
- SLACK_USERGROUP_ID: 사용자 그룹 ID (예: S0123ABCD) 있으면 <!subteam^ID> 멘션

설치:
  pip install slack_sdk

사용:
  from insiders.alert_sinks.slack_bot import SlackBotSink
  sink = SlackBotSink()
  sink.send("재조정 경보")
"""

import os
from typing import Optional
import ssl

try:
    import certifi  # type: ignore
except Exception:  # pragma: no cover - 선택 의존성 미설치 허용
    certifi = None  # type: ignore

try:
    from slack_sdk import WebClient  # type: ignore
    from slack_sdk.errors import SlackApiError  # type: ignore
except Exception:  # pragma: no cover - 선택 의존성 미설치 허용
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore

from ..monitoring import AlertSink


class SlackBotSink(AlertSink):
    def __init__(
        self,
        *,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
        usergroup_id: Optional[str] = None,
        at_channel: bool = False,
    ) -> None:
        self.bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self.channel_id = channel_id or os.environ.get("SLACK_CHANNEL_ID", "")
        self.usergroup_id = usergroup_id or os.environ.get("SLACK_USERGROUP_ID", "")
        self.at_channel = at_channel
        # macOS 등에서 SSL 루트 인증서 이슈를 피하기 위해 certifi 기반 컨텍스트 시도
        ssl_ctx = None
        try:
            if certifi is not None:
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_ctx = None
        self._client = (
            WebClient(token=self.bot_token, ssl=ssl_ctx) if self.bot_token and WebClient else None
        )

    def _prefix(self) -> str:
        parts = []
        if self.at_channel:
            parts.append("<!channel>")
        if self.usergroup_id:
            parts.append(f"<!subteam^{self.usergroup_id}>")
        return " ".join(parts) + (" " if parts else "")

    def send(self, message: str) -> None:  # pragma: no cover - 외부 호출
        if not self._client or not self.channel_id:
            return
        try:
            text = self._prefix() + message
            self._client.chat_postMessage(channel=self.channel_id, text=text)
        except SlackApiError:
            # 조용히 실패 (알림 경로 이슈가 엔진 흐름을 막지 않도록)
            pass
