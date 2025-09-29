from __future__ import annotations

"""
슬랙 연동 통합 모듈 (Webhook + Bot)

개요
- 환경변수 또는 인자를 통해 Slack Webhook / Slack Bot 방식을 자동 선택해 메시지 전송
- 기본 전략: Bot 토큰과 채널 ID가 있으면 Bot 우선, 없으면 Webhook 사용
- 필요 시 둘 다 활성화해 동시 전송(MultiSink)

환경변수 (선택)
- SLACK_WEBHOOK_URL: 웹훅 URL (Incoming Webhooks)
- SLACK_BOT_TOKEN: xoxb- 로 시작하는 봇 토큰
- SLACK_CHANNEL_ID: 채널 ID (예: C0123ABCD)
- SLACK_USERGROUP_ID: 사용자그룹 ID (예: S0123ABCD)
- SLACK_AT_CHANNEL: "1" / "true" 이면 <!channel> 멘션 추가

CLI 사용 예(레포 루트에서)
  python -m insiders.slack_integration --message "테스트" --prefer auto
  python -m insiders.slack_integration --message "봇 멘션" --prefer bot --at-channel \
      --channel-id C09104BMCGH --bot-token xoxb-...
  python -m insiders.slack_integration --message "웹훅" --prefer webhook \
      --webhook-url https://hooks.slack.com/services/XXX/YYY/ZZZ
"""

import argparse
import os
from typing import Optional

from .monitoring import AlertSink
from .alert_sinks.multi import MultiSink
from .alert_sinks.slack import SlackSink
from .alert_sinks.slack_bot import SlackBotSink


def _truthy(s: Optional[str]) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y", "on"}


class SlackIntegration(AlertSink):
    """슬랙 연동 통합 싱크. Bot/Webhook 중 가용 경로로 전송.

    prefer:
      - "auto": Bot 가능하면 Bot, 아니면 Webhook
      - "bot": Bot만
      - "webhook": Webhook만
      - "both": 두 경로 동시 전송
    """

    def __init__(
        self,
        *,
        prefer: str = "auto",
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
        usergroup_id: Optional[str] = None,
        at_channel: bool = False,
    ) -> None:
        self.prefer = prefer
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN")
        self.channel_id = channel_id or os.environ.get("SLACK_CHANNEL_ID")
        self.usergroup_id = usergroup_id or os.environ.get("SLACK_USERGROUP_ID")
        self.at_channel = at_channel or _truthy(os.environ.get("SLACK_AT_CHANNEL"))

        self._sink: AlertSink = self._build_sink()

    def _has_bot(self) -> bool:
        return bool(self.bot_token and self.channel_id)

    def _has_webhook(self) -> bool:
        return bool(self.webhook_url)

    def _build_sink(self) -> AlertSink:
        prefer = self.prefer.lower()
        has_bot = self._has_bot()
        has_webhook = self._has_webhook()

        if prefer == "both" and (has_bot or has_webhook):
            sinks = []
            if has_webhook:
                sinks.append(SlackSink(self.webhook_url))
            if has_bot:
                sinks.append(
                    SlackBotSink(
                        bot_token=self.bot_token,
                        channel_id=self.channel_id,
                        usergroup_id=self.usergroup_id,
                        at_channel=self.at_channel,
                    )
                )
            return MultiSink(*sinks) if sinks else SlackSink("")

        if prefer in {"auto", "bot"} and has_bot:
            return SlackBotSink(
                bot_token=self.bot_token,
                channel_id=self.channel_id,
                usergroup_id=self.usergroup_id,
                at_channel=self.at_channel,
            )
        if prefer in {"auto", "webhook"} and has_webhook:
            return SlackSink(self.webhook_url)

        # fallback: no-op sink
        return SlackSink("")

    def send(self, message: str) -> None:  # pragma: no cover - 외부 전송
        self._sink.send(message)


def main() -> None:  # pragma: no cover - CLI 유틸
    ap = argparse.ArgumentParser(description="Slack 통합 전송 유틸")
    ap.add_argument("--message", required=True)
    ap.add_argument("--prefer", default="auto", choices=["auto", "bot", "webhook", "both"])
    ap.add_argument("--webhook-url", default=None)
    ap.add_argument("--bot-token", default=None)
    ap.add_argument("--channel-id", default=None)
    ap.add_argument("--usergroup-id", default=None)
    ap.add_argument("--at-channel", action="store_true")
    args = ap.parse_args()

    integ = SlackIntegration(
        prefer=args.prefer,
        webhook_url=args.webhook_url,
        bot_token=args.bot_token,
        channel_id=args.channel_id,
        usergroup_id=args.usergroup_id,
        at_channel=args.at_channel,
    )
    integ.send(args.message)


if __name__ == "__main__":
    main()

