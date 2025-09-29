#!/usr/bin/env python3
from __future__ import annotations

"""
재조정(리밸런싱) 모니터링 실행 스크립트

예시 실행:
  python scripts/recalib_check.py \
    --base-csv data/base_scores.csv \
    --cur-csv data/cur_scores.csv \
    --score-col insider_score --target-col target_7d_return \
    --slack  # SLACK_WEBHOOK_URL 환경변수 필요

CSV 포맷(각 파일):
  date, chain, ff_bin, mcap_bin, event_type, insider_score, target_7d_return

동작:
  - base/cur CSV에서 score/target 컬럼을 추출해 PSI, RankIC 평가
  - 임계치 초과 시 Slack(옵션) 또는 콘솔로 알림
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Ensure repo root is on sys.path so 'insiders' package is importable when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from insiders.monitoring import (
    MonitoringThresholds,
    evaluate_recalibration,
    monitor_and_alert,
    DefaultAlertSink,
)
try:
    from insiders.alert_sinks.slack import SlackSink
except Exception:  # pragma: no cover
    SlackSink = None  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-csv", required=True)
    p.add_argument("--cur-csv", required=True)
    p.add_argument("--score-col", default="insider_score")
    p.add_argument("--target-col", default="target_7d_return")
    p.add_argument("--psi-warn", type=float, default=0.1)
    p.add_argument("--psi-alert", type=float, default=0.2)
    p.add_argument("--rankic-drop", type=float, default=0.2)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--slack", action="store_true", help="Slack 웹훅으로 알림 전송")
    p.add_argument("--slack-bot", action="store_true", help="Slack Bot 토큰으로 채널에 전송")
    p.add_argument("--at-channel", action="store_true", help="Slack @channel 멘션 추가")
    p.add_argument("--usergroup-id", default=None, help="Slack 사용자그룹 ID(멘션) 예: S0123ABCD")
    args = p.parse_args()

    base = pd.read_csv(args.base_csv)
    cur = pd.read_csv(args.cur_csv)
    base_scores = base[args.score_col].astype(float).tolist()
    cur_scores = cur[args.score_col].astype(float).tolist()
    base_target = base[args.target_col].astype(float).tolist()
    cur_target = cur[args.target_col].astype(float).tolist()

    th = MonitoringThresholds(
        psi_warn=args.psi_warn,
        psi_alert=args.psi_alert,
        rankic_drop_pct=args.rankic_drop,
        bins=args.bins,
    )

    results = evaluate_recalibration(
        base_scores=base_scores,
        cur_scores=cur_scores,
        base_target=base_target,
        cur_target=cur_target,
        thresholds=th,
    )

    sink = DefaultAlertSink()
    used = "console"
    if args.slack and SlackSink is not None and os.environ.get("SLACK_WEBHOOK_URL"):
        sink = SlackSink()  # type: ignore
        used = "slack_webhook"
    if args.slack_bot:
        try:
            from insiders.alert_sinks.slack_bot import SlackBotSink  # type: ignore

            sink = SlackBotSink(
                at_channel=args.at_channel,
                usergroup_id=args.usergroup_id,
            )
            used = "slack_bot"
        except Exception:
            pass
    monitor_and_alert(results, sink)
    if used != "console":
        print(f"[monitor] alert sink: {used}")


if __name__ == "__main__":
    main()
