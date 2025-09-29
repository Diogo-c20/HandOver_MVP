# 인수인계 — Insiders 위험 점수 엔진

이 문서는 지금까지 작업 내용 요약과 다시 시작할 때 필요한 실행/운영 방법을 쉬운 말로 정리한 문서입니다. 어려운 용어는 최대한 배제했습니다.

## 한눈에 보기(현재 상태)
- 핵심 엔진 완성: 자유유통량(시점 고정), 원지표 계산, 코호트 정규화(분포 기준 0–100), 이벤트(가중 벌점/감점), 시너지 감쇠, 가중합, 등급 산출.
- 설정 파일로 제어: `insiders/config_defaults.yaml` (가중치·임계치·정규화 정책).
- API 준비됨: `uvicorn insiders.api.server:app --reload` 로 실행, `POST /score/insiders`.
- 테스트 포함: `insiders/tests/` 유닛테스트로 핵심 로직 검증 가능.
- 모니터링/알림(옵션): 재조정 필요성 체크 스크립트, 슬랙 연동(웹훅/봇) 코드 존재. 지금은 “나중에” 진행하기로 함.

## 폴더/파일 안내(중요한 것만)
- `insiders/` 패키지 전체
  - `config_defaults.yaml`: 가중치·임계치 기본값
  - `metrics.py`: 원지표 계산(FR, EMR, SPI 등)
  - `normalize.py`: 0–100 정규화(분포 기준, 백오프/시간감쇠 포함)
  - `events.py`: 팀→CEX 등 이벤트 감지(벌점/감점)
  - `scorer.py`: 정규화된 지표 가중합 + 이벤트 더해 최종 점수
  - `api/server.py`: FastAPI 서버(POST /score/insiders)
  - `SCORING_FORMULA.txt`: 점수 산식 설명(요약)
- 스크립트/워크플로우
  - `scripts/recalib_check.py`: “재조정 필요 여부” 점검(분포 변화·예측력) + 슬랙 알림 옵션
  - `.github/workflows/recalib.yml`: 깃허브 액션(수동 실행/스케줄) 템플릿

## 점수 산식(쉬운 설명)
- 원지표는 “시장에 미치는 영향” 위주로 몇 가지 비율을 만듭니다.
  - 예) 팀 지갑이 거래소로 보낸 양 / 자유유통량(t0 고정), 특정 지갑군의 매수 비중 등
- 각 지표는 그대로 비교하면 토큰·체인·규모가 달라 불공정하니, “비슷한 그룹(코호트)” 분포에서 내 위치(백분위/중앙값 기준)를 점수(0–100)로 바꿉니다.
- 라벨 신뢰도(확실/휴리스틱/패턴)에 따라 1.0/0.8/0.6 배로 보수화합니다.
- 정규화된 지표를 가중치대로 섞어 기본 점수를 만들고, 이벤트(벌점/감점)를 더합니다. 중복 이벤트는 큰 것 100% + 나머지 60%만 더합니다.
- 마지막으로 0~100으로 자르고 등급(HIGH/MED/MID/LOW)을 붙입니다.

## 지금 당장 쓰는 법(로컬)
1) 가상환경 만들기/활성화
   - `python3 -m venv .venv && source .venv/bin/activate`
2) 필수 설치
   - `python -m pip install -U numpy pandas pydantic fastapi uvicorn`
3) API 서버 실행
   - `uvicorn insiders.api.server:app --reload`
4) 점수 요청(예)
   - POST `http://127.0.0.1:8000/score/insiders` (요청/응답은 `insiders/api/schemas.py` 참고)
5) 테스트
   - `python -m pip install -U pytest`
   - `pytest -q`

## 가중치/설정 변경
- 파일: `insiders/config_defaults.yaml`
- 바꾸면 바로 로직에 반영됩니다(서버 재시작 필요 시 재시작).
- 기본 가중치(초기값): FR_vol 15, FR_mkt 10, EMR_team 10, team_to_cex_over_ff 5, SPI_VC 10, SPI_mkt 5

## (나중에) 슬랙 알림 연결 — 전체 흐름
“재조정 필요” 경보가 떴을 때 팀 채널로 통보하기 위한 준비입니다. 지금은 미루기로 했고, 나중에 아래 순서대로 진행하면 됩니다.

- 준비물(중 하나 선택)
  - 웹훅 방식: 채널별 웹훅 URL 1개만 있으면 됨
  - 봇 방식: 봇 토큰(xoxb-), 채널 ID(예: C09104BMCGH), (선택) @channel/사용자그룹 멘션

- 로컬에서 빠른 테스트(둘 중 하나)
  - 웹훅: `export SLACK_WEBHOOK_URL='...'` → `python -m insiders.slack_integration --message "테스트(Webhook)" --prefer webhook`
  - 봇: `export SLACK_BOT_TOKEN='xoxb-...' && export SLACK_CHANNEL_ID='C09104BMCGH'` → `python -m pip install slack_sdk && python -m insiders.slack_integration --message "테스트(Bot @channel)" --prefer bot --at-channel`

- 깃허브 액션으로 자동화(원할 때)
  1) 레포 Settings → Secrets and variables → Actions → New repository secret 추가
     - `SLACK_BOT_TOKEN` = xoxb-…
     - `SLACK_CHANNEL_ID` = C09104BMCGH
     - (선택) `SLACK_AT_CHANNEL` = 1
  2) Actions 탭 → “Insiders Recalibration Monitor” → Run workflow (수동 실행)
  3) 스케줄 자동 실행을 원하면 `.github/workflows/recalib.yml`의 `schedule:` 유지(UTC 기준)

## (나중에) 재조정(리밸런싱) — 왜/언제/어떻게
- 왜: 시장 환경이 바뀌면(분포/행동 패턴 변화) 초기 가중치가 덜 맞을 수 있습니다.
- 언제: 분기별 점검 또는 경보 발생 시(분포 변화↑, 예측력↓).
- 어떻게(쉬운 버전)
  1) 데이터 두 묶음 준비(같은 스키마)
     - `data/base.csv`: 과거 구간 점수/성과
     - `data/cur.csv`: 최근 구간 점수/성과
     - 최소 컬럼: `insider_score`, `target_7d_return`(또는 유출량 등)
  2) 재조정 체크 실행
     - `python scripts/recalib_check.py --base-csv data/base.csv --cur-csv data/cur.csv --score-col insider_score --target-col target_7d_return`
     - 경보가 나오면, 가중치를 소폭 조정(±20% 범위 탐색, 합계 동일하게 정규화)해 성능이 가장 좋은 조합으로 업데이트
  3) 적용
     - `insiders/config_defaults.yaml`의 `weights` 수정 → 서버 재시작
  4) 모니터링
     - 1~2주 섀도 모니터링 후 본 적용

## 다음에 해야 할 일(체크리스트)
- [ ] 슬랙 연동 시점 결정(웹훅 vs 봇)
- [ ] 연동 선택 시, Secrets 설정(봇: `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`) 또는 웹훅 URL 준비
- [ ] 깃허브 Actions 수동 실행으로 1회 점검(성공 시 채널 메시지 확인)
- [ ] 실제 운영 데이터로 `data/base.csv`, `data/cur.csv` 대체
- [ ] 재조정 필요 시 가중치 소폭 조정 → `config_defaults.yaml` 업데이트

## 자주 막히는 것 & 해결
- “SSL 인증서 에러(로컬)”: `python -m pip install certifi` 또는 macOS `Install Certificates.command` 실행
- “슬랙 메시지 안 옴”
  - 봇: 토큰 정확한지, 채널에 봇 초대됐는지(/invite), 권한(chat:write) 있는지
  - 웹훅: URL 정확한지, 방화벽/프록시 이슈 없는지
  - GitHub Actions: Secrets 이름/값 정확한지, Actions 권한 허용됐는지
- “깃허브 푸시 안 됨(PAT)”
  - 토큰 만료/권한(repo, workflow) 확인, 또는 SSH 전환

## 보안/운영 팁
- 비밀키는 절대 커밋 금지(레포 Secrets 또는 .env 사용).
- 큰 CSV는 Git LFS 고려(경고는 기능에 영향 없지만 권장 크기 초과).
- 설정 변경은 PR로 리뷰 후 반영 추천.

필요 시, 슬랙 알림/재조정 자동화를 다시 켜는 것부터 같이 진행하면 됩니다. 지금 상태로도 API/점수 산출은 즉시 사용 가능합니다.

*** 끝 ***
