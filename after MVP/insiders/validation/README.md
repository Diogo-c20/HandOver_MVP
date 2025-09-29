# 검증 리소스

Insider Score 모델 검증을 위한 파이썬 파이프라인과 Dune SQL 템플릿을 담고 있습니다.

## 구성
- `validation/insider_validation.py` — 전체 파이프라인(정답셋 생성 → 컴포넌트 검증 → 최종 점수 검증)
- `validation/sql/` — Dune SQL 템플릿 3종
  - `pre_dex_receivers.sql` — 첫 유동성 이전 배포자→지갑 토큰 전송 수신자
  - `deployer_funded_wallets.sql` — 배포자 가스 지원 + 초기 매수 지갑
  - `obvious_non_insiders.sql` — 상장 30일 이후 첫 매수 + CEX 자금

## 준비
- Python 3.10+, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `dune-client`
- Dune API 키: `export DUNE_API_KEY=...`
- SQL 템플릿의 파라미터(`{{token_address}}`, `{{hours_pre_liq}}`, `{{post_liq_minutes}}`, `{{sample_size}}`)를 실제 값으로 치환하거나, 파이썬에서 변수를 주입하세요.

## 실행
- 파이프라인: `python /Users/chorch66/Desktop/Hyperindex/insiders/validation/insider_validation.py`
- 결과물: `artifacts/validation/` 아래 PNG(지표 박스플롯, 점수 분포, PR 커브) 및 콘솔에 임계값(60/80) 성능 지표 출력

## Dune SQL 사용 팁
- 환경에 따라 `dex.trades` 대신 DEX별 트레이드 테이블(예: `uniswap_v2_ethereum.trades`)로 교체하세요.
- 라벨 소스(`labels.*`)는 팀이 관리하는 최신 라벨 테이블로 대체하세요.
- 쿼리를 작은 CTE 단위로 검증하고, 중간 결과 카디널리티를 체크하며 조인 조건을 명확히 하세요.

## 재검증(리밸리데이션) 가이드
- 일정: 최소 주 1회(예: 월요일 09:00) 정기 실행 추천.
- 즉시 재검증 트리거:
  - 모델/가중치/피처 로직 변경 시
  - 신규 토큰/체인 추가 또는 라벨 업데이트 발생 시
  - DEX/라벨 스키마 변경, ETL 백필/리프로세싱 이후
  - 운영 지표(Precision/Recall/F1)가 목표 대비 하락 감지 시
- 운영 팁:
  - 실행 시점별 아티팩트 폴더를 타임스탬프로 분리해 추세 비교
  - 쿼리 버전/파라미터/모델 버전을 함께 로그/메타데이터(JSON)로 저장
  - 이전 실행과 핵심 지표를 비교하는 간단한 리그레션 체크를 추가하세요.
