# Sniper Pipeline Handoff (최신 인수인계)

본 문서는 스나이퍼 점수화 파이프라인을 “다시 시작해도 동일하게” 이어가기 위한 실행 절차, 설정, 통합 포인트, 알림/트러블슈팅을 요약합니다.

## 1) 현재 상태 요약
- 위치: `~/Desktop/Hyperindex/snipers/`
- 실데이터 경로 구축 완료: Dune Saved Query로 t0/초기거래 수집 → 약라벨 → 학습 → 아티팩트 생성
- 아티팩트: `artifacts/{model_weights.json, feature_spec.yaml, thresholds.yaml}`
- Dune Saved Query IDs: t0=`5770522`, early=`5772250`
- 임계 완화: `discover.min_first_window_usd=1000` (필요 시 조정)

## 2) 주요 경로/설정/환경
- 설정: `~/Desktop/Hyperindex/snipers/config.yaml`
  - `discover.query_id_discover: 5770522`
  - `discover.query_id_early_trades: 5772250`
  - `augment.enable_rpc: true` (선택, Alchemy 키 필요)
- 폴더: `~/Desktop/Hyperindex/snipers/{cache|dataset|artifacts}`
- 환경(.env): `~/Desktop/Hyperindex/.env`
  - 필수: `DUNE_API_KEY`
  - 선택: `ALCHEMY_API_KEY_ETH`
  - 러너/페처가 .env 자동 로드(SNIPER_CONFIG_PATH 기본값 포함)

## 3) 실행 방법
- 일괄(권장, 실데이터):
  - `python3 ~/Desktop/Hyperindex/snipers/sniper_real_pipeline.py --config "$SNIPER_CONFIG_PATH"`
  - 순서: fetch_real_data → weak-label → train(logreg_l1)
- 수동 단계별:
  - 수집: `python3 ~/Desktop/Hyperindex/snipers/fetch_real_data.py --config "$SNIPER_CONFIG_PATH"`
  - 약라벨: `python3 ~/Desktop/Hyperindex/snipers/sniper_pipeline_runner.py --config "$SNIPER_CONFIG_PATH" weak-label --snorkel 0`
  - 학습: `python3 ~/Desktop/Hyperindex/snipers/sniper_pipeline_runner.py --config "$SNIPER_CONFIG_PATH" train --model logreg_l1`
  - 스코어: `python3 ... score --input dataset/early_trades.parquet --out dataset/scored.parquet`

## 4) 강라벨(우선 학습) 지원
- 파일 위치: `~/Desktop/Hyperindex/snipers/dataset/scam_labels.csv|parquet`
- 스키마: `token_address,label`(0/1). 존재 시 해당 표본은 강라벨, 나머지는 약라벨 혼합 학습.

## 5) 메인 파이프라인 연동
- 파일: `~/Desktop/Hyperindex/main_pipeline.py`
- 변경: `compute_sniper_subscore`가 학습된 아티팩트를 사용(토큰별 early_trades → FeatureBuilder → SniperScorer)
- 폴백: 데이터/아티팩트 부재 시 기존 휴리스틱(초기 10분 매수자 보유비중)

## 6) 알림(운영)
- 로컬 알림: `~/Desktop/Hyperindex/alerts.py` (JSONL 로그 + 콘솔 에코)
- 스나이퍼 내 알림 트리거(`sniper_ml_pipeline.py`):
  - 아티팩트 신선도 초과/부재 → 재학습 알림
  - 재학습 후 `test_pr_auc` 낮음 → 경고 알림
- 싱크: insiders의 `DefaultAlertSink` 우선 사용(동일 UX) + 로컬 로그. Slack 연동은 전체 파이프라인 완성 후 동일 인터페이스로 확장.

## 7) 자주 겪는 이슈 및 해결
- `python` not found → `python3` 사용
- `.env` 미적용 → 러너/페처가 자동 로드(별도 export 불필요)
- Dune 실행 컬럼 중복(`t0_x/t0_y`) → 페처가 `t0`로 정규화
- 약라벨 단계 KeyError(실데이터 결측 컬럼) → 가드로 0(무표) 처리
- PR 커브 에러(단일 클래스) → τ=0.5 폴백

## 8) 튜닝 포인트
- 데이터 양 확대: `discover.min_first_window_usd` 하향(예: 1000→500), `t0_window_seconds` 확장(예: 300→600)
- 운영 목표: `training.target_precision` 조정, 필요 시 `maybe_update_weights(config, freshness_days=N)` 주기 실행

## 9) 다음 액션(추천)
- 메인 파이프라인 정기 시작 시 `maybe_update_weights`로 신선도/성능 알림 확인
- Slack/Discord 연동 시 insiders의 `slack_integration`를 스나이퍼 알림 싱크로 이식
- 드리프트 모니터(PSI/RankIC) 경고 → 재캘리브 알림/자동화 보강

문의/추가 변경이 필요하면 이 문서 하단에 변경 이력을 덧붙여 주세요.
>   chain,
>   token_address,
>   arbitrary(pair_address) AS pair_address,
>   MIN(block_time) AS t0,
>   arbitrary(dex_type) AS dex_type,
>   approx_percentile(amount_usd, 0.5) AS tvl
> FROM buys
> GROUP BY 1,2;
>
> 초기 거래 쿼리(하드코딩 버전)
> WITH v2 AS (
>   SELECT 'ethereum' AS chain, block_time, tx_hash,
>          token_bought_address AS token_address,
>          pair_address AS pair_address,
>          trader, amount_usd
>   FROM uniswap_v2_ethereum.trades
>   WHERE block_time >= date_add('day', -90, now())
>     AND token_sold_symbol IN ('USDC','USDT','DAI')
> ),
> v3 AS (
>   SELECT 'ethereum' AS chain, block_time, tx_hash,
>          token_bought_address AS token_address,
>          pool_address AS pair_address,
>          trader, amount_usd
>   FROM uniswap_v3_ethereum.trades
>   WHERE block_time >= date_add('day', -90, now())
>     AND token_sold_symbol IN ('USDC','USDT','DAI')
> ),
> buys AS (
>   SELECT * FROM v2
>   UNION ALL
>   SELECT * FROM v3
> ),
> t0 AS (
>   SELECT token_address, MIN(block_time) AS t0
>   FROM buys
>   GROUP BY token_address
> ),
> w AS (
>   SELECT b.*, t.t0
>   FROM buys b
>   JOIN t0 t USING (token_address)
>   WHERE b.block_time BETWEEN t.t0
>     AND date_add('second', 300, t.t0)
> ),
> tx_agg AS (
>   SELECT
>     'ethereum' AS chain,
>     token_address,
>     arbitrary(pair_address) AS pair_address,
>     MIN(block_time) AS block_time,
>     tx_hash,
>     arbitrary(trader) AS wallet,
>     SUM(amount_usd) AS usd_value
>   FROM w
>   GROUP BY token_address, tx_hash
> ),
> qual AS (
>   SELECT token_address, SUM(usd_value) AS window_usd
>   FROM tx_agg
>   GROUP BY token_address
> )
> SELECT a.*, q.window_usd
> FROM tx_agg a
> JOIN qual q USING (token_address)
> WHERE q.window_usd >= 10000;
>
> - 방법 B: 통합 뷰(dex.trades) 계속 쓰려면 이걸로
>   - pair 주소는 project_contract_address만 쓰는 쪽이 안전.
>   - 아래처럼 바꿔서 저장.
>
> t0 쿼리(dex.trades 버전)
> WITH buys AS (
>   SELECT
>     'ethereum' AS chain,
>     block_time,
>     tx_hash,
>     token_bought_address AS token_address,
>     project_contract_address AS pair_address,
>     project AS dex_type,
>     amount_usd
>   FROM dex.trades
>   WHERE blockchain = 'ethereum'
>     AND block_time >= date_add('day', -90, now())
>     AND token_sold_symbol IN ('USDC','USDT','DAI')
> )
> SELECT
>   chain,
>   token_address,
>   arbitrary(pair_address) AS pair_address,
>   MIN(block_time) AS t0,
>   arbitrary(dex_type) AS dex_type,
>   approx_percentile(amount_usd, 0.5) AS tvl
> FROM buys
> GROUP BY 1,2;
>
> 초기 거래도 같은 방식으로 buys/t0/w/tx_agg/qual 구조로 작성하면 됨. trader 컬럼 유무는 테이블 스키마에 따라 다를 수 있으니 Data Explorer로 확인 권장.
