# Dune 프리필터 작업 개요

## 목적
- Dune 단계에서 절대치 컷(유동성·거래·고유트레이더·시총·나이)으로 검색 후보 토큰을 줄입니다.
- 기존 검색 CTE는 수정하지 않고, 추가 CTE(`dex_stats`, `liq`, `meta`)를 합쳐 최종 WHERE 컷을 적용합니다.

## 폴더 구조
- `sql/prefilter_template.sql`: Dune SQL 템플릿(플레이스홀더 기반)
- `config/prefilter.example.yaml`: Balanced 프리셋 예시 구성
- `examples/cli_examples.txt`: CLI 실행 예시 모음
- `ENV.md`: 환경변수와 뷰/CTE 이름 오버라이드 방법

## 사용 방법(요약)
- 구성 파일을 필요에 맞게 수정: `config/prefilter.example.yaml`
- 실행 스크립트(저장 + 자동 폴더 지정):
  - `python collect_tokens.py --query "pepe" --format csv --config prefilter.example.yaml`
- Dune API가 직접 SQL 실행을 허용하지 않는 경우(플랜/엔드포인트 제약):
  1) `python collect_tokens.py --query "pepe" --dry-run`으로 최종 SQL 추출
  2) Dune Studio에 붙여넣어 저장(쿼리 ID 복사)
  3) `.env`에 `DUNE_QUERY_ID=<복사한_ID>` 추가 후 정상 실행

## 컷 기준(기본: balanced)
- `liq_usd ≥ 6,000,000`
- `vol24h_usd ≥ 2,000,000`
- `trades24h ≥ 250`
- `unique_traders24h ≥ 120`
- `mcap_usd ∈ [20,000,000, 5,000,000,000]`
- `age_d ≥ 3일`

## 베이스페어 화이트리스트
- 심볼: `USDC, USDT, DAI, FRAX, TUSD, FDUSD, WETH, WBTC, WBNB, SOL, WSOL`
- 체인별 주소: YAML의 `base_pairs.addresses`에 주입
- 옵션: `include_legacy_stables=false` 시 BUSD 등 레거시 심볼/주소 제외

## 로그/리트라이
- 429/타임아웃 시 최대 3회 지수 백오프
- 적용 파라미터 및 최종 건수 로그 출력

## Dune 뷰/CTE 이름 확인 방법
- Dune Studio에서 기존 “검색 결과” 쿼리를 열어 tokens 후보를 내는 CTE/서브쿼리 이름을 확인하고, 그 이름을 `search_result_cte`로 맞추거나 `.env`의 `DUNE_SEARCH_RESULT_CTE`로 지정하세요.
- 자체 뷰가 없다면, 먼저 Studio에서 프리필터 SQL을 저장해 `DUNE_QUERY_ID` 방식으로 실행하는 것을 권장합니다.
