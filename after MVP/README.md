# Crypto Risk Ingestion

크립토 리스크 파이프라인을 위한 프로덕션급 데이터 수집 SDK와 비동기 워커를 제공합니다.

하이라이트
- 비동기 어댑터: Alchemy, GoldRush, Moralis, Helius, Etherscan, Subgraphs
- 회로 차단 브로커, 토큰 버킷 레이트 리밋, 지터 포함 재시도
- Redis 캐시/락, Postgres 메타데이터, ClickHouse 시계열
- 엄격한 타이핑(MyPy), Ruff 린트, structlog JSON 로깅

빠른 시작
1. `.env` 생성 후 값 채우기(예: `.env.example` 참고).
2. 설치: `poetry install`
3. 린트/테스트근데: `make lint && make test`
4. CLI 예시:
   - `poetry run python -m src.cli ingest holders --chain eth-mainnet --token 0x... --from-block 0 --to-block 100000`
   - `poetry run python -m src.cli backstop holders --chain eth-mainnet --token 0x... --block-height 200000`
   - `poetry run python -m src.cli scan lp --chain bsc-mainnet --token 0x...`
   - `poetry run python -m src.cli parse rights --chain eth-mainnet --address 0x...`

프로젝트 구조는 `src/` 하위에 위치하며, `core/`(SDK), `workers/`(잡), `infra/` 헬퍼로 구성됩니다. 세부 계약은 각 모듈의 docstring을 참고하세요.

## Dune 프리필터 수집(collect)
Balanced 프리셋의 절대 컷으로 토큰 후보를 줄이는 단계입니다. 기존 검색 CTE는 유지하고, 24h 체결/거래/고유 트레이더, TVL, 시총/나이를 기준으로 필터링합니다.

- 실행 예시:
  - `poetry run python -m src.cli collect --query "pepe" --preset balanced --max-candidates 200 \
     --base-symbols "USDC,USDT,DAI,FRAX,TUSD,FDUSD,WETH,WBTC,WBNB,SOL,WSOL" \
     --include-legacy-stables false --config prefilter.example.yaml`
- 환경변수(옵션): `DUNE_API_KEY`, `DUNE_API_URL`, `DUNE_SEARCH_RESULT_CTE`, `DUNE_DEX_TRADES_24H_VIEW`, `DUNE_DEX_POOLS_VIEW`, `DUNE_TOKEN_META_VIEW`
- 출력: 후보 건수와 각 행(JSON 문자열). 결과가 없으면 생성된 SQL을 함께 출력합니다. 추가 컬럼: `liq_usd, vol24h_usd, trades24h, unique_traders24h, age_d, mcap_usd`.
- 구성 우선순위: 프리셋 기본 → per_chain(YAML) → 환경변수(`PREFILTER_*`) → CLI 인자.
- 구성 예시는 `prefilter.example.yaml` 참고.


## Hyperindex 파이프라인(탐색기 없는 모드) 보강 사항
- Sourcify 연동: 탐색기 없이 컨트랙트 검증/소스 확보(내부 `main_pipeline.py`에 반영).
- 온체인 폴백: `eth_getLogs` 기반 전송/홀더 재구성(윈도/이벤트 상한 CLI/ENV로 제어).
- 엔티티 클러스터링(1차): `--cluster` 옵션으로 활성화.
  - 로직 분리: `EntityClustering/entity_clustering.py` 참조.
  - 설명서: `EntityClustering/README_ko.md` 참고.
- 라벨 파일: `labels.yaml`(체인별 주소→라벨) 추가, `config.py`가 자동 로드(PyYAML 설치 시). 없는 경우 내장 샘플만 사용.

### 라벨 사용 방법
1) `labels.yaml`에 체인 키 하위에 주소와 라벨을 추가합니다.
2) (선택) `pip install pyyaml` 설치 시 자동 병합 로드됩니다.
3) 홀더/전송에 라벨이 반영되어 집중도·클러스터링의 노이즈가 감소합니다.


- 클러스터링 결과 저장: `--cluster-out clusters.json` 옵션으로 JSON 저장(토큰/체인/파라미터/클러스터 목록 포함).
- 파라미터 제어: `--cluster-min-group`, `--cluster-window-sec`, `--cluster-top-k`.

### 클러스터링 실행 예시
```bash
python3 main_pipeline.py --chains ethereum --max-tokens 1 --dex-fallback-only \n  --cluster --log-window-blocks 20000 --log-max-events 400
```
