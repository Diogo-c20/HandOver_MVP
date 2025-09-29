# Snipers Docs (HyperIndex)

이 디렉터리는 스나이퍼 탐지 관련 문서/설명을 모읍니다. 코드와 아티팩트는 아래 경로를 참고하세요.

- 런타임/학습 파이프라인: `sniper_ml_pipeline.py` (레포 루트)
- 파이프라인 러너: `hyperindex/snipers/sniper_pipeline_runner.py` (루트 파이프라인을 위임 실행)
- 아티팩트(가중치/피처/임계값): `./artifacts/` (자동 생성)
- 데이터셋 캐시: `./dataset/`, `./cache/`

## 실행 개요
1) 발견/데이터셋: Dune → 초기구간 거래 수집 (t0~t0+윈도우)
2) 보강: RPC/탐색기 (가스/라우터/EOA/라벨)
3) 약라벨 결합 → PU/지도학습 → 임계값 산출
4) 아티팩트 저장 후, `SniperScorer`로 점수 계산

예시:
```
python sniper_ml_pipeline.py discover --days 90 --chains ethereum
python sniper_ml_pipeline.py build-dataset --from-cache 1
python sniper_ml_pipeline.py weak-label --snorkel 0
python sniper_ml_pipeline.py train --model logreg_l1
python sniper_ml_pipeline.py score --input ./dataset/early_trades.parquet --out ./dataset/scored.parquet
```

## 환경 변수
- `DUNE_API_KEY`, `ALCHEMY_API_KEY_ETH`, `ETHERSCAN_API_KEY` (Ethereum)
- (옵션) BNB/SOL: `ALCHEMY_API_KEY_BNB`, `BSCSCAN_API_KEY`, `HELIUS_API_KEY_SOL`, `SOLSCAN_API_KEY`

## 메인 파이프라인 연동
- 가중치는 `./artifacts/model_weights.json` 등에 저장되며, Config는 파라미터/경로/정책을 제어합니다.
- 예시:
```
from sniper_ml_pipeline import get_sniper_scorer, maybe_update_weights
maybe_update_weights("./config.yaml", freshness_days=30)
scorer = get_sniper_scorer("./artifacts")
scored = scorer.score_df(early_trades_features_df)
```

## 참고
- 선택 의존성(xgboost, snorkel) 없으면 자동 폴백; parquet 미지원 시 CSV 폴백.
- 체인별 모델 분리는 선택 사항. 기본은 단일 모델 + 코호트 정규화(`chain` 포함).

## 산식 문서
- `01_ml_weights_formula.md`: 약라벨 결합, PU/지도학습, L1 로지스틱(선택 XGBoost) 학습 산식
- `02_sniper_scoring_model.md`: 런타임 스코어 계산 및 판정 산식
- `03_normalization_formula.md`: 코호트 기반 정규화/퍼센타일/스케일 산식
