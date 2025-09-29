# Normalization Formula (정규화 산식)

학습/런타임 모두 동일한 코호트 기반 정규화를 적용하여 피처 분포 차이를 보정합니다. 변수명은 영문, 설명은 한글입니다.

## 코호트 정의
- `cohort_keys = ["chain", "tvl_quartile", "mcap_bin"]` (기본)
- 각 코호트별로 수치 피처의 위치/척도 통계를 보관하여 정규화에 사용

## 수치 피처 정규화 (Robust Scaling)
- 대상: 원-핫을 제외한 수치 피처(예: `t_since_t0_sec`, `gas_price_pctile`, `buy_size_vs_liquidity`, `same_block_swaps`, `is_contract_caller`)
- 통계: 코호트별 `median`, `IQR = Q3 - Q1`
- 변환: `x_norm = (x - median) / max(IQR, ε)` with `ε = 1e-12`
- IQR=0일 경우: ε로 나눔(또는 향후 MAD 대체 가능)
- 원-핫 피처는 정규화하지 않음

## 퍼센타일 파생(예시)
- `gas_price_pctile`: 체인 단위 퍼센타일 랭크(`groupby(chain).rank(pct=True) * 100`)

## 런타임 재현
- 학습 시 저장한 `feature_spec.yaml`에서 `feature_order`, `one_hot_features`, `scaling`(코호트별 통계)을 로드
- 입력 피처를 동일 순서로 정렬하고 코호트 통계를 적용해 `x_norm` 계산
- 존재하지 않는 원-핫 컬럼은 0으로 채움(동일 차원 맞춤)
