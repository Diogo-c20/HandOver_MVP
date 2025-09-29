# ML Weights Formula (학습 산식)

아래 산식은 스나이퍼 탐지용 가중치(모델 파라미터)를 산출하기 위한 절차입니다. 변수명은 영문, 설명은 한글로 표기합니다.

## 입력/피처
- `x`: 피처 벡터. 학습 시 `feature_spec.feature_order` 순서로 정렬, 코호트 정규화(아래 참고) 적용 후 사용
- 주요 피처 예: `t_since_t0_sec`, `gas_price_pctile`, `buy_size_vs_liquidity`, `same_block_swaps`, `is_contract_caller`, `router_*`(one-hot)
- 타깃(약라벨): `weak_label_prob ∈ [0,1]` (LF 결합 결과)

## 약한 라벨 결합(Weak Labeling)
- LF 목록:
  - `LF_early_and_gas`: `t_since_t0_sec ≤ early_threshold_sec AND gas_price_pctile ≥ gas_percentile`
  - `LF_same_block_burst`: `same_block_swaps ≥ same_block_swaps_min`
  - `LF_known_bot_router`: `router ∈ known_sniper_routers`
  - `LF_team_cex_lp_exclude`: `exclusion_bucket ∈ {burn, lp_lock, treasury, ...}` → 음성(-1)
- 결합:
  - Snorkel 사용 시: `LabelModel(cardinality=2)`로 `p_hat = P(y=1|LFs)` 추정
  - Fallback: 가중 다수결 `z = Σ w_i * y_i` (y_i∈{-1,0,1}), `p_hat = sigmoid(z)`

## PU 학습(옵션)
- Stage A (P vs U):
  - `P = {i | weak_label_prob_i ≥ 0.9}`, `U = 나머지`
  - 임시 분류기로 점수 `s` 예측 후, `RN = {u ∈ U | s_u < α}` 선택
- Stage B (P vs RN):
  - 최종 분류기 학습. 파이프라인은 데이터 부족 시 자동 지도학습으로 폴백

## 지도학습(필수)
- 기본: L1 Logistic Regression (해석 가능 가중치)
  - 로짓: `z = w · x + b`
  - 확률: `p = sigmoid(z) = 1/(1+exp(-z))`
  - 손실: `L = mean( BCE(y, p) ) + λ ||w||₁`
  - 클래스 불균형 보정: class_weight=balanced (폴백 구현 시 가중치 없이 GD)
- 보조(선택): XGBoost (Binary:logistic)
  - 트리 기반. 내보낼 때는 피처 중요도를 선형 계수로 근사(coef), 절편=0 가정

## 분할/지표/임계값
- 분할: 토큰 기준 시간 순(train/test)
- 지표: PR-AUC, Precision@K(상위 k% 컷)
- 임계값(정책):
  - `precision_at_k`: 임계값 `τ`를 탐색해 목표 정밀도 이상에서 최대 재현율 선택
  - `f1_max`: F1 최대점 `τ` 선택
  - `fixed_tau`: 고정 `τ` 사용(기본 0.5)

## 산출물
- `model_weights.json`: `model_type`, `feature_order`, `coef`(w), `intercept`(b), `metrics`
- `thresholds.yaml`: `global: τ` (코호트별 τ는 확장 가능)
