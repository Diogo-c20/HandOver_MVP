# 엔티티 클러스터링(탐색기 없는 1차 휴리스틱)

이 문서는 탐색기(Etherscan 등) 없이 온체인 로그와 최소 메타만으로 지갑들을 동일 엔티티로 묶는 1차 휴리스틱 로직을 설명합니다.

## 목적
- 동일 주체가 운용하는 다지갑 주소들을 자동으로 묶어 위험 분석(내부자·스나이퍼·집결지)을 보조합니다.
- 고비용 탐색기/라벨 API 없이도 최근 구간의 전송 로그로 근사 클러스터를 산출합니다.

## 입력 데이터
- 토큰 전송 기록 DataFrame
  - 필드: `block_timestamp`, `from_address`, `to_address`, `value` 등
  - 타임스탬프가 없을 경우에도 작동하지만, 있으면 동시행동(co-trade) 판정 정확도가 향상됩니다.

## 핵심 휴리스틱과 산식
1) Co-trade(동시 매수/수신 묶음)
- 아이디어: DEX 페어(추정)에서 같은 블록 또는 ±Δt 내 다수 주소가 동시에 토큰을 수신하면, 동일 엔티티(또는 협조군)일 가능성이 큼.
- 구현:
  - `pair = argmax_addr(count(from_address) + count(to_address))`로 가장 자주 등장하는 상대 주소를 페어로 추정
  - 스트림을 시간순으로 훑으며 `from_address == pair` 인 레코드의 `to_address`를 한 배치(batch)에 모음
  - 배치 간 시간 차가 `window_sec`를 초과하면 배치를 닫고, 배치 내 수신자들을 Union-Find로 연결
- 파라미터: `window_sec`(기본 10초), `min_group`(기본 2)

2) Sink(공통 목적지 집결 묶음)
- 아이디어: 여러 발신자가 동일 목적지(페어 제외)로 반복 송금하면, 해당 발신자들이 동일 주체일 가능성.
- 구현:
  - 상위 `max_events` 내에서 `to_address` 카운트를 집계하여 빈번한 목적지 집합 `S`를 선택(빈도 ≥ max(3, min_group))
  - 각 목적지 `s∈S`에 대해, `to==s`인 레코드의 발신자 집합을 모으고 Union-Find로 연결
- 파라미터: `max_events`(기본 1000), `min_group`(기본 2)

3) 클러스터 산출
- Union-Find(Disjoint-Set)로 생성된 연결 성분을 주소 리스트로 정렬하여 반환
- 정렬 기준: 클러스터 크기 내림차순

## 출력
- `List[List[str]]` 형태의 주소 클러스터 목록
- 예: `[[addr1, addr2, ...], [addrX, addrY], ...]`

## 한계 및 보정 포인트
- 라벨 부재: 거래소/브릿지/락커/트레저리 주소가 포함될 수 있으므로, 라벨(내부 DB·Moralis/GoldRush)로 제외/버킷화 권장
- 금액/가스 유사도: co-trade 시 `amount`, `gasPrice` 유사성(±ε) 필터를 추가하면 정확도 향상
- 타임윈도: `window_sec`, `max_events`는 데이터 양/체인 상황에 맞춰 조정 필요
- 전 구간이 아닌 최근 윈도우(로그 폴백)에서는 절대 잔액/오래된 관계가 반영되지 않을 수 있음

## 사용 방법(요약)
- 코드: `EntityClustering/entity_clustering.py`의 `cluster_entities()`를 호출
- 최소 필요 컬럼: `from_address`, `to_address` (선택: `block_timestamp`)
- 예시:
```python
from EntityClustering.entity_clustering import cluster_entities
clusters = cluster_entities(transfers_df, window_sec=10, min_group=2, max_events=1000)
```

## 향후 보강 제안
- 라벨 통합: 거래소/브릿지/락커/트레저리 라벨을 적용하여 노이즈 제거
- Feature 추가: 금액 로그스케일 유사도, 가스 상위 분위수, 반복 빈도 가중치
- Trace(조건부): Deployer/내부이체를 통한 출처/목적지 정확도 향상
- Solana: 페어/LP 이벤트와 계정 그래프를 이용한 유사 규칙 확장

