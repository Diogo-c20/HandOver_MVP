# 하이퍼인덱스 엔티티 클러스터링 인수인계

이 문서는 탐색기 없는(Explorerless) 모드로 구축한 엔티티 클러스터링과 관련된 현재 상태, 실행 방법, 설정, 다음 단계 제안을 정리합니다. 이 파일만 보면 이전 대화 없이도 동일한 작업을 이어갈 수 있습니다.

## 현재 상태 요약
- 탐색기 미사용(EXPLORERLESS=1)으로 동작 확인.
- 전송/홀더 수집:
  - 1차: Alchemy 강화 API(가능 시)
  - 폴백: `eth_getLogs` 기반 최근 윈도우 수집, 블록 타임스탬프는 `eth_getBlockByNumber` 캐시 활용
  - 홀더 재구성: 최근 로그로 from/to/value 합산(근사), 라벨 적용
- 클러스터링 휴리스틱(1차): `EntityClustering/entity_clustering.py`
  - co-trade: 추정 페어(from=pair)로부터 같은 블록/±window_sec 내 수신자 묶음
  - sink: 동일 목적지(페어 제외)에 여러 발신자가 반복 송금 시 발신자 묶음
  - Union-Find로 연결 성분 산출
- 라벨: `labels.yaml` + `config.py` 자동 로드(있으면 병합). 홀더/전송에 `label/from_label/to_label` 반영.
- main_pipeline 통합: `--cluster` 플래그로 실행 시 클러스터링 수행, 콘솔 출력 및 JSON 저장 옵션 제공.

## 주요 파일/변경점
- `EntityClustering/entity_clustering.py`: 클러스터링 로직 모듈화(입력 DF → 클러스터 목록)
- `EntityClustering/README_ko.md`: 한글 설명(목적/휴리스틱/파라미터/한계/예시)
- `EntityClustering/HANDOFF.md`: 본 인수인계 문서
- `labels.yaml`: 체인별 주소 라벨 샘플. PyYAML 설치 시 `config.LABELS`에 자동 병합
- `config.py`
  - `LABELS` 로더 추가(LABELS_FILE 경로 지원)
  - `EXPLORERLESS` 플래그 지원(Explorer 경로 미사용)
  - BNB RPC 우선순위: QuickNode → Alchemy → Ankr → Public
- `main_pipeline.py`
  - Sourcify 기반 소스/검증(탐색기 미사용 시)
  - 폴백 로깅/유틸: `get_logs_transfers`, `enriched_transfers_from_logs`, `reconstruct_holders_from_logs`
  - 라벨 적용: `apply_labels(holders_df, transfers_df, chain)`
  - 클러스터링 실행(옵션): `--cluster`, 출력 Top-K, JSON 저장 `--cluster-out`
  - 폴백 파라미터화: `--log-window-blocks`, `--log-max-events`

## 실행 방법
1) 프로젝트 폴더에서 실행(.env가 여기 있기 때문)
```bash
cd /Users/chorch66/Desktop/Hyperindex
# (선택) 가상환경 활성화: source .venv/bin/activate
export EXPLORERLESS=1
python3 main_pipeline.py   --chains ethereum   --max-tokens 1   --dex-fallback-only   --cluster   --log-window-blocks 20000   --log-max-events 400   --cluster-top-k 5   --cluster-out clusters.json
```
- 결과: 콘솔에 상위 K개 클러스터 프리뷰, `clusters.json` 파일에 클러스터 전체 저장.

2) BNB(주소 직접 지정 예시)
```bash
python3 main_pipeline.py   --chains bnb   --dex-fallback-only   --token-addresses bnb:0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c   --cluster --log-window-blocks 20000 --log-max-events 400
```

## 구성/환경 변수
- `.env`(프로젝트 루트)에서 Alchemy, QuickNode 등 키를 로드합니다.
- Explorerless 모드: `EXPLORERLESS=1`
- 라벨 파일 경로: `LABELS_FILE=/full/path/to/labels.yaml` (미설정 시 기본 `./labels.yaml`)
- 폴백 제어(ENV도 지원):
  - `LOG_FALLBACK_WINDOW_BLOCKS` (기본 50000)
  - `LOG_FALLBACK_MAX_EVENTS` (기본 1000)

## 검증 포인트(체크리스트)
- [ ] 콘솔에 `🧩 클러스터 수: N | 상위 클러스터 크기: M` 출력 확인
- [ ] `clusters.json`가 생성되고, `clusters` 배열에 주소 리스트들이 들어있는지 확인
- [ ] `labels.yaml`에 추가한 라벨이 holders/transfers의 `label/from_label/to_label`로 반영되는지 확인
- [ ] Alchemy 호출량이 과하면 `--log-window-blocks`, `--log-max-events` 축소로 재실행

## 한계/주의
- 로그 기반 홀더 재구성은 “최근 윈도우” 기준의 근사치입니다(절대 잔액과 다를 수 있음).
- 페어 추정은 가장 빈번한 카운터파티를 활용합니다. 라벨(예: Pair/Router)을 확충하면 정확도↑
- 탐색기 비사용으로 txlist/internal 같은 세부는 트레이스(QuickNode/Infura) 연동 시 보강 가능합니다.

## 다음 단계 제안(우선순위)
1) 라벨 정제
   - labels.yaml 확충(CEX/브릿지/락커/트레저리/Pair/Router)
   - 클러스터링 전처리에서 라벨 주소를 별도 버킷화 또는 제외
2) 결과 저장(선택)
   - ClickHouse/PG에 `entity_clusters`, `entity_members` 적재(DAL 메소드 추가)
   - CLI: `--persist` 옵션으로 저장 토글
3) Co-trade 필터 정교화
   - 금액(로그스케일) 유사도, 가스 상위 분위수 필터(옵션)
4) Sink 버킷화/완화
   - 상시 목적지 후보(멀티시그/트레저리/Pair/Router) 완화 규칙
5) Moralis/GoldRush 라벨/홀더 병합
   - 호출량 감소 + 라벨 신뢰도 향상

## 트러블슈팅
- `can't open file '/Users/.../main_pipeline.py'`: 현재 디렉토리 문제. 프로젝트 폴더로 이동해서 실행하세요.
- 네트워크 오류 수가 많음: 로그 윈도/이벤트 상한 축소, 요청 간격 증가(`MIN_REQUEST_INTERVAL_SEC`), 프로바이더 우선순위 조정
- 라벨 미반영: PyYAML 설치 또는 labels.yaml 경로 확인

## 문의/컨벤션
- 코드 수정 시 README.md(루트)에 사용법/옵션을 동기화합니다.
- 파라미터 기본값은 보수적으로 유지하고, 비용 이슈 시 CLI/ENV로 조절합니다.
