# 그래프 생성 감사: 원 논문 조건과 현재 구현의 구분

최초 감사 기준: 2026-09-12, 실행 코드 `34b0487`. 2026-09-13의 실제 입력 수정은
마지막 절에 별도로 기록한다. 이 문서는 학습 완료 보고서가 아니다.
현재 구현을 **ASGCN 저자의 실험 조건과 완전히 동일하다고 판정할 수 없다.**
현재 반경·시간 축척·샘플링·풀링의 선택을 확인하지 않은 상태에서 대규모
학습을 재시작하거나, 저장 방식을 바꿨다는 이유로 문제 해결을 주장하면 안 된다.

## 1. 원문에서 확인한 것과 확인하지 못한 것

[ASGCN 원문](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)의
pp.1622–1624, Table 4와 현재 코드를 대조했다.

| 항목 | 원문에서 확인되는 범위 | 현재 구현 / 판정 |
|---|---|---|
| 입력 | `(x,y,t,p)` 이벤트 | 해당 형식으로 변환 |
| 샘플링 | 그래프 생성 전 균일 샘플링 R | sequence ordinal `0,R,2R,...`; R=1이면 전부 유지 |
| 연결 | 유클리드 거리 `< D`, 무방향 그래프 | 3축 정규화 거리, 양방향 저장/계수 |
| 좌표 단위 | 사용 축·축척을 확정할 충분한 명세 미확인 | `x/(W-1), y/(H-1), (t-origin)/s_t`는 프로젝트 선택 |
| 시간 창 | sliding window ΔT | 수치와 경계 정책은 프로젝트의 명시적 선택 |
| 국소 계산 | 새 노드의 K-hop 영향 범위 | 현재 구현의 실제 연산량은 별도 검증 대상 |
| 중간 pooling | 평균 feature, 원본 edge의 cluster 연결로 remap | 4층 뒤 pooling, 셀 크기·edge feature 집계는 프로젝트 선택 |
| 과제 | 이벤트 분류 | U-Net 영상 복원은 별도 응용, 원 실험 재현이 아님 |

Table 4는 N-Cars에서 R=10…80, D=2…9를 비교한다. R=10과 D=2를 모든
데이터셋의 고정 조건으로 해석하거나, D=2를 곧바로 2픽셀로 해석할 수 없다.
EventHDR/EventAid에 그대로 적용할 근거도 아니다.

[공식 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32154),
[공식 Poster 연결](https://underline.io/lecture/111062-leveraging-asynchronous-spiking-neural-networks-for-ultra-efficient-event-based-visual-processing),
[교신저자 공식 소개](https://yjsjy.uestc.edu.cn/gmis/jcsjgl/dsfc/dsgrjj/20675?yxsh=08)
및 제목·저자명 검색에서 저자 구현 저장소나 추가 설정 명세를 확인하지 못했다.
이는 공개 코드가 어디에도 없다는 증명이 아니다. 동명의 감성분석 ASGCN 코드를
이 논문의 저자 코드로 사용하지 않는다.

## 2. 현재 코드가 만드는 그래프의 정확한 의미

관련 코드: `data/eventhdr.py`, `data/eventaid_r.py`, `stream_model.py::_prepared`,
`stream_graph.py`, `implicit_radius.py`, `stream_sampling.py` (모두 `src/asgcn_unet` 아래).

현재 3축 설정에서 서로 다른 노드 i,j의 연결 조건은 다음과 같다.

```text
((x_i-x_j)/(W-1))² + ((y_i-y_j)/(H-1))² + ((t_i-t_j)/s_t)² < r²
```

실제 부동소수점 판정은 float64 좌표의 `norm((pos_i-pos_j)/r) < 1`이다.
경계 등호는 제외한다. polarity는 노드 feature에 있지만 3축 거리에는 없다.
서로 다른 시퀀스는 연결하지 않고, self-edge는 제외한다. 동일 값의 이벤트가
원본에 두 행으로 있으면 두 노드이며, 서로의 거리가 0인 연결이 가능하다.
원본 행을 임의로 중복 제거하지 않는다.

시간 원점은 시퀀스에 고정된다. window가 움직일 때 기존 노드의 좌표를
다시 min/max 정규화하지 않는다. 노드 feature의 시간은 그 노드가 전달된
프레임 구간의 시작 기준이고, topology 시간은 고정 시퀀스 원점 기준이다.
이 둘은 코드에서 서로 다른 용도로 사용된다.

readout 시각 q에서 `t >= q - window_seconds`인 전달 완료 노드가 남는다.
cutoff와 정확히 같은 이벤트는 유지된다. **EventHDR에서는 타임스탬프만으로
원본 전체를 자르면 안 된다.** 해당 프레임의 `end_idx`까지 전달된 원본 prefix
`[0:end_idx)` 안에서 window를 골라야 한다. 공식 predecessor index 방식 때문에
시각은 readout 이전이지만 아직 해당 프레임에서 전달하지 않은 행이 있을 수 있다.

R은 데이터셋/프레임마다 카운터를 리셋하지 않는 시퀀스 균일 샘플링이다.
R=1, `max_events=null`, full-resolution/no-ROI 조건에서는 원본 이벤트를 유지한다.
이 조건은 과거 정적 모델의 8,192 이벤트 제한과 동일하지 않다.

## 3. 왜 수억 edge가 될 수 있는가

다음은 `r=0.08`, `s_t=0.05초`라는 **조건부 기하 계산**이다.
현재 서버의 설정 파일을 새로 읽어 검증한 측정치가 아니다.

| 센서 크기 W×H | 같은 시각의 x축 한계 | 같은 시각의 y축 한계 | 같은 위치의 시간 차 한계 |
|---|---:|---:|---:|
| 320×240 | 25.52 px | 19.12 px | 4 ms |
| 1180×720 | 94.32 px | 57.52 px | 4 ms |

위 값들은 타원체의 개별 축 절편이며 직육면체 안의 모든 쌍이 연결된다는 뜻은 아니다.
고정 정규화 반경은 큰 센서에서 더 넓은 픽셀 범위를 연결한다. 50 ms의 노드
수명과 4 ms의 같은 위치 연결 범위도 서로 다른 양이다.

R=1의 높은 이벤트 밀도와 위 거리 규칙이 결합하면 한 노드의 이웃 수가 커진다.
노드 수 N, 평균 차수 d의 양방향 edge 수는 `N*d`이다. 같은 graph를
무방향 한 쌍씩 세면 그 절반이다. 저장 방식이 `implicit_radius`여도 이 수는
줄지 않는다. SNN의 spike가 희소해도 정의된 graph 자체의 차수가 자동 감소하지 않는다.
4층 뒤 pooling은 첫 4층이 보는 raw graph를 없애지 않는다.

사용자가 제공한 과거 v3 보고서의 raw readout은 424,096,036 directed edges,
prefix-union 상한은 440,677,422였다. 상한은 실제 한 arrival prefix의 측정 최대와
다르다. 현재 v4의 182-batch 로그는 topology 구간 1,151,127 ms / batch wall
1,155,176 ms로 약 99.65%이다. 이는 graph 관련 구간이 병목이라는 근거이지
순수 CUDA kernel 시간이나 GPU utilization 측정은 아니다. 이 감사에서 속도 문제가
해결되었다고 주장하지 않는다.

## 4. 원본 데이터의 시간 단위 확인

### EventHDR: 저자 입력 코드가 초 단위를 뒷받침함

[저자 loader의 고정 revision](https://github.com/yunhao-zou/EventHDR/blob/151d0b6d05dfc39d0acce276796a30e9ac3ac5ac/data_loader/dataset.py)은
원본 `events/ts`와 image timestamp를 별도 배율 없이 읽고, 원본 시간에
초 단위 구간을 더한다. [공식 README](https://github.com/yunhao-zou/EventHDR)가
연결하는 [rosbag 변환 코드](https://github.com/TimoStoff/event_utils/blob/dc0a0712156bb0c3659d90b33e211fa58a83a75f/lib/data_formats/rosbag_to_h5.py)는
`secs + nsecs/1e9`로 이벤트와 프레임 시간을 만든다.
[H5 packager](https://github.com/TimoStoff/event_utils/blob/dc0a0712156bb0c3659d90b33e211fa58a83a75f/lib/data_formats/event_packagers.py)는
이를 그대로 저장한다. 따라서 이 형식의 두 배율 1.0을 뒷받침하는 근거가 있다.
모든 배포 파일의 실제 생성 revision까지 입증한 것은 아니다.

로컬 `data/EventHDR/train/26.h5`의 제한된 metadata/원본 slice 확인 결과:
events 1,118,211개, ts float64, 약 1.000001초 구간, image 간격 약 0.002초,
센서 320×240이다. 이는 저자 코드와 일치하지만 전체 서버 데이터 검사 결과는 아니다.

### EventAid-R: 필드명은 단위의 증거가 아님

[공식 배포 페이지](https://sites.google.com/view/eventaid-benchmark)에서 이번 조사로
시간 단위 명세/변환 코드를 확인하지 못했다. 로컬 R-bear의 frame/event 시간은
공유하는 정수 기반으로 보이지만, 그것만으로 마이크로초를 증명할 수 없다.
우리 코드의 `t0_us/t1_us`라는 이름도 독립적인 근거가 아니다.
따라서 `1e-6`의 author-format 근거는 **미확인**으로 남긴다.
두 시계를 같은 잘못된 배율로 변환해도 내부 정합성 검사는 통과할 수 있다.

[저자 공개 논문 미러](https://liboyu02.github.io/assets/pdf/Duan_TPAMI25.pdf)의
동기화·timestamp 대응 설명과 [arXiv 원문](https://arxiv.org/html/2312.08220)에서도
배포 TXT 열의 저장 단위는 확인하지 못했다. 저자 출판 목록이 연결한 보충자료는
읽기 실패/HTTP 403으로 내용을 확인하지 못했다. 센서 응답 정밀도나 FPS는 저장
단위의 증명이 아니다.

로컬 R-bear의 PNG는 1265×705로, 모든 Aid를 1180×720으로 취급해서는 안 된다.
일부 연속 동일 이벤트 행도 확인했지만 전체 중복 빈도는 측정하지 않았다.

## 5. 학습과 분리한 원본 graph 진단

`scripts/audit_raw_event_graph.py`는 명시적으로 지정한 EventHDR H5의 한 readout을
진단한다. 모델, checkpoint, U-Net, CUDA, 학습, SSH 또는 웹 서버를 실행하지 않는다.
R=1의 전체 전달 완료 window를 후보 노드 집합으로 유지하고, 사용자가 지정한
query 노드의 이웃을 production radius index와 독립적인 전 후보 거리 계산으로
대조한다. query 수를 줄이는 것은 **진단 범위 제한**이지 실제 graph의 node/edge cap이 아니다.

결과에는 원본 행 번호, raw/seconds/정규화 시간, 위치, 이웃 ID와 거리,
query별 차수, 실제 검사 범위가 포함된다. 전체 directed edge 수를 계산하지 않았다면
`null`로 기록한다. 한 query/한 window의 통과를 전체 graph, 전체 데이터,
비동기 state update, 모델 정확도 또는 논문 완전 재현으로 보고하지 않는다.
항상 `report_eligible=false`, `paper_exact=false`인 diagnostic 결과이다.

원본 값 상세는 query와 그 이웃에 대해 기록한다. 나머지 노드도 후보 검사에는
포함되지만 JSON 상세에서는 생략된다. 3축 topology만 검사하므로 프레임별 polarity
정규화나 전체 모델 feature의 동등성을 보증하지 않는다. 원본 integer timestamp는
JSON에서 보존하고, 서로 다른 연속 시간이 float64 변환에서 합쳐지면 거부한다.
파일 크기·수정 시각은 metadata identity이며 내용 hash가 아니다. 코드 hash도 명시한
진단/geometry 파일들의 부분 식별자이며 전체 실행 소스의 증명이 아니다.

실행 인자는 원본 파일/프레임, 두 시간 배율, 시간 창, 시간 축척, 반경,
query 선택, CPU thread, RAM budget/reserve와 새 output을 명시한다.
`--help`로 실제 인자를 확인한다. EventAid까지 검증한다고 표현하지 않는다.
출력은 새 파일만 생성하며 기존 실험/원본/output을 덮어쓰지 않는다.
원본 metadata를 확인한 뒤 decode/전체 window 배치 전에 메모리 계획을 검사한다.

현재 로컬 Windows 환경에서는 기존 안전 검사에서 Job Object 소속은 확인되지만
중첩 RAM/CPU 제한을 검증할 수 없어 실제 데이터 실행이 거부되었다.
이 보호를 우회하거나 실제 실행에 resource mock을 적용하지 않는다.
작은 synthetic CPU tests의 통과와 실제 원본 graph 진단 완료를 구분한다.

2026-09-12 최초 진단의 검증 상태 (후속 수정의 테스트 수는 마지막 절과 별개):

- 새 진단의 synthetic CPU tests: 26개 통과 (저장/복원 predecessor 인덱스 포함).
- 기존 radius/input/EventHDR topology/resource 회귀 tests: 193개 통과.
- 변경 Python 파일 Ruff, CLI `--help`, diff whitespace 검사: 통과.
- 실제 원본 graph 실행: Windows 자원 검사에서 거부; 미완료.
- 전체 학습/평가, CUDA 측정, 저자 실험 재현: 실행하지 않음 / 미검증.
- 서버 작업, 기존 실험 종료/삭제/덮어쓰기, commit/push: 수행하지 않음.

## 6. 다음 판단의 조건

1. 실제 데이터의 단위와 해당 실행 config를 확정한다. EventAid 단위 미확인을 숨기지 않는다.
2. 한 raw window의 좌표·경계·이웃 연결부터 검증하고 실제 N/차수를 읽는다.
3. 원 논문 조건을 확정할 추가 자료가 없으면 연구용 복원 adaptation의 거리/샘플링
   선택을 별도로 정당화하고 사용자 승인을 받는다. 임의로 R=10이나 작은 반경을 적용하지 않는다.
4. 이후에만 전체 구조의 correctness와 자원 측정으로 진행한다. 작은 진단 통과는
   기존 full preflight/학습 계약의 면제 사유가 아니다.

최초 9월 12일 변경은 기존 `src`/모델/학습 config를 수정하지 않았다. 진단을 `scripts`에
분리해 `src/**/*.py` 기반 source hash에 포함시키지 않는다. Git revision까지 포함한
실행 식별자는 별개이므로, 실행 중인 서버에 자동 pull/push하거나 checkpoint 호환을
임의 보장하지 않는다. 기존 실험의 중단·삭제·재시작은 수행하지 않았다.

## 7. 2026-09-13: 실제 입력 수정과 설정 기반 진단 전환

### 실제 입력 경로 수정

`stream_input.to_physical_seconds`가 먼저 float64로 변환하면 원본의 서로 다른 큰
정수 시간이 같은 값이 되는 경우를 검사에서 놓쳤다. 원본 dtype/값의 차이를
보존한 상태로 시간 변환을 검사하도록 수정했다. EventHDR의 원본 이벤트 읽기,
프레임 timestamp attribute, 경계 인덱스 검사 및 인덱스 복원 청크 경계도 포함한다.
정상적으로 표현 가능한 입력의 시간 배율·출력 좌표·프레임 인덱스 정책은 유지한다.
원본의 서로 다른 시간이 합쳐지는 경우에만 명시적으로 거부하며, 시간 offset을
추측하거나 이벤트를 합치고 버리는 방법으로 처리하지 않는다.

또한 같은 EventHDR 시퀀스의 H/W가 바뀌면 학습 측 상태 초기화와 scanner의
이전 좌표 재사용이 달라질 수 있었다. 물리 스트리밍 입력에서는 한 H5 안의
모든 프레임 H/W가 같은지 이미지 metadata만으로 확인하고, 다르면 그래프 계산 전에
명시적으로 거부한다. 건너뛰는 프레임도 검사한다. 리사이즈·노드 삭제·조용한 상태
초기화를 적용하지 않는다. 기존 비스트리밍 입력의 가변 크기 지원은 유지한다.
실제 데이터에서 해상도가 변한다고 확인된 것은 아니다.

이 검사는 실제 `src` 경로의 수정이다. **소스 해시가 달라지므로 기존 preflight나
정확 재개 기록을 새 코드의 검증 결과로 재사용하면 안 된다.** 기존 기록과 checkpoint는
보존한다. 이 수정이 관측된 수억 edge의 원인이나 성능 문제를 해결했다는 뜻은 아니다.
EventAid TXT가 float64로 이미 파싱된 뒤에는 원본 문자열 단계의 정밀도 손실을
이 공통 helper만으로 복구/검증할 수 없다. 그 경로의 전체 단위·직렬화 검증 완료를
주장하지 않는다.

### 현재 학습 설정을 그대로 읽는 진단

`scripts/inspect_streaming_graph.py`는 기존 training JSON을 읽어 다음 값을 전달한다.

- `model.graph_radius`, `model.stream_config.window_seconds`, `time_scale_seconds`
- dataset의 event/frame 시간 배율 및 training root
- 명시적으로 선택한 파일과 파일 내부 프레임 번호

지원 범위는 현재 R=1, 3축, 전체 센서, frame_stride=1인 EventHDR 물리 스트리밍이다.
다른 설정이면 R/반경/해상도를 자동 대체하지 않고 이유를 밝히고 거부한다. 분할
manifest 또는 `allowed_files`에 포함된 파일인지 확인한다. 설정과 manifest hash는
진단 전후에 확인하고 결과에 기록한다. 전체 분할 재검증이나 학습 인증은 아니다.

아래는 이 코드가 서버에 배포된 뒤 저장소 루트에서 실행하는 **진단 예시**다.
`26.h5`의 25번 프레임은 검사 대상 선택이지 최종 학습 데이터 축소가 아니다.
반경이나 시간 배율을 이 명령에서 다시 정하지 않는다.

```bash
python -B scripts/inspect_streaming_graph.py --config runs/streaming-v4-34b0487/configs/train.json --source-file 26.h5 --frame-index 25 --cpu-threads 4 --memory-budget-mib 1024 --reserve-memory-mib 1024
```

이 실행은 새 `runs/graph-inspection-*/graph.json`과 `graph.html`을 생성한다. 기존 학습을 실행,
재개 또는 종료하지 않는다. GPU 번호를 설정하거나 SSH/웹 서버를 열지 않는다.
CPU/RAM 검사가 거부하면 이를 우회하지 않는다.

진단 완료 시 터미널의 `GRAPH_INSPECTION_SUMMARY` 아래에 노드 수, 검사한 query별
이웃 수와 일치 여부, 경과 시간을 출력한다. 거대한 이웃 목록을 채팅에 복사할 필요가 없다.
기본 실행의 전체 edge 수 `null`은 미측정이지 0이 아니다. 원본 행/이웃 전체 추적은
같은 새 결과 폴더의 `graph.json`에 보존한다.

### 실제 그래프를 오프라인으로 보기

같은 진단 명령이 이제 전체 윈도우의 **실제 노드 좌표를 빠짐없이** 저장하고
`graph.html`에 데이터와 화면 코드를 모두 내장한다. 터미널 숫자만으로 가상의
그래프를 그리지 않는다. MobaXterm의 파일 패널에서 출력된 `graph.html` 한 파일을
로컬로 가져와 브라우저로 열면 된다. 원본 H5나 전체 실험 폴더를 옮길 필요가 없고,
웹 서버·SSH 터널·네트워크 요청·GPU·모델 추론을 사용하지 않는다.

- XY: 실제 센서 픽셀 위치와 이벤트 분포.
- 시공간 3D: 고정 원점과 학습 설정의 시간 배율을 적용한 실제 거리 좌표의 회전 투영.
- 선택 query: 저장된 모든 incoming 이웃과 이웃→선택 노드 연결선. 세 query 사이를
  바꾸어 보고 노드의 원본 행 번호·픽셀·시간·거리 좌표를 확인할 수 있다.
- 전체 노드 N과 화면에 제공된 노드 수, 선택 query의 차수, 전체 E의 측정 여부를
  구분한다. 모든 N개 점을 그려도 모든 E개 선을 그렸다는 뜻은 아니다.

모든 노드의 좌표를 저장하는 것은 O(N) 출력이며 전체 엣지를 열거하는
`--count-all-nodes`를 자동 실행하지 않는다. 표시 배율·시점 변경은 그래프 반경,
시간 창, sampling이나 모델 설정을 바꾸지 않는다. 좌표·JSON·HTML용 메모리도
명시적 예산에 포함하고 초과하면 노드를 줄이는 대신 거부한다.

이미 저장된 구형 `graph.json`만 변환할 수도 있다. 이 경우 저장된 query+이웃
좌표만 표시하며 **부분 노드 / 전체 N**을 항상 구분한다. 없는 좌표를 복구하거나
추정하지 않으므로 전체 점구름을 보려면 위 진단을 새 폴더로 한 번 실행한다.

```bash
python -B scripts/export_graph_inspection.py --report runs/graph-inspection-9dc1ihfc/graph.json --cpu-threads 4 --memory-budget-mib 1024 --reserve-memory-mib 1024
```

변환기는 새 `runs/graph-view-*/graph.html`만 생성하고 기존 보고서는 보존한다.
저장된 oracle 일치 기록을 보여주는 것이며 원본 데이터 재검증이나 ASGCN 논문과의
완전한 동등성을 인증하지 않는다. 진단용 합성 browser fixture는 `SYNTHETIC`으로
명확히 표시되며 실제 실험 결과로 제공하지 않는다.

시각화 변경의 로컬 검증: 합성 CPU 회귀 테스트 200개, 전체 Python Ruff,
변경한 12개 파일의 privacy scan 및 whitespace 검사가 통과했다. 실제 모델 입력
경로의 기존 합성 회귀도 포함한다. 별도 headless Chrome에서 28,879개 합성 노드의
두 캔버스, query 전환, 이웃 확대, 회전, 좌표 조회, partial/empty,
320/390/1360px와 dark mode를 확인했고 페이지 오류·외부 요청은 0건이었다.
합성 스크린샷도 확인했다. 초기 Windows sandbox 실행의 access-violation 진단과
pytest cache 접근 경고는 최종 실행에서는 발생하지 않았다. 실제 서버 데이터로
HTML을 생성하거나 전체 학습·평가를 실행한 것은 아니다.

### 전체 윈도우 차수의 명시적 검사

`--count-all-nodes`를 추가하면 같은 한 윈도우의 모든 노드 차수를 집계한다.
`full_count`에 N, directed/undirected E, 최소·평균·최대 차수, 고립 노드 수,
집계 시간과 candidate 작업량을 기록한다. 엣지 배열 전체를 보관하지 않으며,
선택된 query의 차수는 독립 거리 oracle 결과와 다시 비교한다. 전체 노드에 대한
독립 oracle 비교가 아니므로 `full_graph_oracle_verified=false`이다.

전체 차수 집계는 실제 E에 비례한 시간이 들 수 있다. 기본 선택-query 검사에
몰래 추가하지 않고 opt-in으로 분리했다. 미측정 전체 E는 계속 null이며, 고립 노드가
없는 것처럼 0을 반환하지 않는다. 빈 그래프를 실제 집계했을 때만 N=E=0이며
차수 최소/평균/최대는 null이다. 어느 경우에도 full-training preflight가 아니다.

### 이번 수정의 검증 상태

최종 수정본에서 합성 CPU 회귀 테스트 488개가 통과했다. 입력/시간/해상도 경계,
radius·topology·preflight, 실제 forward를 사용하는 합성 모델 테스트,
학습 상태와 checkpoint/scan 재개 테스트, 두 진단 실행기의 테스트를 포함한다.
변경 Python 파일의 Ruff와 whitespace 검사, 실행기 `--help`도 통과했다.
이는 실제 데이터 학습·평가나 CUDA 측정이 아니다. 기존 Windows 자원 검사
제약을 우회하지 않았으며, 실제 원본 graph 진단·저자 조건 동등성·관측된 dense
graph의 처리 속도 개선은 여전히 검증되지 않았다. 기존 모델 폭/깊이, 반경,
시간 창, R, 해상도, 데이터 및 학습 epoch/physical batch 설정은 변경하지 않았다.

## 8. 2026-09-15: 검사 도구가 아닌 실제 생성 경로 수정

`stream_geometry.physical_node_positions`와 `prepare_stream_nodes`가 실제 모델의
원본 이벤트 검증·고정 좌표 변환을 담당한다. 모델 `_prepared`와 preflight는 이 경로를
사용하며, raw 진단의 별도 좌표 수식도 제거해 같은 좌표 빌더를 호출한다. 이벤트별
그룹 라벨은 Python 리스트로 확장하지 않고 packed tensor에서 한 번에 만든다.
정상 입력의 기존 거리 수식과 노드 선택은 유지한다. 입력 float64를 조용히 복구한
것처럼 취급하지 않으며, float32 feature overflow는 graph 생성 전에 거부한다.

생성 및 입력 경로에서 확인한 다음 결함을 수정했다.

- 이전 materialized graph가 독립 stream을 가로질러 연결되어 있으면 거부한다.
- implicit graph 내부 stream ID와 바깥 상태 ID가 다르면 cached degree를 재사용하지 않는다.
- 모델이 scanner와 마찬가지로 이전 readout보다 앞에서 시작하는 겹친 frame interval을
  거부한다. EventHDR의 원본 predecessor index에 따른 지연 전달 이벤트는 계속 허용한다.
- factory가 명시적인 `random_crop=False`를 무시하지 않는다. 미지정 때의 split 기본값은
  유지한다. EventHDR의 잘못된 frame_stride는 int 변환/1로 clamp하지 않고 거부한다.
- 설정 기반 진단은 target normalization/channel/tone map 설정도 실제 loader에 전달한다.

현재 전체 센서·R=1·stride=1인 실행의 반경/시간 창/모델/학습 batch는 바꾸지 않았다.
이 수정들이 서버의 수억 edge를 유발했다고 확인된 것은 아니다. 거리·시간 단위를
원문과 동일하다고 새로 인증하지 않으며, 반경을 pixel로 재해석하거나 R을 임의로
높이지 않는다. `r=0.08, s_t=0.05`의 축 절편은 앞 절의 값 그대로다.

### 실제 생성 경로를 사용하는 합성 CPU 회귀

`tests/test_stream_graph_input_path.py`는 작은 synthetic HDF5를 실제 EventHDR 로더와
6층·64채널 모델에 넣는다. 저장/복원 predecessor index, v3/v4, materialized/implicit,
ANN 학습-mode/inference-mode의 16가지 조합에서 각 readout의 **실제 persistent graph**
노드·전체 edge·거리·degree를 원본 전달 prefix에서 독립적으로 계산한 all-pairs oracle과
대조한다. 별도 2가지 v4 storage 테스트는 실제 `_update` 호출을 관찰해 같은 timestamp의
arrival 직후와 expiry/readout 직후도 모두 비교한다. empty interval과 cutoff equality,
아직 전달하지 않은 predecessor 행도 포함한다. 초기화/도착/만료를 테스트용 다른 builder로
대체하지 않는다. 작은 합성 입력은 correctness 검사이며 데이터셋 축소 학습이 아니다.

이 테스트는 원본 서버 데이터의 밀도·시간 단위·GPU 처리량이나 저자 실험 동등성의
검증이 아니다. Windows 실제 데이터 자원 검사 제한은 우회하지 않았다. 기존 실험과
checkpoint는 보존하며, 수정 전 source hash의 preflight를 새 코드 인증으로 재사용하지 않는다.

최종 검증: 위 생성 경로 테스트와 입력/namespace/반경/preflight/모델/학습 상태 및
checkpoint 회귀를 포함한 합성 CPU 테스트 **588 passed (50.95s)**. 변경 Python 파일의
Ruff와 `git diff --check` 통과. 첫 sandbox 실행은 585 passed/1 failed였으며 실패는
기존 provenance 검사의 Git subprocess stdout 읽기에서 Windows 파일/pipe 오류가 난
경우였다(초기 access-violation 진단도 출력됨). 소스나 assertion을 우회하지 않고 로컬
실행 권한을 사용해 동일 회귀에 arrival-prefix 2개를 더한 최종 실행은 588개 모두
통과했고 해당 런타임 진단이 재출력되지 않았다. 실제 데이터/GPU/전체 학습·평가,
commit/push, 서버 작업 중단·재시작은 수행하지 않았다.
