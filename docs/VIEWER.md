# 실제 복원 PNG·시공간 그래프 생성 및 확인

기본 결과 확인 경로는 `scripts/generate_result_visualizations.py`다.
이 Python 코드는 **기존 평가의 실제 GT·복원 PNG와 같은 프레임의 실제 이벤트**를 결합해
PNG·그래프 JSON·오프라인 비교 화면을 한 번에 생성한다. 빈 뷰어나 붙여 넣은 평가표로
실제 결과 생성을 대신하지 않는다.

## 한 번에 생성하기

완료된 평가 폴더와 당시 사용한 config·원본 H5/ZIP이 있는 컴퓨터에서,
저장소 root와 기존 학습·평가 Python 환경을 사용한다. 새 연결·다운로드·학습은 수행하지 않는다.

```bash
python -B scripts/generate_result_visualizations.py --eval-root runs/fast/eval-2960f09 --cpu-threads 4 --memory-budget-mib 1024 --reserve-memory-mib 1024
```

`--cpu-threads 4`는 CPU thread 수이고 GPU 번호가 아니다. **현재 사용이 허용된 CPU 자원**에
맞춰 지정한다. `--memory-budget-mib 1024`는 진단의 추가 작업 메모리 계획 예산,
`--reserve-memory-mib 1024`는 예산 외에 남아 있어야 하는 여유 RAM이다.
실측 가용 RAM·노출된 현재 cgroup과 조상 제한·CPU affinity/quota 및 명시적 scheduler CPU 제한을
대조한다. 필수 측정에 실패하거나 예산·여유분이 부족하면 생성 전에 거부한다.
다른 작업의 점유 변동까지 격리하거나 OOM이 절대 없다고 보장하는 설정은 아니다.

기본 config는 `configs/aid-fast.json`과 `configs/hdr-fast.json`이다.
다른 설정으로 평가했다면 `--aid-config` / `--hdr-config`에 **실제 평가 당시 설정**을 지정한다.
모델·변환·manifest 계약이 저장된 평가와 다르면 임의로 설정을 맞추거나 원본을 바꾸지 않고 실패한다.

출력 이름을 지정하지 않으면 다음과 같은 **새 고유 폴더**를 만든다.

```text
runs/fast/visualization-<UTC>-<id>/
├── index.html                     # 실제 PNG·그래프·평가표를 담은 오프라인 화면
├── generation.json                # 성공 여부, 생성 범위와 자원 계획
├── aid/
│   └── 00000000/                  # 저장된 평가 프레임 index
│       ├── gt.png                 # 원래 평가에서 저장한 GT
│       ├── ann.png                # 원래 ANN 복원 PNG
│       ├── snn_literal_eq15_T4.png # 존재하는 각 SNN 모드도 별도 PNG
│       ├── events-xy.png          # 실제 모델 입력 이벤트의 XY 점유
│       ├── graph-xyt.png          # 실제 시공간 그래프의 표시용 투영
│       └── graph.json             # 모든 입력 노드·전체 통계·표시용 엣지
└── hdr/
    └── ...
```

`--output-dir`로 새 폴더를 지정할 수도 있다. 기존 폴더가 있으면 덮어쓰지 않는다.
실패 시 새 폴더가 이미 만들어졌다면 `generation.failed.json`에 원인을 기록하고,
기존 평가·실험 결과는 그대로 둔다. 다시 시도할 때 기존 결과를 삭제할 필요가 없다.

생성된 `index.html` 한 파일을 PC에서 더블클릭해 나란히·겹쳐 비교, 확대, 평가표,
이벤트 이미지와 회전 가능한 시공간 그래프를 확인한다. 원격에서 생성했다면 생성 후 HTML을
PC에 한 번 복사해야 하지만, 표시를 위한 SSH 터널이나 실행 중인 웹 서버는 필요 없다.
PNG 파일을 별도로 쓰려면 해당 새 출력 폴더의 이미지를 사용한다.
HTML을 보는 동안 인터넷·Python·GPU·H5/ZIP·모델 추론·그래프 재계산은 필요하지 않다.

## 실제 입력과 생성 범위

입력 루트 아래 `aid/ann`, `aid/snn_literal_eq15_T4`, `hdr/ann` 등 각 모드 폴더에
`metrics.json`, `frames.csv`, `predictions/*_gt.png`, `predictions/*_pred.png`가 필요하다.
`benchmark.json`은 있으면 포함한다. 해당 프레임의 원본 EventHDR H5 또는 EventAid-R ZIP은
실제 config의 `dataset.root`에서 읽는다. 체크포인트를 로드하지 않으므로 재학습·재보정도 필요 없다.

- 발견된 모든 활성 모드의 **모든 저장 PNG 프레임**을 처리하고 각 프레임의 실제 그래프를 생성한다.
  조용히 일부 프레임만 선택하거나 그래프 없는 성공 결과로 바꾸지 않는다.
- GT·ANN/SNN 복원 PNG는 원문 바이트 그대로 복사·포함한다. 크기·색상·압축을 다시 만들지 않는다.
  새로 생성하는 PNG는 이벤트 점유와 그래프 표시 자료이며 가짜 모델 예측이 아니다.
- 기존 전체 평가에서 생성한 복원 PNG를 쓰므로 **recurrent context를 보존**한다.
  선택 프레임만 state를 초기화해 재추론한 이미지를 기존 결과처럼 보여주지 않는다.
- CSV 프레임 신원·파일명·프레임 개수·모드 신원·모드 간 GT 동일 바이트와 해상도를 확인한다.
- 저장된 모델·변환·manifest 계약과 원본의 선택 window·sample identity·GT 픽셀을 대조한다.
  생성한 전체 node/edge 수가 각 모드의 CSV 통계와 다르면 실패한다.
- SHA256이 같은 PNG는 HTML 안에서 중복 저장하지 않는다.
- `.failed-*` / `.incomplete-*` 보관 폴더는 경고를 표시하고 제외한다.
- PSNR/SSIM은 저장된 float 지표이며 PNG에서 다시 계산하지 않는다.
- `report_eligible`은 저장된 보고 적격성이지 화질 보증이 아니다.
- `save_predictions=20`이면 각 모드에 저장한 그 20개 프레임 범위다.
  저장되지 않은 전체 평가 프레임을 추가 추론하지 않으며, 기존 전체 품질 평가는 축소되지 않는다.
- 원본 데이터·저장 PNG·필수 계약이 없거나 대응 관계가 다르면 명시적으로 실패한다.
  빈 화면·합성 영상·추정 그래프로 대체하지 않는다.

## 그래프 계산과 정확성의 경계

실제 원본 sample의 기존 전처리·sampling factor·좌표 정규화·반경·차원 수를 그대로 적용한다.
모든 모델 입력 노드를 유지하며 CPU float32의 strict-radius (`distance < radius`) 규칙으로
전체 directed edge 수와 node별 degree를 계산한다. 모델·데이터·추론 경로는 변경하지 않는다.

계산은 RAM 예산에 맞춘 벡터화 tile로 나누며 전체 엣지 목록을 메모리에 보관하지 않는다.
모든 node pair를 확인하는 `O(N²)` 진단이므로 큰 프레임의 처리 시간은 실제 측정이 필요하다.
이 계산은 이미 저장한 PNG 프레임용이지 전체 품질 평가를 다시 수행하는 경로가 아니다.

`--display-edges 5000`은 **그릴 선과 표시 JSON의 엣지 배열만** 제한한다.
모든 노드·전체 edge 통계·전체 degree 계산은 줄이지 않는다. `graph.json`의 `nodes`는
정규화한 `[x, y, t, polarity]`, `edges`는 표시용 `[source, destination]`이다.
브라우저 이웃 조회는 **포함된 표시 엣지**만 대상으로 하며 전체 이웃 목록이라고 주장하지 않는다.

현재 원본의 선택 프레임과 저장된 결과를 대조한 진단이다. 전체 원본 파일의 과거 SHA-256을
다시 검증하거나 과거 GPU graph tensor와 bitwise 일치를 증명한 것은 아니다.
저장된 모델·실행 설정 hash는 확인하지만 전체 protocol·dataset 계약의 hash는 재검증하지 않는다.
CPU 재구성·검증 항목·이 제한을 `graph.json`과 HTML에 기록한다.
생성 자체는 `report_eligible=false`, `model_inference=false`이며 새로운 품질 평가가 아니다.

## 자원과 파일 안전

- 생성은 CPU 전용이다. SSH 접속·웹 서버 실행·GPU 초기화·GPU 번호 선택·모델 추론을 하지 않는다.
- 원본 H5/ZIP은 선택된 프레임을 읽는 용도이며 전체 데이터셋 인덱스를 새로 만들지 않는다.
- report는 작은 버퍼로 순차 파싱하고 선택 PNG의 source identity만 유지한다.
  전체 sampling identity 배열을 한꺼번에 Python 객체로 만들지 않는다.
- PNG는 하나씩 Base64로 출력하고 전체 이미지 문자열을 한꺼번에 쌓지 않는다.
  개별 PNG·JSON·디코딩·원본 읽기에도 임시 메모리가 필요하다.
- RAM 여유와 노출된 CPU 제한을 초기 단계와 주요 할당 전에 반복 확인한다.
  namespace 바깥처럼 OS 조회에 노출되지 않는 상위 자원 제한은 확인 범위 밖이며 기록에 명시한다.
- 기본 한도: 전체 입력 2,048 MiB, 출력 256 MiB, 선택 메타데이터 8 MiB,
  PNG 한 장 32 MiB, 예상 RGBA 디코딩 한 장 128 MiB.
  생성 CLI의 `--max-output-mib`는 HTML 크기만 조정한다. 그래프와 원본 처리에는 별도의 작업 예산을 적용한다.
  이 한도는 상한이며 실제 RAM 예산에 맞춰 파싱·인코딩 허용량이 더 엄격해질 수 있다.
- 브라우저 직접 입력은 PNG/JSON 파일당 32 MiB, PNG 예상 RGBA 128 MiB다.
- 한도 초과는 **전체 생성 실패**다. 모델·노드·전체 엣지·프레임·해상도를 자동 축소하지 않는다.
  예산을 올리기 전 현재 허용된 RAM·디스크·브라우저 부담을 확인한다.
- 점검은 시점별 snapshot과 알고리즘 메모리 계획이지 peak RSS 강제 제한이나 완전한 자원 격리가 아니다.
  다른 실험이 이후 RAM을 소비할 수 있으며, CPU 허용량 확인이 현재 idle/exclusive CPU 보증도 아니다.
- 새 폴더에만 결과를 만들고 HTML은 새 임시 파일을 완성한 뒤 기존 출력이 없는 경우에만 게시한다.
  실패 시 현재 HTML 생성의 임시 파일만 제거하며 원본·기존 결과와 다른 실험은 그대로 둔다.
- HTML은 CSP `connect-src 'none'` 등으로 네트워크를 차단한다.
  이미지·평가 메타데이터와 원본 선택 정보가 포함되므로 공유 범위를 확인한다.

## 보조 경로: 이미 있는 PNG만 내보내기

`scripts/export_results_html.py`는 **기존 PNG·평가표만 묶는** 별도 stdlib 도구다.
실제 이벤트 그래프를 생성하려는 경우에는 위 `generate_result_visualizations.py`를 사용한다.

```bash
python -B scripts/export_results_html.py --eval-root runs/fast/eval-2960f09 --output runs/fast/asgcn-offline-results.html
```

이 보조 도구에는 PyTorch·NumPy·Pillow나 GPU가 필요 없으며, 저장 그래프가 없으면 그래프 미포함이다.
기존 그래프를 `--graph-json saved-graphs.json`으로 명시할 수 있다.
wrapper 형식은 `schema: asgcn_offline_graphs_v1`, `graphs: [...]`이며 각 항목은
`dataset`(aid/hdr), `index`, `sample_id`, 저장 배열인 `graph`를 갖는다.
해당 식별자가 CSV와 다르면 거부하지만 원본 topology를 다시 검증하지는 않는다.
보조 도구의 `--max-*-mib`로 입력/출력 한도를 명시하며 기존 출력은 덮어쓰지 않는다.

HTML에서 PC의 PNG 두 장이나 JSON을 직접 선택하는 기능도 있지만, 직접 선택한 자료는 평가표의
프레임과 연결 미검증으로 표시한다. 평가표만 있는 파일은 실제 PNG·그래프 확인을 대신하지 않는다.

기존 `scripts/view_results.py`는 과거 서버형 진단 도구로 남아 있다.
그래프 요청 때 CPU에서 원본과 전체 그래프를 다시 만들며 공유 서버 RAM/cgroup 사전 검사가
없으므로 기본 결과 확인 경로로 사용하지 않는다. 새 생성 CLI는 웹 서버를 시작하지 않는다.

## 검증 범위

생성기·source reader·그래프·자원 점검의 단위/통합 테스트와 `file://` 브라우저 smoke test는
**합성 테스트 자료**로 수행한다. 작은 CPU 테스트를 실제 사용자 데이터 생성 완료,
모델 성능 검증, 전체 학습·평가 완료 또는 서버 사고 원인 확인으로 표현하지 않는다.
실제 서버의 H5/ZIP·PNG와 현재 할당에서 실행한 자원·처리 시간 검증은 별도다.

`tests/test_result_visualization.py`의 합성 CPU 통합 테스트는 임시 폴더에 실제 생성 경로를
통과한 `generated` 폴더를 만든다. Playwright가 있는 Node에서
`tests/result_visualization_browser_smoke.cjs`에 그 폴더 경로 하나를 전달하면
GT·모드별 복원·진단 PNG와 그래프 연결을 오프라인으로 검사한다.
`ASGCN_PLAYWRIGHT_MODULE`로 기존 Playwright 경로, `ASGCN_BROWSER_CHANNEL=chrome` 등으로
기존 브라우저를 선택할 수 있다. 새 설치나 HTTP 서버가 필요하지 않다.
