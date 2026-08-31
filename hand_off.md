# ASGCN-U-Net 프로젝트 인계서

이 문서는 연구자가 현재 저장소를 교차검증하고 Linux GPU 서버에서 전체 실험을
이어가기 위한 기준 문서다. 코드와 config가 최종 진실이며, 아래 내용은 2026-09-01의 현재
구현과 일치하도록 다시 대조했다.

## 0. 검증 기록과 배포 판정 기준

### GPU 처리량 병목 후속 수정

사용자 제공 B4 서버 로그는 실제 CUDA 학습 중 약 1,216 MiB process GPU memory를 보였다.
첫 10 warmup 이후 50개 성공 update 측정 창에서 CUDA stream 경과시간 평균은 graph 약 22.16 ms,
encoder 47.59 ms, decoder 6.14 ms, backward 50.19 ms였다. 이는 VRAM 부족의 증거가 아니며,
전체 GPU utilization이나 배치 크기의 최적성을 입증하지도 않는다. host/CUDA 시간을 합산하지 않는다.
`gradient_check`는 update당 두 번 측정하므로 scope 평균을 update 평균으로 오인하지 않는다.

후속 변경은 그래프의 조회/크기 전송 묶음화, 작은 hash 상수의 device/stream별 재사용,
완전한 temporal context 배치의 indexing 생략과 선택형 spline 융합 backend다.
`torch`는 기준 구현이며 `torch_fused`와 CUDA-only `triton`은 동일 입력 수치·속도 비교 대상이다.
기본 배치/모델/데이터/40 epochs를 임의로 줄이지 않고 finite 검사·AMP rollback·exact-resume 보호를 유지한다.

`scripts/bench.py`는 같은 실제 EventHDR 연속 프레임 집합으로 backend/배치 크기를 비교하고,
새 subprocess마다 모델·optimizer·scaler를 초기화한다. 실제 학습 함수와 시퀀스 context를 사용하며
데이터 대체, checkpoint 생성/수정 또는 기존 학습의 재개는 하지 않는다. 새 소스의 GPU 커널 수치·
속도·전체 학습은 로컬 CPU 테스트로 검증됐다고 주장하지 않는다. 상세 범위와 실행은
[PERF.md](docs/PERF.md#동일-실데이터-gpu-비교)를 따른다.

이번 변경의 최종 전체 Windows CPU 검사는 **1,286 passed, 82 skipped, 1 warning (73.36초)**이며
종료코드 0이다. skip에는 CUDA 실행 불가 항목이 포함되고, 기존 test-only quantized buffer의
deprecated API 경고 1개가 있었다. 해당 전체 실행에서 native access-violation 출력은 없었다.
Ruff·diff 검사와 tracked text/도달 가능한 Git history의 개인정보 검사를 통과했다.
동일 프레임/상태 보존, gradient 방향 오류 거부, 실패/OOM의 성공 표시 방지, 익명화 전 원본 trace의
비공개 임시 저장·실패 시 정리를 회귀검사한다. 아래 1167/41 등은 이전 단계의 기록이다.

### 독립 시퀀스 미니배치 경로

단일 프레임 update 반복을 줄이기 위해 `configs/batch.json`과 training protocol v6를 추가했다.
기존 `configs/train.json`의 v5 경로는 보존한다. B=4는 프레임 간 edge가 없는 disjoint graph와
vectorized recurrent U-Net의 실제 배치 처리이며, `forward_sample` 네 번 호출이나 gradient
accumulation이 아니다. 시퀀스당 시간 순서, 전체 프레임, 40 epochs, 모델 크기는 유지한다.
Pooled-node BN과 배치 평균 loss/optimizer update는 학습 프로토콜 변경이므로 별도 run으로 시작한다.
기준선 checkpoint의 exact-resume 보호를 우회하지 않는다.

`batching.py`는 최대 B개 활성 시퀀스와 해상도별 배치를 스케줄링한다. `training.py`는 시퀀스별
detach/독립 저장소 context와 temporal loss를 관리하며, 성공한 update만 commit하고 끝난 시퀀스를
제거한다. AMP 실패는 배치 전체를 재시도한다. `preflight.py`는 실제 최대 B 구성, 밀집/첫/빈/희소
배치의 공유 학습 경로를 CUDA에서 검사해야 gate를 발급한다. 이전 topology 수치 재사용과 이전
GPU 측정 통과는 구분하며 GPU 검사는 항상 다시 수행한다.

`timing.py`는 실제 학습 10 warmup 뒤 50 step의 host/CUDA 단계 시간을 수집한다. `history.json`은
validation을 제외한 epoch wall time, 처리 frame 수, optimizer update 수와 frame/s를 기록한다.
그래프는 여전히 매 프레임·매 epoch 생성하며 전체 graph cache는 구현하지 않았다. Batch 4의
동일 입력 대비 MIG 가속률·전체 peak·수렴 결과는 아직 없다. 위 초기 서버 측정과 구분한다.
구현 계약·한계·서버 명령은 [TRAIN.md](docs/TRAIN.md)를
기준으로 한다. 아래 997/40 기록은 이전 AMP 수정 시점이며 이번 배치 경로의 검증 수치가 아니다.

이번 배치·artifact 보호 수정의 전체 Windows CPU 회귀검사는 **1167 passed, 41 skipped,
1 warning (75.08초)**다. CUDA 하드웨어·Linux shell·Windows symlink 제약 검사는 skip이며,
warning은 test-only quantized buffer의 PyTorch deprecated API 경고다. 일부 테스트 실행에서는
Windows native access-violation 진단도 출력됐으나 해당 실행은 pytest 결과까지 반환됐다.
최종 전체 재실행은 종료코드 0이며 위 1 warning 외 native 진단 출력은 없었다. Ruff와 diff 검사는
통과했다. 이 기록은 Linux wrapper 실제 실행, 실제 MIG throughput/VRAM 또는 전체 학습 완료의 증거가 아니다.

추가 artifact 회귀검사는 잘못된 fresh/resume 시도에서 기존 config·gate·hash cache·checkpoint
바이트 보존을 확인한다. CLI는 gate를 먼저 저장하지 않고 engine의 모든 재개 검증 후에만
run metadata를 게시한다. 기존 99,088-frame 학습에서 제공된 2시간 이상 소요/592 MiB 화면은
문제 제기 자료이며, 새 배치 경로의 성능 측정치로 사용하지 않는다.

2026-08-31 서버 진단에서 첫 EventHDR 프레임의 event 수가 0이고, 기본 FP16 scale 65,536에서만
`decoder.enc1.body.0.bias` gradient가 비유한 값이 되는 것을 확인했다. 같은 입력의 FP16 scale 1과
FP32는 finite loss/gradient였고 norm은 약 16.83이었다. 이는 사용자 제공 단일 프레임 진단 결과이며
수정된 코드의 GPU 전체 학습 성공 기록이 아니다.

기존 단일 프레임 경로는 training protocol **v5**, `same_sample_backoff_v1`로 AMP overflow를 처리한다. 같은
프레임을 최대 16회 재시도하며 실패한 시도의 BN/mutable buffer와 Python/NumPy/Torch/CUDA RNG를
복원한다. 실패 시 optimizer update나 프레임 건너뛰기는 없고, recurrent/temporal state는 성공한
시도만 반영한다. finite gradient 확인 후 GC·clip·update를 수행하며 원래 backend 오류를 숨기지 않는다.
epoch history의 `amp`에 retries/retried_samples/scale을 남긴다. 실제 NaN loss·AMP-off 잘못된 gradient·
지속 overflow는 여전히 hard failure다.

새 preflight는 CUDA topology와 event-only HDF5 경로, 기본 CPU helper thread 4개, 구간별 atomic
scan journal을 사용한다. 최종 report v2는 첫 프레임·첫 zero-event·최소 양수 node 입력의 초기 모델
검증과 상위 3개 밀집 표본 학습 검증을 분리 기록한다. 검증된 이전 전수 기록은 명시적으로 이관하되
원래 CPU/CUDA 출처를 유지하고 GPU 측정은 전부 새로 수행한다. metadata-only 학습 실패는 명시적인
재시작 옵션으로 directory를 보존하며 checkpoint/history/unknown file은 자동으로 옮기지 않는다.
명령과 재사용 조건은 [서버 복구 절차](docs/SERVER.md#amp-첫-step-오류와-이전-profile-이관)를 따른다.

이 AMP/scan/복구 수정의 Windows CPU 통합 회귀검사는 **997 passed, 40 skipped**다.
Ruff와 Git diff whitespace 검사도 통과했다. skip은 CUDA 하드웨어, Linux 전용 실행,
Windows symlink 권한 제약에 따른 것으로 해당 검사를 통과했다는 의미가 아니다. 이번 로컬 환경에는
Bash가 없어 수정된 shell wrapper의 실제 Linux 실행은 아직 검증하지 못했다. 별도의 CUDA full-model
zero-event 회귀검사를 추가했지만 로컬에서는 skip됐으며, 서버 실데이터로 수정본을 검증한 기록은 없다.

2026-08-31 연산 최적화에서는 radius candidate 확장/compaction, Spline custom autograd의 저장
tensor 메모리와 SNN 고정 첫 layer 계산을 개선했다. 기존 모델·config·sampling·전체 실험 범위는
변경하지 않았다. 전체 CPU 회귀검사는 **860 passed, 35 skipped**이며 수치·시간·저장 메모리 검증과
GPU 미검증 범위는 [PERF.md](docs/PERF.md)를 따른다. C++/CUDA extension을 새로 구현한 것은 아니다.
새 source contract가 적용되므로 이전 GPU 측정/checkpoint의 exact-resume 보호를 우회하지 않는다.
위 860/35는 앞선 연산 최적화 시점의 기록이며 후속 AMP/scan 수정 전체의 검증 수치가 아니다.

이 파일과 `README.md`, `code_summary.md`는 source snapshot의 설명이며 원격 배포 성공 확인서가 아니다.
설치와 실험 절차는 README를 기준으로 하며, 배포 검증은 해당 commit의 CI와 아래 release gate로
확인한다. 문서와 기본 산출물은 서버 계정명·hostname·사용자별 absolute home path를 기록하지 않는다.
저장소 내부에는 사용자별 식별자를 회귀 fixture로도 보존하지 않는다. 대신
`scripts/scan_private_text.py`와 `tests/test_repo_hygiene.py`가 다음을 검증한다.

- 모든 Git tracked text와 생성된 `code_summary.md`의 generic user-home/labelled identity 검사
- 저장소 밖 로컬 denylist 또는 로컬 환경변수 `PRIVATE_MARKERS_B64`로만 주입한 실제 marker의 current
  tree 및 전체 local-ref reachable history 검사; 로컬 `--require-external-patterns`에서 빈 denylist 거부
- 실제 marker를 받지 않는 CI의 generic current-tree/history 검사와 clean provenance gate;
  history 검사는 로컬/CI 모두 shallow clone 거부
- Python 문자열 결합·constant f-string·Base64 표현을 복원해 숨은 marker 검사
- shebang entrypoint의 Git 실행 권한을 Windows에서도 검사해 Linux checkout 권한 누락 방지
- 설치 명령이 README의 HTTPS clone·Conda 단일 환경 경로로 통합됐는지 문서 검토

환경 전환 기준 commit `1afb40f`는 2026-08-30 [CI 7개 job](https://github.com/costunder/asgcn-unet/actions/runs/33308213057)을
통과했다. Linux Conda 고정 profile의 실제 설치·정확한 버전 검증과 pytest **598 passed, 4 warnings**를
확인했다. 같은 전환 작업의 Windows CPU pytest는 **571 passed, 27 skipped**로 종료했다.
skip은 Linux 전용 설치 shell test 22건과 symlink 권한 관련 5건이다. 성공 종료 중에도 Windows native
access-violation 진단이 출력되어 무경고 검증으로 간주하지 않는다. shell entrypoint 16개는 MSYS Bash에서
각각 구문 검사했다. 해당 SHA의 Linux Conda 실제 설치·고정 profile·전체 pytest와 별도 cross-platform
CI 결과로 배포 판정을 확인하며, 이 기록은 실제 CUDA 본실험 완료를 뜻하지 않는다. 후속 변경은 해당
commit SHA의 로컬 실제-marker release gate 기록과 GitHub Actions 필수 gate 통과를 별도로 확인한다.
실제 marker는 GitHub secret·변수·
workflow·log·artifact로 전송하지 않는다. CI의 generic 검사만으로 로컬 실제-marker 검사를 대신하거나,
로컬 테스트 결과와 이 문서만으로 원격 배포 완료를 간주하지 않는다.

유지관리자와 외부 검토자는 특히 다음을 확인해야 한다. 확인된 배포의 실험 사용자가 전체 회귀검사를
다시 수행해야 한다는 뜻은 아니다.

1. 전체 tracked text와 `code_summary.md`뿐 아니라 모든 local ref의 history를 저장소 밖 로컬
   실제-marker denylist로 검사하고, marker를 GitHub에 전송하지 않았는가?
2. README가 필요한 서버 환경과 Conda 설치·데이터 준비·학습·평가를 일관되게 안내하는가?
3. clone 이후 명령이 저장소 root 기준으로 실행 가능한가?
4. 저장소 관리와 실험 실행 절차가 분리되어 있는가?
5. 기존 Conda 환경과 clone 폴더를 자동 삭제하거나 덮어쓰지 않는가?
6. 변경 사항이 model/data/experiment protocol에 미치는 영향을 검증했는가?
7. 실제 GPU의 full-topology/densest-step profile이 checkpoint의 verified preflight gate로 이어지는가?
8. ANN/SNN 평가 artifact가 sealed lineage와 현재 data/source/runtime/precision protocol을 갖는가?

## 1. 한 줄 결론과 주장 범위

이 프로젝트는 EventHDR로 event-to-frame ANN을 학습하고, ASGCN 논문의 공개 graph/SNN 수식을
적용해 변환한 뒤 EventHDR 공식 eval과 EventAid-R에서 평가하는 연구 코드다. graph encoder 뒤에는
과제용 residual U-Net과 analog ConvGRU가 붙는다.

저자 공식 저장소나 공식 checkpoint를 실행한 것이 아니며 원 논문의 classification pipeline도
재현하지 않는다. 사용할 수 있는 표현은 `ASGCN paper-core 기반 event-to-frame 복원 적응` 또는
`공개 수식 기반 graph/SNN core 구현`이다. 다음 주장은 금지한다.

- 저자 공식 ASGCN 코드 또는 공식 checkpoint
- 공식 ASGCN 완전 재현
- 원 논문의 classification 성능 재현
- 완전한 spiking network
- FPGA/ASIC latency·전력·에너지 실측 또는 반도체 통합 구현 완료

repository의 HTTPS clone 주소는 다음과 같다.

```text
https://github.com/costunder/asgcn-unet.git
```

## 2. 프로젝트 목표와 전체 파이프라인

목표는 DVS event interval을 저지연 graph 연산으로 처리해 luminance frame을 복원하고, ANN과
ANN→SNN graph encoder의 품질·지연·발화율을 같은 조건에서 비교하는 것이다.

```text
EventHDR H5 / EventAid-R ZIP
  -> event interval [N, x, y, t, p] + target luminance frame
  -> spatial crop (기본 full frame)
  -> exact-size max_events cap (기본 8,192)
  -> ASGCN 고정 event sampling factor R (기본 1)
  -> normalized node feature [x,y,t,p]
  -> strict d(i,j)<D undirected radius graph
  -> scalar u=d/D degree-1 open B-spline graph encoder
       ANN: affine -> BatchNorm -> ReLU
       SNN: BN fold -> Eq.(6) normalization -> explicit IF timesteps
  -> graph feature rasterization
  -> residual U-Net + bottleneck analog ConvGRU
  -> sigmoid [0,1] luminance frame
  -> quality / temporal / latency / graph / firing-rate metrics
```

SNN으로 바뀌는 부분은 graph encoder뿐이다. rasterization, U-Net, ConvGRU와 output head는 analog다.
세부 수식 대응과 공개 논문 대비 가정은 `docs/ASGCN.md`가 기준이다.

## 3. 데이터셋의 정확한 역할과 용량

| 데이터 | 공식 배포 구조 | 표시 용량 | 현재 프로토콜의 역할 |
|---|---|---:|---|
| EventHDR train | `1.h5`–`51.h5`, 51개 | EventHDR 전체 약 25.72 GB | ANN gradient 학습, ANN→SNN calibration |
| EventHDR eval | `1.h5`–`19.h5`, 19개 | 위 전체에 포함 | 마지막 epoch 1회 내부 검증, ANN/SNN 공식 eval 내부 결과 |
| EventAid-R | `R-*.zip`, 14개 | 약 24.68024 GB | 학습·calibration 뒤 외부 일반화 평가 |

두 데이터셋의 공식 배포 표시 용량 합은 약 50.40 GB로 100 GB 미만이다. EventAid-R은 ZIP을
추출하지 않고 직접 읽어 중복 저장을 피한다. EventHDR 기본 경로는 서버에 H5를 직접 다운로드한다.
선택적으로 기존 archive를 import할 때는 archive와 배치된 H5가 일시적으로 함께 존재할 수 있으므로
서버의 실제 여유 공간은 별도로 확인한다.

### 3.1 EventHDR 획득과 배치

공식 배포는 [EventHDR 저장소](https://github.com/yunhao-zou/EventHDR)의 OneDrive 링크다. 기본 경로는
**Linux 서버 직접 다운로드**이며 PC 경유·SFTP 전송, 사용자 로그인·브라우저·쿠키를 요구하지 않는다.
Python 표준 라이브러리 HTTP로 익명 접근해 train 51개와 eval 19개의 H5를
`data/EventHDR/{train,eval}`에 저장한다.

```bash
bash scripts/get_hdr.sh --download

# 한 split만 필요할 때; 전체 실행 전에는 두 split 모두 준비
bash scripts/get_hdr.sh --download --split train
bash scripts/get_hdr.sh --download --split eval
```

같은 명령을 다시 실행하면 `.part` 파일을 이어받으며 일시적 요청 실패를 재시도하고 만료된 링크를
갱신한다. 정확한 파일 집합, API metadata의 byte size·SHA-256과 HDF5 signature를 확인한다. 익명
token과 임시 서명 다운로드 URL은 메모리에서만 사용하고 파일·로그에 남기지 않는다. API가 제공하는
현재 파일 hash와 대조하는 것이며, 저자가 별도로 서명·배포한 checksum release를 대신하지는 않는다.

2026-08-30 확인한 경로는 **문서화되지 않은 OneDrive 익명 호환 endpoint**다. Microsoft의 안정적
API 계약을 보장하지 않으며 공유 상태나 서비스 변경에 따라 실패할 수 있다. 실접속 검증은 70개 H5
metadata와 train/eval 각 H5의 첫 8 bytes, 새 익명 token의 공유 접근 갱신에 한정된다. 범위 요청은 HTTP 206과 올바른
Content-Range·HDF5 signature를 반환했으나 전체 약 25.72GB 다운로드·전체 decode·GPU 본실험은
수행하지 않았다.

이미 서버에 있는 ZIP, 압축 해제 directory 또는 shared filesystem은 선택적으로 안전하게 import할
수 있다. 아래 경로는 직접 다운로드의 필수 선행 단계가 아니다.

```bash
# 이미 가진 train/eval 포함 ZIP을 직접 읽어 data/EventHDR로 복사
bash scripts/get_hdr.sh --archive data/_archives/EventHDR.zip

# 이미 풀어 둔 EventHDR/{train,eval} 또는 {train,eval} root에서 복사
bash scripts/get_hdr.sh --source /absolute/path/EventHDR

# shared storage를 복사하지 않고 split directory symlink로 연결
bash scripts/get_hdr.sh --source /shared/datasets/EventHDR --link

# train/eval ZIP을 따로 받았을 때
bash scripts/get_hdr.sh --archive data/_archives/train.zip --split train
bash scripts/get_hdr.sh --archive data/_archives/eval.zip --split eval

# 현재 목적지 재검사
bash scripts/get_hdr.sh --check
```

`scripts/get_hdr.py`의 import/check는 train의 정확한 51개 이름, eval의 정확한 19개 이름,
missing/extra/nested H5, archive 중복·경로 이탈, HDF5 magic과 선택 데이터 100 GB 미만을 검사한다.
복사는 `.part` 임시 파일 뒤 atomic replace로 완료하며 기존의 다른 크기 파일은 덮어쓰지 않는다.
이 로컬 import/check는 다운로드 모드의 원격 SHA-256 대조와 구분되며 배포자 cryptographic
checksum 검증을 대신하지 않는다.

### 3.2 EventAid-R 획득과 배치

`manifests/eventaid_r.json`에는 공식 benchmark page가 연결한 14개 Dropbox URL, scene 이름과 표시
용량이 고정돼 있다. Linux에서는 인자 없이 실행하면 전체 14개를 받는다.

```bash
bash scripts/get_aid.sh
```

Linux downloader는 재개 가능한 `curl`, retry와 ZIP container 검사를 사용한다. 공식 checksum이
없으므로 최종 내용 검사는 뒤의 `inspect --validate-all` 단계에서 모든 event block과 target을
decode하는 방식이다.

### 3.3 loader 의미론

EventHDR loader는 H5의 `events/{xs,ys,ts,ps}`와 이미지 `timestamp`를 검사한다. `event_idx`가 있으면
유효성을 검사하고 그대로 사용한다. 없으면 `max(searchsorted(events/ts, timestamp, side="left") - 1, 0)`으로
누락 인덱스만 읽기 시점에 복원한다. 이는 참조 packager의 predecessor 호환 정책이며 표준 half-open
timestamp 구간과 다르다. 원본 H5는 수정하지 않는다. 전체 timestamp를 bounded block으로 검증하며
NaN/Inf·비단조 timestamp·완전히 분리된 이미지/event 시간 범위는 복원 실패로 처리한다.
정의와 근거는 [실험 프로토콜](docs/EXPERIMENT.md#eventhdr-이벤트-인덱스)에 있다.
`inspect.event_indexing`은 파일별 저장·복원 이미지 수를, sample metadata의 `event_idx_source`는
`stored`/`timestamp_predecessor_v1`을 기록한다. 기존 index identity에 `start_idx/end_idx`가 포함되므로
선택된 경계도 checkpoint의 data protocol에 결합된다.
timestamp·event boundary가 단조롭고 좌표·polarity가 유효한지 확인한다. `frame_stride=1`에서 모든
target interval을 유지하며 event가 0개인 interval도 삭제하지 않는다. 빈 interval은 zero-node graph와
zero raster를 거쳐 recurrent decoder로 전달된다. `frame_stride>1`이면 건너뛴 interval의 event를
다음 선택 target까지 합치지만 기본값은 1이다. image key는 `image<정수>` 형식과 numeric suffix
uniqueness를 강제하고 숫자순으로 읽으므로 `image10`이 `image2`보다 앞서는 문자열 정렬 오류를 허용하지
않는다.

EventAid-R loader는 일반 ZIP 안의 `event/i.txt`, `gt/j_img.png` 또는 `.jpg`/`.jpeg`,
`timestamps.txt`, `shape.txt`를 직접 읽는다. 영상 재압축이나 확장자 변경을 요구하지 않는다.
`inspect.scenes`는 전체 GT의 `target_formats`와 `layout`을 기록한다.
config의 `target_offset=1`은 event interval `i`를 다음 GT `i+1`과 짝짓는 구현 가정이다.
offset은 bool이나 실수를 정수로 조용히 변환하지 않고 정확한 정수만 받는다.
연속 ID, timestamp coverage, shape, 좌표와 polarity뿐 아니라 numeric event/GT ID 중복,
case-insensitive ZIP member 중복, metadata member 중복, 중복 timestamp와 잘못된 shape token도
거부한다. 이 pairing을 저자 공식 코드로 확인한 것은 아니므로 보고서에서 가정으로 표시하고 offset이
다른 실험과 결과를 섞지 않는다. full inspect는 각 event block의 원 timestamp min/max와 interval
`t0/t1`, span ratio, offset 및 범위 이탈 수를 집계한다. 공식 14 ZIP에서 timestamp basis와 단위가
확인되기 전까지 이 집계는 diagnostic이며 strict rejection 조건이 아니다.

공식 `R-traffic`은 `event_upload/`, `gt_upload/`, `timestamps_upload.txt`, `parts.txt`를
사용한다. parts의 네 inclusive 구간 `1–1297`, `1967–2913`, `3201–5169`, `5377–8511`을
event/GT ID 집합과 정확히 대조한다. timestamp는 원래 ID가 아니라 업로드된 ID의 숫자 정렬
순번으로 연결한다. 구간 안에 다음 timestamp와 offset target이 모두 존재하는 event만 쌍을
만들어, 기본 offset 1에서는 7,348개 중 끝 경계 4개를 제외한 7,344개를 평가한다. metadata에
`part_index`, `sequence_id`를 추가해 state·temporal metric을 경계에서 초기화하지만, 원래 ID와
ZIP 단위 `scene`을 보존해 14-scene macro 가중치를 바꾸지 않는다. part/timestamp도 sample
identity hash에 포함한다. 선언되지 않은 누락 파일을 추정해 건너뛰거나 일반·upload 구조를 섞지
않는다. 일반 구조의 `R-building`은 timestamp가 GT보다 한 행 많으므로 coverage만 검사하고,
upload 구조는 정확한 행 수를 요구한다.

두 dataset 모두 `target_normalization.mode=integer_dtype_max`를 명시해 정수 dtype maximum으로
`[0,1]` luminance를 만든 뒤 기본 config에서 `log1p(5000*x)/log1p(5000)`를 적용한다. float target은
`known_scale` 또는 `already_normalized`를 명시해야 하고 NaN/Inf와 `[0,1]` 범위 위반은 거부한다.
`percentile_debug_only`는 `debug_only=true`인 비보고용 진단에서만 허용된다. EventAid-R의 8-bit 영상에
같은 log mapping을 쓰는 것은 출력 수치 domain을 맞추기 위한 cross-domain 선택이지 두 센서의
radiometric response가 같다는 뜻이 아니다.

## 4. EventHDR manifest의 진실

`manifests/eventhdr_split.json`은 다음 의미를 갖는다.

```json
{
  "status": "final",
  "split_schema": "official_separate_roots_v1",
  "group_semantics": "h5_sequence_file_not_physical_scene",
  "train_files": ["1.h5", "...", "51.h5"],
  "val_files": ["1.h5", "...", "19.h5"]
}
```

여기서 `final`은 공식 배포 file set과 separate root manifest가 확정됐다는 뜻이다. H5 번호가 물리
scene ID라는 뜻이 아니며, 공개 자료에서 51개 train H5와 물리 촬영 scene 사이의 대응표는 확인되지
않았다. 따라서 이 split으로 physical-scene-disjoint 일반화를 주장할 수 없다.

train과 eval은 서로 다른 directory라 `1.h5` 같은 basename이 겹친다. factory는
`official-train-h5::1.h5`와 `official-eval-h5::1.h5`처럼 split-local sequence group ID를 자동으로
만들어 recurrent state와 macro metric을 분리한다. 공식 schema에 임의의 physical-scene field를
추가하면 거부한다. root의 missing/undeclared H5도 학습 전에 거부한다.

`configs/train.json`의 `validate_every=null`은 EventHDR 공식 eval을 매 epoch 보지 않고 마지막
40번째 epoch에서 단 한 번만 실행한다. 그 하나의 candidate를 `best.pt`로 export하므로 epoch 간
selection은 하지 않는다. 그래도 같은 eval에서 산출한 수치는 독립 test나 physical-scene test가
아니며 `EventHDR official eval internal result`로만 보고한다. 이 결과를 보고 hyperparameter를
바꾸면 이후 run에서는 사실상 개발 정보로 사용한 것이므로 독립성 주장을 더 할 수 없다.

EventAid-R은 training과 calibration에 사용하지 않는다. 외부 결과를 본 뒤 radius, cap, threshold,
tone mapping 또는 checkpoint를 바꾸면 기존 EventAid-R 결과를 잠긴 외부 일반화 평가로 부를 수 없다.

## 5. 기본 config와 학습 규칙

`configs/train.json`의 핵심값은 다음과 같다.

| 영역 | 기본값 |
|---|---|
| seed / device | `2026` / `auto` |
| data | train 51 H5, eval 19 H5, full frame, stride 1, log tone map |
| event cap | crop 뒤 정확히 최대 8,192개 |
| graph | `x,y,t`, radius 0.08, chunk 512, directed edge guard 2,000,000 |
| spline encoder | hidden 64, 6 layers, open degree 1, K=5, root weight |
| decoder | raster downsample 4, base 48, output 1, ConvGRU on |
| training | 40 epoch, batch 1, chronological, workers 4, persistent/prefetch 2 |
| optimizer | Adam + gradient centralization, lr `1e-3`, weight decay `5e-3` |
| scheduler | MultiStepLR epoch 20/30, gamma 0.1 |
| stability | CUDA AMP same-sample backoff 최대 16회, L2 grad clip 1.0, 실제/지속 non-finite fail-fast |
| preflight | train 전체 topology scan, edge 상위 10개 기록, 상위 3개 CUDA 학습 step |
| validation | 마지막 epoch 1회, 전체 19 H5, recurrent context policy 기록 |

event cap은 `N>8192`일 때 `np.linspace(0,N-1,8192)`로 시간축 전체에서 정확히 8,192개를 선택한다.
8,193개 입력이 절반으로 급락하는 ceil-stride 경계 문제를 피하며 양 끝 event를 포함한다. cap이
필요 없으면 원본을 그대로 쓴다. metadata의 `raw_event_count`, `cropped_event_count`,
`retained_event_count`, `dataset_sampling_ratio`와 model diagnostics의 `event_sampling_factor`,
`effective_sampling_ratio`가 provenance와 CSV에 남는다.

batch size는 recurrent chronology 때문에 1이고 shuffle하지 않는다. H5/ZIP group, sensor shape,
sequence index가 정확히 이어질 때만 ConvGRU state와 temporal reference를 유지한다. 불연속에서는
초기화한다. state와 이전 prediction은 매 frame detach하므로 full-sequence BPTT가 아니다.

### 5.1 loss

기본 loss는 다음 합이다.

```text
L = 1.0 * Charbonnier(epsilon=1e-3)
  + 0.2 * (1 - Gaussian SSIM)
  + 0.1 * spatial gradient L1
  + 0.2 * frame-delta temporal L1
```

temporal term은 같은 group·shape에서 sequence index가 1 증가할 때만
`L1((pred_t-pred_t-1),(gt_t-gt_t-1))`로 계산한다. optical-flow warp metric이 아니며 이전
prediction은 detach돼 있다. SSIM은 `[0,1]`, Gaussian 11×11, sigma 1.5, valid convolution이고 작은
영상에는 들어맞는 가장 큰 홀수 window를 쓴다.

loss component는 GPU tensor로 유지하고 total과 component를 한 벡터로 묶어 CPU에 한 번만
전송한다. 이 벡터를 finite 검사, 누적, logging에 함께 사용해 frame마다 여러 `.item()` 동기화가
발생하지 않게 했다. gradient norm의 non-finite 검사는 optimizer step 전에 별도로 유지한다.

## 6. graph, B-spline, decoder 요약

좌표는 `x/(W-1)`, `y/(H-1)`, interval 내 normalized `t`이고 polarity는 `-1/+1` node feature다.
기본 거리에는 `x,y,t`만 쓴다. radius graph는 cell 폭을 정확히 `D`로 두고 `3^d` 인접 cell에서
후보를 찾은 뒤 Euclidean `distance<D`를 다시 검사한다. 모든 ordered source를 처리해 무방향 쌍의
양 방향 edge를 만들고 self-loop는 제외한다. chunking은 exact 계산 분할이며 approximation이 아니다.

edge pseudo-coordinate `u=distance/D`에 open degree-1 B-spline basis 두 개만 활성화된다. layer는
node를 K=5 control point에 한 번 projection하고 edge마다 두 control point만 gather한 뒤 destination
incoming degree로 평균한다. 고정 graph의 basis/index는 graph layer와 IF timestep 전체에서
재사용한다. mean message에 root transform과 bias를 더하고 ANN에서는 BatchNorm과 ReLU를 적용한다.

`EventGraph.in_degree`는 graph 생성 때 한 번 계산해 6개 layer와 모든 SNN timestep, diagnostics가
공유한다. `spline_chunk_size=65536`은 최대 2,000,000 edge의 message gather를 chunk 단위로 제한해
peak GPU memory를 낮추며 edge를 생략하지 않는다. 테스트는 chunked/unchunked 출력뿐 아니라 input,
edge attribute, kernel, root, bias gradient 동등성까지 확인한다. SNN state도 마지막 layer의 full spike
sum만 유지하고 `standard_if`에서는 불필요한 previous-spike tensor를 만들지 않는다.

node feature는 downsample 4의 raster cell 안에서 평균된다. decoder는 stem, 두 residual encoder
level, 두 residual block bottleneck, analog ConvGRU, bilinear upsampling과 skip connection, sigmoid
head로 구성된다. decoder 구현은 `src/asgcn_unet/unet.py`의 `RecurrentUNetDecoder`에 분리돼 있고
`src/asgcn_unet/model.py`의 `ASGCNUNet`이 graph-raster bridge와 함께 호출한다. 자세한 구현 가정과
식 (15) 모호성은 `docs/ASGCN.md`를 본다.

## 7. ANN→SNN calibration과 IF 경로

`best.pt`는 변환되지 않은 clean `ann_inference` checkpoint다. calibration은 EventHDR train만 사용한다.
engine은 변환 전에 `best.pt`의 학습 당시 train 전체 content digest, target/event transform, final split
manifest, source tree/Git 계약과 terminal validation 완료 상태를 현재 실행과 대조한다. 그 manifest의
모든 training sample도 dataset index `0..N-1` 순서로 정확히 한 번씩 보정해야
`calibration_protocol.sealed=true`가 된다. 일부 sample을 지정한 기본 실행은 변환 전에 실패한다.
validation protocol v7은 train/validation index의 전체 sample 수, group별 수와 ordered sample identity
SHA-256을 저장한다. 보고용 ANN은 train/validation sample cap이 모두 `null`이고 실제 validation sampling이
이 commitment와 일치해야 하며, SNN calibration sampling은 source ANN의 train commitment와 같아야 한다.
EventHDR 보고용 ANN은 `best_validation_macro_ssim` 대체 경로를 허용하지 않고, 사전 계획 epoch에서
완료된 `single_final_epoch` terminal validation을 반드시 요구한다.
학습 checkpoint의 verified CUDA preflight gate도 이 계약에 포함되므로 profile을 우회한 비보고용 ANN을
보정해 보고용 SNN으로 승격할 수 없다.

calibration은 index별 `dataset[index]` 직렬 loop가 아니라 선택 순서를 보존한 `Subset`과 기존
DataLoader를 사용한다. `batch_size=1`, `shuffle=false`를 강제하면서 train config의 worker,
pin-memory, persistent worker와 prefetch 설정을 재사용해 HDF5 decode와 GPU 계산을 겹친다.

1. ANN graph layer의 BatchNorm을 kernel/root/bias에 fold한다.
2. 각 layer의 feature별 ReLU maximum `lambda_l`를 측정한다.
3. 식 (6)의 `lambda_(l-1)/lambda_l`와 `1/lambda_l` scaling을 적용한다.
4. dead channel의 raw maximum은 0과 mask로 보존하고 식 (6)에 쓰는 effective scale만 1로 두며,
   모든 threshold를 정확히 1로 둔다.
5. `best_snn.pt`에 valid sample count, dead-channel summary, persistent conversion flag와 tensor
   SHA-256뿐 아니라 source ANN checkpoint/model SHA, 선택 sample identity, data/transform/manifest,
   source와 runtime을 묶은 calibration protocol을 저장한다.
6. model state 최상위 persistent buffer `calibration_attempts`, 32-byte
   `calibration_commitment_digest`, `calibration_commitment_sealed`에 실제 시도 수와
   protocol/count/sampling 및 valid/minimum/dead-channel summary core commitment를 저장한다.
   metadata summary와 layer별 `calibration_samples_seen`, `calibration_activation_max` raw tensor,
   `normalization_scale`, `dead_channel_mask`도 교차검증한다. normalization 뒤에도 raw maximum과
   mask가 보존되므로 dead channel이 있는 checkpoint의 save→strict reload가 가능하고, 부분 보정
   tensor에 전체 metadata를 이식하거나 mask/dead-channel 수만 바꾸는 우회는 load 단계에서 거부한다.

CLI의 `--allow-unsealed-calibration`은 합성 fixture나 비보고용 진단에서만 계약 불일치를 허용한다.
이 경우 `sealed=false`와 모든 불일치 이유가 checkpoint에 영구 기록되므로 보고용 matrix에 사용하면
안 된다. override 사용 자체가 taint이므로 전체 sample을 처리했더라도 `sealed=false`다. 기본 shell
wrapper와 `run.sh`는 이 우회를 노출하지 않는다. SNN 평가 seal은 checkpoint를
다시 열 때 transform, manifest, 전체 selected identity/sampling, runtime, calibration data와 source ANN
training data, `training_source == calibration_source`를 독립적으로 재검사한다. source ANN의 epoch,
model hash, selection score/rule과 terminal-validation state를 묶은 전체 reporting contract도 별도 hash로
보존·재검증한다.

SNN inference는 threshold/normalization/calibration metadata와 layer state가 모두 일치해야 열린다.
초기 membrane은 0.5 threshold, spike amplitude는 threshold, soft reset을 쓴다.

- `literal_eq15`: 논문 식 (15)의 `+previous_spike` self-feedback까지 문자 그대로 실행한다.
- `standard_if`: 그 항을 제거한 비공식 rate-conversion 대조군이다.

두 dynamics는 같은 `best_snn.pt`에서 inference-only override로 비교한다. 마지막 graph layer의 spike
rate에는 `lambda_L`를 곱해 analog decoder 단위로 보낸다. 이는 `literal_eq15`의 ANN parity 증명이
아니다.

## 8. `scripts/run.sh`의 전체 실행 순서

이 script는 설치나 데이터 다운로드를 하지 않는다. 환경과 전체 데이터가 이미 준비된 뒤 저장소
루트에서 실행한다.

```bash
bash scripts/run.sh all
```

script가 노출하는 stage는 `check`, `profile`, `train`, `calibrate`, `eval`, `all`이다. `all`은 다음 다섯 stage를
순서대로 실행한다.

1. `check`: dependency/CUDA/full-data coverage 검사, `configs/train.json`으로 EventHDR train+eval 전체,
   `configs/aid.json`으로 EventAid-R 전체를 `inspect --validate-all`
2. `profile`: 정답 이미지 decode 없이 EventHDR train 전체 graph topology를 CUDA로 계산하고
   원자적 journal에 저장한다. edge 수 상위 10개를 기록하고 상위 3개의 CUDA 학습 step과 VRAM을
   측정한다. 별도로 최초·빈 이벤트·최소 비어 있지 않은 표본을 각각 fresh 초기화로 검사한다.
   검증된 기존 스캔은 명시적으로 재사용할 수 있으나 GPU 학습 검사는 항상 새로 실행한다.
3. `train`: profile을 현재 config/data/source/CUDA runtime에 다시 결합한 뒤 EventHDR ANN 40-epoch 학습
   또는 `RESUME_CHECKPOINT` exact resume
4. `calibrate`: EventHDR train 전체를 사용한 ANN→SNN calibration
5. `eval`: EventHDR와 EventAid-R의 전체 quality evaluation + compute benchmark matrix

`runs/profile.json`은 전수 topology와 실제 선택 표본 학습 step을 결합한 empirical gate다. 결과 자체가
명시하듯 `absolute_vram_guarantee=false`이며 전체 40 epoch의 모든 미래 allocator 상태를 증명하지
않는다. 기존 profile output은 묵시적으로 덮어쓰지 않는다. 합성 fixture용
`--allow-unverified-preflight`는 public config와 checkpoint에 `report_eligible=false`로 영구 기록되며
`run.sh all`, 기본 train wrapper와 scheduler wrapper에서는 허용되지 않는다.

`train.json` inspect가 manifest의 train 51개와 eval 19개 split을 모두 검사하므로 같은 EventHDR eval을
`hdr.json`으로 다시 decode하는 중복 검사는 두지 않는다. 오래 걸려도 파일을 조용히 제외하지 않는다.

전체 decode가 완료된 뒤 MIG의 `profile failed: Invalid device id`로 중단됐다면 `all`부터 되풀이하지
않고 코드를 갱신한 후 `profile` → `train` → `calibrate` → `eval`을 순서대로 실행한다.
이는 topology scan 전 runtime 정보 조회에서 실패해 학습 산출물이 없는 경우의 절차다.
이미 만들어진 report/checkpoint는 삭제하거나 자동 덮어쓰지 않는다.
구체적인 조건과 명령은 [서버 재개 안내](docs/SERVER.md#mig에서-전체-데이터-검사-후-profile만-실패한-경우)에 있다.

`eval` stage matrix는 다음 18개 run이며 각 run마다 `evaluate`와 `benchmark`를 둘 다 실행한다.

| dataset | mode | dynamics | T | checkpoint |
|---|---|---|---|---|
| EventHDR | ANN | 해당 없음 | 해당 없음 | `best.pt` |
| EventHDR | SNN | `literal_eq15` | 4, 8, 16, 32 | `best_snn.pt` |
| EventHDR | SNN | `standard_if` | 4, 8, 16, 32 | `best_snn.pt` |
| EventAid-R | ANN | 해당 없음 | 해당 없음 | `best.pt` |
| EventAid-R | SNN | `literal_eq15` | 4, 8, 16, 32 | `best_snn.pt` |
| EventAid-R | SNN | `standard_if` | 4, 8, 16, 32 | `best_snn.pt` |

전체 schedule만 확인하려면 다음을 사용한다.

```bash
DRY_RUN=1 bash scripts/run.sh all
```

중요 override는 `RESUME_CHECKPOINT`, `PROFILE_OUTPUT`, `PROFILE_SAMPLES`, `PROFILE_TOP_DENSITY`,
`PROFILE_RESUME`, `PROFILE_REUSE_REPORT`, `PROFILE_CPU_THREADS`, `RESTART_TRAIN`,
`CALIBRATION_SAMPLES`, `SIMULATION_STEPS_LIST`,
`BENCHMARK_WARMUP`, `BENCHMARK_STEPS`, 세 config path, ANN/SNN checkpoint path와
`REQUIRE_CUDA`다. calibration output과 evaluation artifact는 기본적으로 덮어쓰지 않는다. fresh
training도 run directory에 기존 핵심 artifact가 있으면 중단한다. 기존 결과를 보존한 채 새 output
directory/config를 쓰는 것이 원칙이다.

## 9. 평가 지표와 artifact

quality는 frame별 PSNR, Gaussian SSIM, RMSE와 조건부 `temporal_l1`이다. `eval.lpips=true`와 optional
dependency가 있을 때만 LPIPS를 계산한다. 결과는 다음 세 수준으로 집계한다.

- `micro`: 모든 frame 평균
- `macro`: group별 평균을 다시 같은 가중치로 평균
- `per_scene`: 호환성을 위해 유지된 JSON key; EventHDR에서는 H5 sequence-file group,
  EventAid-R에서는 ZIP scene group이다

EventHDR의 `macro`를 physical scene macro라고 부르면 안 된다. standalone evaluation은 H5 filename을
group으로 쓰고, training final validation은 split-local H5 group ID를 쓴다.

frame metric은 MSE를 한 번만 계산해 PSNR/RMSE가 공유하고, Gaussian SSIM window는 device/dtype/
channel/window별 bounded cache를 쓴다. PSNR·SSIM·RMSE·선택적 LPIPS·temporal 값을 stack해 frame당
한 번의 CPU transfer로 가져온다.

evaluate latency는 dataset read와 host-to-device copy 뒤에 graph construction+model forward를
동기화해 잰다. benchmark는 dataset I/O와 H2D를 timer 밖에 두고 warmup 뒤 CUDA Event 또는 CPU
`perf_counter`를 쓴다. benchmark가 기록하는 항목은 mean/p50/p90/p95/p99/max latency, FPS,
raw/retained events per second, graph nodes per second, event retention, 평균 node/edge, isolate ratio,
max degree, SNN layer별 firing rate, 전체 firing rate, RTF p95, deadline miss ratio와 peak allocated/
reserved GPU memory다.

`eval.output_dir` 아래 run label은 다음과 같다.

```text
ann/
snn_literal_eq15_T4/
snn_literal_eq15_T8/
...
snn_standard_if_T32/
```

`metrics.json`, `frames.csv`, `predictions/`는 evaluate가 만들고 `benchmark.json`은 benchmark가 만든다.
동일 run label의 기존 artifact가 있으면 덮어쓰지 않고 실패한다. prediction filename은 순번, 안전한
slug와 sample ID hash를 조합해 OS 금지 문자와 충돌을 피한다.

JSON/CSV/checkpoint는 unique temporary file에 끝까지 기록하고 flush한 뒤 원자적으로 replace한다.
JSON writer는 `allow_nan=false`이고 evaluate/benchmark는 target, prediction, metric, graph diagnostic,
latency와 interval metadata를 저장 전에 재귀적으로 finite 검사한다. 완전 일치 frame의 PSNR은 무한대
대신 문서화된 120 dB 상한을 사용한다. `scripts/run.sh`의 각 stage는 `runs/status/<stage>.json`에
`RUNNING`, `COMPLETED`, `FAILED`와 종료 코드를 남겨 partial artifact와 완료 run을 구분한다.

기본 평가 config는 quality 비교를 위해 `precision=fp32`, `tf32=false`를 명시한다. 별도 성능 run은
`amp_fp16` 또는 `bf16`을 선택할 수 있지만 결과에는 requested/effective precision, autocast dtype,
model parameter dtype, device와 requested/effective TF32가 함께 기록되므로 FP32 표와 섞지 않는다.
config/checkpoint/output 경로는 shareable artifact에서 repository-relative label 또는 `$EXTERNAL/<name>`로
저장하며 routine environment report도 명시적으로 private provenance를 요청하지 않는 한 사용자 경로와
hostname을 출력하지 않는다.

보고용 ANN 평가에는 verified CUDA preflight가 포함된 clean `ann_inference`, finite macro-SSIM selection,
training protocol v5(단일 프레임) 또는 v6(독립 시퀀스 배치)와 validation protocol v7이 필요하다.
v6는 일치하는 실제 full-batch CUDA gate도 요구한다. 보고용 SNN은 그 ANN에서 봉인된
`calibration_protocol.sealed=true`를 요구한다. `metrics.json.evaluation_protocol`과
`benchmark.json.benchmark_protocol`은 public config/model, checkpoint file·tensor와 lineage,
현재 eval dataset의 전체 content SHA-256·transform·manifest·coverage·sampling, source,
runtime/precision을 path-free canonical hash로 결합한다. 18-run 연속 실행은 SHA path-token과
size/mtime/ctime signature만 담는 host-local cache로 대용량 파일 hash를 재사용한다.
보고 가능한 quality evaluation은 `eval.max_samples=null`이어야 한다. cap이 있는 실행은 기본적으로
거부되고 합성 테스트용 명시적 우회를 써도 `report_eligible=false`다. 반면 benchmark의 warmup/측정
step 수는 품질 sample cap이 아닌 별도 compute-only sampling protocol이다.
EventHDR lineage는 planned/completed/checkpoint epoch가 같고 selection rule이
`single_final_epoch_macro_ssim`인 경우에만 보고 가능하다. `check_env.py`와 `inspect`의
`--include-private-host-provenance`는 로컬 진단 전용이며 그 출력은 공개·첨부하지 않는다.
source ANN은 EventHDR 학습만 허용하고 training protocol의 seed, optimizer/scheduler, loss, data order,
AMP와 runtime을 public config에 다시 결합한다. 평가 root도 explicit expected file count와 final/fixed
manifest의 정확한 파일 집합을 증명해야 한다. EventHDR 실행은 현재 content·transform·manifest가 source
ANN validation과 같아야 하며 protocol hash는 실제 mode, SNN T와 effective dynamics를 포함한다.

이 SHA-256 계약은 reproducibility identity와 우발적·부분적 metadata 변조 검출 수단이며 전자서명이
아니다. checkpoint와 내부 hash 전부를 악의적으로 다시 쓰는 공격까지 인증하려면 각 최종 artifact의
file SHA-256을 별도 immutable 실험 원장, 서명 release 또는 접근통제된 archive에 보존해야 한다.

합성 테스트 전용 `--allow-unsealed-checkpoint-for-non-reporting`은 결과에
`report_eligible=false`와 모든 이유를 영구 기록한다. public shell/scheduler wrapper는 이 옵션을
노출하지 않으며 해당 결과는 보고용 표에 포함하면 안 된다.

## 10. provenance, checkpoint integrity와 exact resume

학습 directory의 핵심 artifact는 `config.json`, `history.json`, `last.pt`, `best.pt`와 hidden data hash
cache다. `validate_every=null`이므로 `history.json`의 validation은 마지막 epoch에서만 채워지고
`best.pt`는 그 마지막 candidate다. training protocol에는 처음부터 `planned_epoch=40`을 봉인하고
checkpoint에는 terminal validation의 완료 여부와 완료 epoch를 저장한다. 마지막 평가가 끝난 뒤
epochs만 늘려 같은 run을 resume하는 것은 거부하며, 연장 실험은 새 output의 새 protocol로 분리한다.

validation protocol에는 dataset transform, manifest schema와 모든 file 목록/group mapping,
validation sample identity/context policy, SSIM 정의, selection rule과 train/eval 원본 전체 file의
SHA-256 결합 digest가 저장된다. 절대 root와 mtime은 checkpoint 비교 identity가 아니어서 같은 byte의
복사본을 다른 mount에서 쓸 수 있다. local hash cache의 key도 absolute path 대신 path SHA token으로
저장해 cache artifact가 mount/account를 노출하지 않게 했다. 같은 source의 size/mtime/ctime이 모두
같을 때만 cached full hash를 재사용하며, 기본 `train.rehash_data=true`는 첫 sealed run에서 51+19 H5를
전부 다시 읽어 digest를 계산한다.

training protocol에는 optimizer/GC 축, scheduler, loss weights, gradient clip, data order/workers,
effective AMP, 봉인된 final-only validation terminal epoch, recurrent detach,
torch/CUDA/cuDNN/GPU/TF32/determinism, `src/**/*.py` tree hash, Git commit과 source dirty 여부가 들어간다.
checkpoint의 model tensor bytes도 이름·dtype·shape를 포함해 SHA-256으로 묶는다.

이 digest는 checkpoint 내부 metadata와 tensor bytes의 **일관성·우발적 손상 탐지** 용도다. 공격자가
checkpoint와 내부 digest를 함께 바꾸는 경우를 막는 authenticity 서명이 아니다. 배포물의 출처 인증이
필요하면 저장소 밖의 신뢰 경계에서 생성한 signed manifest 또는 서명된 release checksum이 별도로
필요하다.

exact resume은 다음을 모두 요구한다.

- resume checkpoint가 같은 configured run directory 안에 있을 것
- model config, validation protocol, training protocol과 source/data digest가 일치할 것
- optimizer, scheduler, GradScaler, history, epoch와 best score가 있을 것
- Python, NumPy, torch와 visible CUDA device별 RNG state가 유효할 것
- 과거 `best.pt`가 존재하고 `last.pt`의 best digest/protocol과 일치할 것
- ANN/SNN conversion state와 checkpoint type이 일치할 것
- checkpoint의 verified CUDA preflight gate가 현재 config/data/source/runtime과 일치할 것

학습만 이어갈 때는 다음을 사용한다.

```bash
RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  bash scripts/run.sh train
```

provenance가 엄격하므로 commit이나 source/runtime 변경 뒤에는 resume이 거부될 수 있다. 또한 runtime
상태를 기록하고 비교하더라도 PyTorch/CUDA의 모든 kernel이 bitwise deterministic하다는 보장은
없다. exact resume은 저장된 state와 protocol의 정확한 복원을 뜻하며 서로 다른 hardware에서의
bitwise 동일성을 과장하지 않는다.

## 11. Linux GPU 서버 절차

실제 연산은 Linux GPU server에서 수행한다. 설치는
[README의 설치 및 실행](README.md#설치-및-실행)을 단일 진입점으로 사용한다.
저장소를 clone하고 `asgcn` Conda 환경에 설치하며, 이후 명령은 저장소 root에서 실행한다.
기존 환경이나 프로젝트 폴더는 덮어쓰지 않는다.

Git은 서버에 이미 설치된 것을 사용한다. `conda create -n asgcn --override-channels -c conda-forge
python=3.12.14 pip`로 생성한 non-base 환경을 활성화하고 `bash scripts/setup.sh`로 직접 설치한다.
다운로드·run·개별 train/calibrate/eval·scheduler는 같은 Conda Python을 사용한다. `.env` 복사는
필요 없으며 설치기는 기존 `.env`도 읽거나 변경하지 않는다. 재접속 시에는 기존 저장소로 이동해
`conda activate asgcn`만 한다. 전체 Ruff/pytest와 history/provenance release gate는 유지관리용이며,
설치할 때마다 실험 사용자가 반복할 절차가 아니다.

기존 서버 전환은 [환경 전환 안내](docs/SERVER.md#기존-venv-설치에서-전환)를 따른다. 실행 중인
job을 종료하고 결과를 보존한 뒤 같은 Conda 환경에서 pull·설치한다. 기존 데이터는 다시 받지 않으며,
이전 환경은 새 Conda runtime과 GPU/data 검증을 통과하기 전에는 정리하지 않는다. source/runtime이
바뀌면 기존 run의 exact resume이 거부될 수 있으므로 checkpoint를 강제로 이어 붙이지 않는다.

유지관리자는 다음 배포 조건을 확인한다.
배포할 commit SHA에 대한 로컬 실제-marker release gate 기록과 같은 SHA의 GitHub Actions 필수 gate 통과를
대조한다. CI는 실제 marker를 받지 않는다. 둘 중 하나라도 확인되지 않으면 해당 checkout을 본실험에
사용하지 않는다. 최신 CI badge나 문서의 상태 설명만으로 대신하지 않는다.

프로젝트 core는 Python 3.10 이상을 지원하지만 기본 Linux 서버 실행 profile은
`constraints/server.json`의 Python **3.12.14**, PyTorch **2.13.0+cu126**, CUDA runtime **12.6**으로
고정한다. `constraints/server.txt`는 pip·setuptools·wheel, CUDA library·Triton을 포함한 전이
의존성의 정확한 버전과 배포 파일 SHA-256을 고정한다. setup은 `--require-hashes --only-binary=:all:`로
설치하며 최신 bootstrap package를 임의로 올리지 않는다. `constraints/py312.txt`의 core/dev 버전도
함께 관리하며 핵심은 numpy 2.5.2, h5py 3.16.0, Pillow 12.3.0, pytest 9.1.1, Ruff 0.16.5다.
Linux wheel은 glibc 2.28 이상을 요구하며 setup은
공식 CUDA 12.6 index의 고정 build를 설치한다. `.env`나 `TORCH_VERSION`·`TORCH_INDEX_URL` 변경으로
다른 CUDA build를 섞지 않는다. 이 profile은 Conda 자체의 모든 package·OS·driver·GPU까지 동결하거나
hardware 간 bitwise 동일성을 보장하는 lock이 아니다.
이는 확인된 Python·PyTorch·CUDA baseline에 맞춰 새로 만든 고정 profile이며 이전 서버의 전체
package inventory를 복제한 것은 아니다. 유지관리자는 `constraints/server.in`에서 lock을 재생성해
profile과 일치하는지 검증한다. lock 생성 도구 `uv`는 서버 설치의 필수 프로그램이 아니다.
설치는 GPU가 없는 login node에서도 할 수 있지만 실제 GPU allocation에서 CUDA 검증이 실패하면
본실험을 진행하지 않는다. `nvidia-smi`와 `df -h .`는 필요할 때 사용하는 선택 진단이다.

core runtime dependency는 torch, NumPy, h5py, Pillow와 tqdm이다. development extra는 pytest와
Ruff다. optional eval extra의 LPIPS는 고정 서버 profile에 포함하지 않는다. 추가 package를 설치하는
확장 실험은 dependency 변경을 따로 기록하고 runtime을 다시 검증하며 기본 profile의 결과와 혼동하지 않는다.

데이터 배치 뒤 전체 readiness를 검사한다.

```bash
conda activate asgcn
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt --runtime-profile constraints/server.json

asgcn-unet inspect --config configs/train.json --samples 2 --validate-all
asgcn-unet inspect --config configs/aid.json --samples 2 --validate-all
```

`check_env`는 CUDA, GPU 이름/VRAM, Python/torch/CUDA/cuDNN, server profile·lock mismatch, glibc, data와 runs의 남은
공간, runs 쓰기 가능 여부, EventHDR exact 51/19 이름과 EventAid-R exact 14 ZIP을 출력·검사한다.
CUDA가 사용 가능할 때는 먼저 `torch.cuda.init()`으로 초기화한 뒤 runtime 장치 수와 각 장치의
이름/VRAM을 조회한다. MIG에서 초기화 전 NVML count로 반복 범위를 만들면 실제 runtime 장치보다
큰 index에 접근할 수 있으므로 초기화 전 count를 쓰지 않는다. CUDA가 불가능하면 장치 속성을
조회하지 않으며 `--require-cuda`는 계속 실패한다. 초기화·조회 중 `AssertionError`, `RuntimeError`,
`OSError`, `DeferredCudaCallError`도 실패로 종료한다. 공개 오류에는 예외 종류만 출력하고 원문
traceback에 담긴 host 경로는 노출하지 않는다. 원문 예외는 `--include-private-host-provenance`를
명시한 비공개 진단에서만 출력한다. scheduler의 장치 visibility나 GPU 할당은 변경하지 않는다.
profile의 `_runtime_provenance`도 cuDNN version·GPU 속성을 읽기 전에 CUDA를 초기화한다.
PyTorch 2.13의 cuDNN 초기화 자체가 장치 수를 먼저 읽고 capability를 순회하므로, 단지 profile의
명시적 장치 조회 순서만 바꾸어서는 부족하다. `check_env`와 profile은 별도 프로세스여서 앞 단계의
초기화 상태를 재사용할 수도 없다. engine의 CUDA RNG capture/restore 역시 모든 장치의 상태를
열거하기 전에 초기화한다. 이 수정은 의존성 lock, 모델·데이터 protocol 또는 GPU visibility를 바꾸지 않는다.
`--validate-all`은 모든 target/event block을 실제 decode하므로 전체 데이터에서는 오래 걸린다.

## 12. scheduler

SLURM과 PBS/Torque 각각 profile, train, calibration, evaluation entrypoint가 있다.

```text
server/profile.sbatch      server/profile.pbs
server/train.sbatch        server/train.pbs
server/calibrate.sbatch    server/calibrate.pbs
server/eval.sbatch         server/eval.pbs
```

기본 요청은 GPU 1개, CPU 8개, RAM 32 GB다. profile/calibration은 12시간, train은 48시간, evaluation은
8시간으로 작성돼 있으나 partition/account/GPU type/resource 이름과 walltime은 cluster 규칙과 실제
측정에 맞춰 바꿔야 한다. wrapper는 `PROJECT_ROOT` 또는 scheduler submit directory를 검증하고 잘못된
checkout에서 실행하지 않는다. `CUDA_MODULE`은 opt-in이다.

SLURM dependency 예시:

```bash
conda activate asgcn
profile_id=$(sbatch --parsable \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX" server/profile.sbatch)
train_id=$(sbatch --parsable --dependency=afterok:${profile_id} \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX" server/train.sbatch)
cal_id=$(sbatch --parsable --dependency=afterok:${train_id} \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX" server/calibrate.sbatch)
sbatch --dependency=afterok:${cal_id} \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",CONFIG_PATH=configs/hdr.json,CHECKPOINT_PATH=runs/train/best_snn.pt,INFERENCE_MODE=snn,SIMULATION_STEPS=16,SNN_DYNAMICS=literal_eq15 \
  server/eval.sbatch
```

SLURM에는 login 환경 전체가 아니라 job에 필요한 변수만 전달한다. 각 job은 전달한 `CONDA_PREFIX`의
Python을 사용하며, 명시적 `PYTHON_BIN`도 같은 Conda 환경을 가리켜야 한다. scheduler log는 기본적으로 host/job
식별자를 생략하고 config/checkpoint basename만 기록한다. 정확한 값은 공개하지 않을 로컬 진단에서만
`INCLUDE_PRIVATE_HOST_PROVENANCE=1`로 opt-in한다. Slurm의 `slurm-...-<job-id>.out/.err`와 PBS의
`<job-name>.o<job-id>` 같은 raw scheduler log는 파일명 자체에 job ID가 있으므로 그대로 공개하지 않는다.
기본 provenance log만 `logs/public/train.stdout.log` 같은 중립 파일명으로 복사하고 저장소 밖 로컬
denylist로 공개 후보의 내용을 로컬에서 검사한 뒤 공유한다.

```bash
python scripts/scan_private_text.py logs/public/train.stdout.log \
  --root "$PWD" --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

scan 실패 log와 `INCLUDE_PRIVATE_HOST_PROVENANCE=1` opt-in log는 rename/copy 여부와 무관하게 비공개다.

PBS/Torque dependency 예시:

```bash
conda activate asgcn
profile_id=$(qsub -v CONDA_PREFIX="$CONDA_PREFIX" server/profile.pbs)
train_id=$(qsub -W depend=afterok:${profile_id} \
  -v CONDA_PREFIX="$CONDA_PREFIX" server/train.pbs)
cal_id=$(qsub -W depend=afterok:${train_id} \
  -v CONDA_PREFIX="$CONDA_PREFIX" server/calibrate.pbs)
qsub -W depend=afterok:${cal_id} \
  -v CONDA_PREFIX="$CONDA_PREFIX",CONFIG_PATH=configs/hdr.json,CHECKPOINT_PATH=runs/train/best_snn.pt,INFERENCE_MODE=snn,SIMULATION_STEPS=16,SNN_DYNAMICS=literal_eq15 \
  server/eval.pbs
```

전체 18-run matrix를 scheduler로 돌리려면 dataset/dynamics/T별 eval job을 각각 제출해야 한다.
단일 allocation에서 순차 실행할 때만 `scripts/run.sh`를 직접 사용한다. 이 저장소는 검증되지 않은
Docker 경로를 제공하지 않고, MobaXterm/SSH에서 사용하는 native Conda 환경과 scheduler wrapper만
지원한다.

## 13. 파일별 책임

| 경로 | 책임 |
|---|---|
| `src/asgcn_unet/graph.py` | node 정규화, exact radius graph, cached degree, chunked B-spline, BN fold, Eq.(6), IF loop |
| `src/asgcn_unet/model.py` | `ASGCNUNet`, graph build 연결, rasterization, decoder 연결, diagnostics |
| `src/asgcn_unet/unet.py` | residual U-Net, analog ConvGRU, `RecurrentUNetDecoder` |
| `src/asgcn_unet/data/eventhdr.py` | H5 index/검증, zero-event 보존, frame interval 구성 |
| `src/asgcn_unet/data/eventaid_r.py` | ZIP 직접 읽기, next-GT pairing, timestamp/shape 검증 |
| `src/asgcn_unet/data/common.py` | luminance/tone map, crop, exact-size event cap, sample schema |
| `src/asgcn_unet/data/factory.py` | manifest schema, exact coverage, split-local H5 group |
| `src/asgcn_unet/losses.py` | Charbonnier, SSIM loss, gradient loss |
| `src/asgcn_unet/metrics.py` | Gaussian SSIM, PSNR, RMSE, temporal metric, micro/macro 집계 |
| `src/asgcn_unet/engine.py` | train/validation/calibration/evaluate/benchmark, checkpoint·resume·provenance |
| `src/asgcn_unet/preflight.py` | 전체 train topology scan, 최고 밀도 CUDA 학습-step 측정·재검증 |
| `src/asgcn_unet/cli.py` | inspect/profile/verify-profile/train/calibrate/evaluate/benchmark CLI |
| `src/asgcn_unet/scan.py` | 원자적 구간 저장, hash 검증, 단일 writer lock과 전수검사 재개 |
| `src/asgcn_unet/recovery.py` | 명시적 metadata-only 학습 실패 보존·재시작 |
| `configs/train.json` | EventHDR 51 train + 19 final-only internal eval 학습 protocol |
| `configs/hdr.json` | EventHDR official eval ANN/SNN 공용 설정 |
| `configs/aid.json` | EventAid-R 14-scene ANN/SNN 공용 설정 |
| `manifests/eventhdr_split.json` | official separate roots와 H5 sequence-file semantics |
| `manifests/eventaid_r.json` | 14 ZIP 이름, URL, 표시 용량 |
| `constraints/server.json`, `constraints/server.txt`, `constraints/server.in` | 고정 Linux 서버 runtime profile, 전이 의존성 hash lock과 유지관리용 입력 |
| `constraints/py312.txt` | core/dev package 버전 교차검사 |
| `scripts/setup.sh`, `scripts/check_env.py` | server 설치와 환경/data inventory |
| `scripts/get_hdr.py`, `scripts/get_hdr.sh` | EventHDR 서버 직접 다운로드 CLI와 선택적 archive/source/shared import/check |
| `scripts/hdr_http.py` | 익명 OneDrive HTTP 조회, 재개·재시도·링크 갱신과 크기/SHA-256/HDF5 검증 |
| `scripts/get_aid.sh` | EventAid-R 14 ZIP 다운로드/재개/검사 |
| `scripts/train.sh`, `scripts/calibrate.sh`, `scripts/eval.sh` | 개별 GPU wrapper |
| `scripts/run.sh` | `check|profile|train|calibrate|eval|all`, CUDA gate와 18-run matrix orchestration |
| `scripts/scan_private_text.py` | current tracked text와 전체 reachable Git blob의 generic/external-marker privacy gate |
| `scripts/build_code_summary.py` | deterministic `code_summary.md`, file/snapshot SHA와 clean source provenance gate |
| `server/` | SLURM/PBS profile→train→calibrate→eval entrypoint |
| `docs/ASGCN.md` | 논문 core와 구현 가정의 경계 |
| `docs/EXPERIMENT.md`, `docs/SERVER.md` | 실험 protocol과 server 운용 보조 문서 |
| `tests/` | fixture 기반 CPU unit/integration/end-to-end 회귀검사 |

`code_summary.md`는 generator output 자신을 제외한 모든 tracked UTF-8 text를 `# 파일경로`와 원문 code
block 형태로 담는다. header에는 file별 SHA-256과 전체 snapshot SHA-256, 생성 당시 Git provenance가
있다. dirty working tree에서 생성하면 superseded commit을 계속 가리키지 않도록 commit/tree identity를
`null`로 두고 snapshot SHA-256을 검증 identity로 사용한다. clean generation의 commit/tree도 summary를
포함하는 새 commit과 자기참조적으로 같을 수 없으므로 현재 remote SHA라고 주장하지 않는다. 본문과
manifest의 정확한 동일성은 `--check`, remote 배포 동일성은 별도의 remote commit/CI 확인으로 판정한다.
release에서는 코드·README·인계서·서버 문서와 검증 기록을 검토한 뒤
source commit을 확정한다. clean source에서 summary를 재생성하고 summary만 별도
commit한다. 이 사이에 다른 tracked 파일 수정이나 source history의 amend/rebase/rewrite를 끼우지
않는다. 변경이 필요하면 새 source commit을 확정한 뒤 다시 생성한다.
`--check --require-clean-provenance`는 기록된 source commit/tree가 유효하고 현재 HEAD까지 summary 외
tracked source 차이가 없는지 검사한다. dirty snapshot에는 이 release-only gate 통과를 주장하지
않는다. summary-only commit 뒤 배포/CI 결과는 해당 최종 SHA의 Actions와 별도 배포 기록으로
확인한다. 결과를 문서에 추가하는 경우에도 source 변경이므로 위 commit·재생성 순서를 다시 따른다.

## 14. 테스트 상태와 검증 범위

환경 전환 기준 commit `1afb40f`의 2026-08-30 Windows CPU 통합 pytest 결과는 **571 passed, 27 skipped**다.
skip 22건은 Linux 전용 설치 shell test, 5건은 symlink 권한 관련 검사다. Windows native access-violation
진단이 출력된 후에도 pytest는 exit 0으로 종료했으나, 원인이 확인되지 않아 무경고 검증으로 인정하지 않는다.
이 결과는 기존 pytest 임시 디렉터리 권한 충돌을 피하도록 저장소 밖 새 임시 디렉터리를 지정해 얻었다.
16개 shell entrypoint의 MSYS Bash 구문 검사도 수행했다. 실제 Linux Conda 설치와 동일 고정 profile의
pytest, 별도 Ubuntu/Windows Python matrix는 해당 배포 SHA의 CI에서 확인한다.
아래 명령은 source 검증용이며 원격 배포 성공 여부는 뒤의 release gate로 별도 판정한다.

history scanner는 `git rev-list --objects --all`의 LF 출력으로 통일했다. 구 Git이 `-z`를 받아도 최신
NUL object protocol을 출력하지 않는 차이를 회피하며, LF fixture·SHA-1/SHA-256·공백/탭/CR/비ASCII
path hint와 malformed OID 회귀를 추가했다. path hint는 개행에서 잘리거나 동일 blob의 여러 경로 중
하나일 수 있으므로 실제 내용은 항상 OID로 읽는다. 로컬 Git 2.53.0.windows.3에서는 검증했지만 실제
Linux Git 2.47.3 실행기는 없으므로 해당 환경의 실측 통과를 주장하지 않는다.

```bash
python -m compileall -q src tests scripts
python -m ruff check .
python -m pytest -q
(
  for script in scripts/*.sh server/*.sbatch server/*.pbs; do
    bash -n "$script" || exit 1
  done
)
python scripts/build_code_summary.py --check
python scripts/scan_private_text.py --all-tracked --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
git diff --check
```

배포 전에는 complete non-shallow clone에서 다음 로컬 실제-marker release gate를 별도로
실행한다. marker는 저장소 밖 로컬 denylist 또는 로컬 환경변수 `PRIVATE_MARKERS_B64`로만 주입한다.
`--require-external-patterns`는 빈 denylist를 거부하며, 실제 marker를 GitHub에 업로드하지 않는다.

```bash
python scripts/scan_private_text.py \
  --all-tracked --all-history --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

source commit과 summary-only commit을 확정한 뒤에는 같은 최종 SHA에서 위 로컬
실제-marker 검사와 아래 clean provenance gate를 통과해야 한다. CI는 실제 marker 없이 generic
current-tree/history 검사와 clean provenance를 실행한다. 원격 배포 후 같은 최종 SHA의 GitHub
Actions 필수 job 전체의 성공을 확인하며, CI 성공으로 로컬 실제-marker 검사를 대신하지 않는다.

```bash
python scripts/build_code_summary.py --check --require-clean-provenance
```

2026-08-30 실파일 검증: 공식 train `38.h5`(388,073,496 bytes,
SHA-256 `6bec6badc2ed41be079723e1fc6e081808684904b6dd83f5db179f2760ee7cf6`)의 1,129개 이미지 전부에
`event_idx`가 없고 `timestamp`는 존재했다. 복원 인덱스 전체를 독립 NumPy 계산과 대조하고 1,129개
sample을 전부 decode했으며 전후 원본 SHA-256이 같았다. 기존 `26.h5`의 저장 인덱스 500개도 같은
복원 계산과 일치했고, 기존 인덱스를 유지한 전체 500개 sample decode가 통과했다. 이 검증은 두 파일의
로더 검증이며 전체 70개 H5·EventAid-R·GPU 학습 완료를 뜻하지 않는다.
이 인덱스 수정 후 로컬 Windows CPU 통합 pytest는 **598 passed, 27 skipped**로 종료했다.
skip은 Linux 전용 설치 shell test 22건과 symlink 권한 관련 5건이며, 이 실행에서는 native
access-violation 진단이 발생하지 않았다. 배포 판정은 수정 commit의 CI 결과로 별도 확인한다.

같은 날짜 EventAid-R 호환성 검증에서는 공식 ZIP 14개의 central directory와 작은 metadata를
HTTP Range로 읽고, 수정한 production `_build_index()`로 **14/14 인덱싱 성공 및 51,512개 쌍**을
확인했다. 실제 GT 51,529개는 JPG 10개 장면과 PNG 4개 장면에 분포한다. 일반 13개 장면의
44,168개 쌍과 `R-traffic` 7,344개 쌍의 합이며, 인덱스 검증 전송량은 합계 9,314,919 bytes였다.
이는 전체 ZIP 내용 decode나 전체 GPU 실험 완료가 아니다.

실제 데이터 decode는 PNG `R-bear`의 전체 65개 쌍, JPG `R-ball`의 첫·중간·마지막 3개 쌍,
`R-traffic` 첫 구간의 첫·마지막 2개 쌍에서 완료했다. 별도로 R-traffic 네 구간의 경계를 포함한
실제 event block 12개를 CRC 검사와 함께 읽어 정렬 순번 기반 timestamp 대응 및 구간 끝 제외를
검증했다. 다른 구간의 추가 영상 Range 읽기는 원격 HTTP 200/403 응답으로 중단돼 전체 영상 검증으로
계산하지 않았다. 이 점은 서버에 이미 받은 ZIP을 읽는 로더의 동작과 구분한다.
JPEG/parts/state 분리 수정 후 Windows CPU 통합 pytest는 **660 passed, 27 skipped**로 종료했고,
Ruff도 통과했다. skip 27개는 위와 같은 플랫폼·권한 조건이며 이번 실행에서도 native
access-violation 진단은 발생하지 않았다.

2026-08-31 사용자가 제공한 서버 로그에서는 EventHDR 전체 inspect 요약이 **106,707 samples**로
끝났고, EventAid-R은 **51,512/51,512 samples**, `validation_complete=true`로 전체 decode를 완료했다.
EventAid-R 검사 시간은 56분 9초였다. 이후 profile의 환경 검사는 고정 Conda runtime과 A100 MIG
할당을 확인했지만, 새 profile 프로세스의 runtime 정보 조회에서 `Invalid device id`로 중단됐다.
이 로그는 서버 데이터 읽기 검증의 증거이며 topology scan·CUDA 학습 step·40-epoch 학습이나 최종
평가 완료의 증거는 아니다. 로컬에서 전체 데이터를 독립적으로 재검증한 결과와도 구분한다.

같은 서버 로그의 EventAid-R timestamp 진단은 총 **4,427,295,458 events** 중 **244,912,587 events**가
현재 pairing interval 밖에 있음을 기록했다(`outside_interval_fraction=0.055318780804983265`, 약
**5.531878%**). `strict_interval_validation=false`이므로 전체 decode 통과만으로 event/GT 시간 정렬의
의미적 타당성이 확정되지는 않는다. EventHDR 학습 진행과 별개로 EventAid-R 복원 품질을 해석할 때는
공식 timestamp 기준·구간 의미와 장면별 정렬을 검토해야 한다. 이 수치를 없애기 위한 offset 자동
튜닝, event 삭제나 검증 결과의 사후 재라벨링은 하지 않는다. 이번 MIG 수정도 timestamp 정책을 바꾸지 않는다.

MIG profile 수정 후 로컬 Windows CPU 통합 pytest는 **684 passed, 27 skipped**로 종료했다.
새 24개 회귀검사는 초기화 전 physical count 8개와 초기화 후 runtime count 1개/3개의 차이,
cuDNN/RNG 열거 순서, CPU 선택, 초기화 실패 전파 및 전체 runtime GPU의 RNG 보존을 검증한다.
설치된 PyTorch cuDNN 초기화 함수 자체도 모의 8→1 장치에 적용해 수정 전 `Invalid device id`,
명시적 초기화 후 성공을 확인했다. 이는 실제 서버 MIG의 topology scan이나 학습 완료를 뜻하지
않는다. skip 27개는 위와 같은 플랫폼·권한 조건이며 Ruff도 통과했다.

주요 회귀 범위는 다음과 같다.

- strict undirected radius graph와 cell implementation의 pairwise reference parity
- degree-1 open B-spline endpoint, gradient, 초기화, hand calculation과 autograd
- BN fold, 식 (6), dead-channel raw/scale/mask save→strict reload, IF soft reset, dynamics 차이, basis cache
- EventHDR numeric image order와 EventAid exact offset/timestamp diagnostic을 포함한 구조·좌표·polarity·pairing·multiprocess safety
- EventAid PNG/JPG/JPEG decode, 확장자 간 numeric GT 중복, upload parts의 정확한 파일·timestamp coverage,
  구간 내부 pairing, part 경계의 state/temporal reset과 ZIP 단위 macro 집계 보존
- exact-size event cap 경계와 zero-event frame
- manifest separate-root/physical-scene claim 차단과 exact file coverage
- final-only validation, balanced/context schedule, loss/gradient non-finite guard
- checkpoint tensor digest, conversion state, provenance와 exact resume 거부 조건
- sealed calibration data/source/runtime/final-manifest, 전체 sample coverage와 변조 거부,
  final-only terminal epoch 변경 차단
- full-train topology scan, 최고 밀도 CUDA probe report 계약과 train preflight 재결합/우회 전파
- evaluate/benchmark checkpoint seal·전체 data/source/runtime/precision provenance,
  strict/atomic artifact와 전체 orchestration matrix
- float target 계약, EventAid ZIP logical duplicate, config/loss/batch/sample-limit fail-fast
- 전체 tracked text의 generic/external-marker privacy scan과 deterministic code snapshot
- 구·신 Git의 공통 LF history 열거와 모든 shell entrypoint의 개별 구문 검사
- 산출물의 absolute path/hostname redaction과 Linux 서버 설치 절차
- CUDA 초기화 전 physical count와 초기화 후 runtime count가 다른 MIG 모의 장치, 다중 GPU,
  CUDA 불가·0-device·초기화/조회 오류와 공개/private opt-in 예외 출력의 31개 CPU 회귀검사
- EventHDR 서버 HTTP 다운로드의 정확한 metadata, SHA-256·HDF5 검사, Range 이어받기,
  오류 재시도, 익명 token 갱신·공유 재접근, URL·symlink·로그 보호 회귀검사
- Conda interpreter 선택, nested venv·외부 pip destination 거부, 설치 전 OS/Python/glibc 검증,
  해시 lock 설치 순서, 정확한 CUDA build·전이 package profile, scheduler·standalone 실행 경로

GitHub Actions는 Ubuntu/Windows의 Python 3.10/3.11/3.12 pytest matrix와 Linux Conda Python 3.12.14
설치·고정 dependency profile·전체 pytest·Ruff/shell syntax/privacy/snapshot job을 정의한다.
Conda job은 GPU가 없는 runner에서 공식 cu126 wheel 설치를 검증하며 CUDA 학습은 실행하지 않는다.
외부 Action은 mutable tag 대신 검증한 40-character commit SHA로
고정한다. unit test는 공식 대용량 데이터나 GPU 없이 fixture로 실행되므로 test 통과는 전체 데이터
GPU 품질·속도 결과가 생성됐다는 뜻이 아니다.

## 15. 현재 한계와 교차검증 체크리스트

2026-08-31까지 로컬 검증에서는 전체 데이터 CUDA 본실험과 A6000/A100 profile/benchmark를 실행하지 않았다.
같은 날짜의 사용자 제공 서버 로그는 전체 decode, 이전 profile 뒤 학습 진입과 첫 step의 AMP 실패,
그리고 해당 첫 샘플의 scale별 backward 진단까지 확인한다. 현재 수정된 코드의 전체 GPU 실행은
아직 검증되지 않았다.
다음 항목은 실제 server에서 `scripts/run.sh`가 완료된 뒤 결과 파일로 검증해야 한다.

- EventHDR/EventAid-R 전체 decode 로그와 총 frame 수를 해당 실험 기록에 보존
- EventAid-R event/GT timestamp 기준과 pairing interval 밖 event의 의미
- A100/A6000 각각의 full topology scan, 최고 밀도 sample CUDA step과 `runs/profile.json`
- 40-epoch loss/history, 마지막 epoch internal eval과 checkpoint digest
- all-sample calibration의 layer별 valid count/dead channel
- 18개 mode/dynamics/T의 quality, latency, memory, graph와 firing-rate artifact
- A6000/A100별 driver, CUDA wheel, torch, peak memory와 walltime

알고 있어야 할 구조적 한계:

- cell search는 exact지만 dense event cell의 최악 복잡도는 여전히 `O(N²)`다.
- single-GPU, chronological batch 1, sample별 Python loop라 전체 실행 시간이 길 수 있다.
- 8,192-event cap은 메모리 안전 선택이며 고이벤트 interval 정보를 줄인다.
- EventHDR H5는 물리 scene ID가 아니며 official eval은 독립 test가 아니다.
- EventAid-R `target_offset=1`과 log tone mapping은 명시적 cross-domain 가정이다.
- `literal_eq15`의 self-feedback은 표준 rate-conversion과 수학적 긴장이 있다.
- decoder가 analog라 firing-rate/latency를 완전한 neuromorphic system 수치로 해석할 수 없다.
- EventAid-R downloader의 ZIP 검사는 미제공 공식 checksum을 대체하지 못한다. EventHDR 직접
  downloader는 공식 OneDrive metadata의 SHA-256과 받은 H5의 전체 byte hash를 비교한다.
- optional LPIPS는 core lock에 포함되지 않는다.
- 실제 sensor ingest, network transport, compression, RTL, synthesis와 power 측정은 범위 밖이다.

연구 결과를 교차검증할 때는 다음 항목을 확인한다.

1. 결과 설명이 `paper-core 기반 복원 적응` 범위를 넘어 공식 재현을 주장하는가?
2. EventHDR가 정확히 train 51/eval 19이고 EventAid-R이 정확히 14 ZIP인가?
3. H5 sequence group을 physical scene으로 잘못 해석했는가?
4. `validate_every=null`과 마지막 epoch 단 1회 internal eval이 실제 checkpoint에 기록됐는가?
5. event cap이 정확히 8,192개를 선택하고 zero-event interval을 보존하는가?
6. graph가 strict `<D`, 양방향, self-loop 없음이며 cell optimization이 pairwise reference와 같은가?
7. ANN과 SNN checkpoint type, BN fold, Eq.(6), threshold와 tensor digest가 일치하는가?
8. EventAid-R을 본 뒤 model/config를 바꾸지 않았는가?
9. 학습 checkpoint가 현재 data/source/runtime에 재검증된 CUDA preflight gate를 가지는가?
10. ANN/SNN artifact의 `report_eligible`, evaluation/benchmark protocol과 calibration seal이 유효한가?
11. 보고한 숫자가 실제 `profile.json`, `metrics.json`, `benchmark.json`, `history.json`과 server
    provenance에 있는가?

이 열한 항목 중 하나라도 확인되지 않으면 해당 수치는 예비 내부 결과로만 취급한다.
