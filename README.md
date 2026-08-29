# ASGCN paper-core 기반 전체 Event-to-Frame 실험

[![CI](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml)

EventHDR 전체 공개 배포본으로 학습하고 EventHDR 공식 eval과 EventAid-R 전체에서 평가하는
event-to-frame 연구 코드다. MobaXterm으로 Linux GPU 서버에 SSH 접속한 뒤 clone, 설치, 데이터
구축, 학습, ANN→SNN 보정, 전체 평가를 한 번에 재현할 수 있다.

기본 실험 범위는 다음과 같다.

- EventHDR `train/1.h5`–`51.h5` 전체로 40 epoch ANN 학습
- EventHDR `eval/1.h5`–`19.h5` 전체로 마지막 epoch에 한 번 내부 평가
- EventHDR train의 모든 calibration sample로 BN folding·parameter normalization 수행
- EventHDR eval과 EventAid-R 14개 ZIP 전체에서 ANN 평가
- 두 SNN dynamics(`literal_eq15`, `standard_if`)를 각각 `T=4,8,16,32`로 전체 평가
- 전체 품질 지표와 별도의 compute-only latency benchmark 기록

## 구현 범위와 모델 구조

이 저장소는 ASGCN 저자의 공식 코드를 그대로 실행한 완전 재현본이 아니다. [AAAI 공식
논문](https://ojs.aaai.org/index.php/AAAI/article/view/32154)에 공개된 uniform sampling, radius
graph, B-spline graph convolution, BN folding, parameter normalization과 IF 식을 직접 복원하고,
원 논문의 분류 head를 이 과제의 프레임 복원 head로 바꾼 **ASGCN paper-core 기반 확장**이다.
논문에 공개되지 않은 spline·좌표·threshold 결합 세부값은 config에 고정한 구현 가정이다. 근거와
가정의 경계는 [docs/ASGCN.md](docs/ASGCN.md)에 정리했다.

```text
events [N, x, y, t, p]
  -> full-frame 좌표 정규화
  -> max_events 초과 시 정확히 max_events개를 균일 index로 결정론적 선택
  -> ASGCN sampling factor R
  -> undirected radius graph
  -> B-spline graph encoder (ANN 또는 IF-SNN)
  -> feature rasterization
  -> residual U-Net + analog ConvGRU
  -> luminance frame
```

SNN 변환 대상은 graph encoder다. residual U-Net과 ConvGRU decoder는 ANN/SNN 평가 모두에서 analog
연산으로 남는다. `literal_eq15`는 논문 식 (15)의 self-feedback 항을 문자 그대로 실행하고,
`standard_if`는 그 항을 제거한 rate-conversion 대조군이다. 후자를 공식 저자 설정이라고 주장하지
않는다.

기본 config는 모든 파일과 모든 frame sample을 사용하며 `crop_size: null`, `frame_stride: 1`,
`max_train_samples: null`, `max_val_samples: null`, `eval.max_samples: null`이다. 다만 그래프 메모리를
제어하기 위해 한 frame의 crop 후 event가 8,192개를 넘으면 시작·끝을 보존하는 균일 index로 정확히
8,192개를 선택한다. 이는 ASGCN 논문의 고정 sampling factor `R`과 구분해 결과 metadata에 기록되는
복원 시스템용 안전 제한이다.

## 데이터와 실험 역할

| 데이터 | 공개 파일 | 용량 | 이 저장소의 역할 |
|---|---:|---:|---|
| EventHDR train | H5 51개 | EventHDR 합계 약 25.72GB | ANN 학습 및 ANN→SNN 보정 |
| EventHDR eval | H5 19개 | 위 합계에 포함 | 마지막 epoch 내부 평가와 최종 평가 |
| EventAid-R | ZIP 14개 | 약 24.68GB | 학습·보정에 쓰지 않는 외부 평가 |

두 데이터의 합계는 약 **50.4GB**로 100GB 미만이다. 가상환경, checkpoint, prediction, 로그와
EventHDR 업로드용 ZIP을 동시에 보관하면 추가 공간이 필요하다.

EventHDR 공식 배포는 train 51 H5와 eval 19 H5를 서로 다른 root로 제공한다. 공개 자료에는 이
70개 H5와 실제 촬영 physical scene 사이의 완전한 대응표가 없다. 따라서
`manifests/eventhdr_split.json`은 공식 train/eval root를 그대로 고정하며 H5 sequence file을 recurrent
state와 macro metric의 group 단위로 사용한다. 확인되지 않은 physical-scene 대응을 만들어내지
않는다.

`configs/hdr_train.json`의 `validate_every: null`은 EventHDR eval을 **40번째(마지막) epoch에 단 한 번만**
평가한다는 뜻이다. eval loss는 gradient에 들어가지 않고 여러 epoch 중 checkpoint를 고르는 데도
사용되지 않는다. 호환성을 위해 파일명은 `best.pt`지만, 이 설정에서는 마지막 epoch에서 한 번
계산한 finite macro SSIM을 가진 모델이다. EventAid-R은 그 이후 외부 일반화 평가에만 사용한다.

두 loader는 target을 `[0,1]` luminance로 만든 뒤 동일한
`log1p(5000*x)/log1p(5000)` tone mapping을 적용한다. EventAid-R은 event block `i`를 다음 GT
`i+1`과 짝짓는 `target_offset: 1`을 사용한다. 이는 이 저장소의 정렬 가정이지 ASGCN 논문 값은
아니다.

## MobaXterm/Linux GPU 서버 재현

MobaXterm은 SSH/SFTP 접속에만 사용한다. 아래 명령은 접속한 Linux 서버의 저장소 root에서
실행한다.

### 1. 비공개 저장소 clone

서버 SSH key가 GitHub 계정에 등록되어 있으면:

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction
```

HTTPS를 사용하면 GitHub 계정 비밀번호가 아니라 private repository 접근 token이 필요하다.

```bash
git clone https://github.com/costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction
```

### 2. Python/CUDA 환경 설치

먼저 서버 상태를 확인한다.

```bash
python3.12 --version
nvidia-smi
ldd --version | head -n 1
curl --version | head -n 1
```

재현 프로필은 Python 3.12, `constraints/py312.txt`, PyTorch 2.13.0을 사용한다. Linux의 locked
PyTorch profile은 glibc 2.28 이상을 요구하며 설치 스크립트가 이를 먼저 검사한다. 서버 NVIDIA
driver에 맞는 CUDA wheel index는 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)
에서 확인한다.

```bash
cp .env.example .env
# .env의 PYTHON_BIN과 필요 항목을 서버에 맞게 확인한다.
# 아래 값은 공식 선택기에서 확인한 실제 URL로 바꾼다.
export TORCH_INDEX_URL='<official PyTorch CUDA wheel index URL>'

bash scripts/setup.sh
source .venv/bin/activate
python scripts/check_env.py --lock constraints/py312.txt
python -m pip check
python -m pytest -q
```

로그인 node가 GPU를 숨기는 클러스터에서는 설치 시 `.env`의 `REQUIRE_CUDA=0`을 유지해도 된다.
학습 전에는 반드시 실제 GPU allocation 안에서 다음 검사를 통과시킨다.

```bash
source .venv/bin/activate
python scripts/check_env.py --require-cuda --lock constraints/py312.txt
```

### 3. EventAid-R 전체 자동 다운로드

EventAid-R은 공식 manifest의 14개 ZIP을 내려받고 ZIP container를 검증한다. 압축을 풀 필요가 없으며
loader가 ZIP을 직접 읽는다.

```bash
source .venv/bin/activate
bash scripts/get_aid.sh --all
```

중단되면 같은 명령을 다시 실행한다. 유효한 ZIP은 유지하고 `curl --continue-at -`로 미완료 파일을
이어받는다.

### 4. EventHDR 전체 가져오기

EventHDR는 [공식 저장소](https://github.com/yunhao-zou/EventHDR)의
[공식 OneDrive 배포 폴더](https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N)에서
받는다. OneDrive 공개 폴더가 무인 `curl` 요청을 거부하므로 가짜 자동 downloader를 두지 않았다.
브라우저로 받은 파일을 MobaXterm SFTP로 서버에 올린 뒤 아래 세 방법 중 하나를 사용한다.

train/eval을 각각 ZIP으로 받았다면:

```bash
bash scripts/get_hdr.sh --archive /path/to/train.zip --split train
bash scripts/get_hdr.sh --archive /path/to/eval.zip --split eval
```

train/eval을 함께 포함한 하나의 ZIP을 받았다면:

```bash
bash scripts/get_hdr.sh --archive /path/to/EventHDR.zip
```

이미 압축을 풀었거나 서버 shared storage에 있다면 복사하거나 symlink한다.

```bash
bash scripts/get_hdr.sh --source /path/to/EventHDR
# 대용량 복제를 피하려면:
bash scripts/get_hdr.sh --source /shared/path/EventHDR --link
```

각 split만 따로 존재하는 source에는 `--split train` 또는 `--split eval`을 함께 준다. importer는
정확히 train `1.h5`–`51.h5`, eval `1.h5`–`19.h5`인지, 예상 밖 H5가 없는지, 각 파일이 HDF5인지,
합계가 100GB 미만인지 검사한다. 기존의 다른 파일을 덮어쓰지 않으며 복사는 `.part` 임시 파일 뒤
원자적으로 완료된다.

최종 배치는 다음과 같다.

```text
data/
├── EventHDR/
│   ├── train/1.h5 ... 51.h5
│   └── eval/1.h5 ... 19.h5
└── EventAid-R/
    └── R-*.zip                  # 14개
```

가져온 EventHDR와 두 데이터의 전체 coverage를 확인한다.

```bash
bash scripts/get_hdr.sh --check
python scripts/check_env.py --require-full-data --lock constraints/py312.txt
```

### 5. 전체 실험 한 번에 실행

GPU allocation 안에서 다음 한 명령을 실행한다.

```bash
source .venv/bin/activate
mkdir -p logs
bash scripts/full.sh 2>&1 | tee logs/full.log
```

MobaXterm 연결이 끊겨도 계속 실행하려면 `tmux` 안에서 시작한다.

```bash
tmux new-session -s asgcn
source .venv/bin/activate
mkdir -p logs
bash scripts/full.sh 2>&1 | tee logs/full.log
```

분리는 `Ctrl-b`, `d`, 재접속은 `tmux attach -t asgcn`이다.

`scripts/full.sh`은 다음 순서를 fail-fast로 실행한다.

1. CUDA, locked dependencies, EventHDR 51+19 H5, EventAid-R 14 ZIP coverage 검사
2. EventHDR train/eval과 EventAid-R의 모든 sample을 실제 decode하고 event/target 구조 검사
3. EventHDR train 전체 40 epoch ANN 학습
4. EventHDR train의 모든 calibration sample로 `best_snn.pt` 생성
5. EventHDR eval과 EventAid-R 전체에 대해 ANN 1회 및
   `literal_eq15`/`standard_if` × `T=4,8,16,32` 평가·benchmark

전체 품질 평가는 모든 sample을 사용한다. 기본 `BENCHMARK_STEPS=100`은 GPU compute latency를 재는
별도 timing 반복 수이며 품질 평가를 100 sample로 줄이는 설정이 아니다. `save_predictions: 20`도
PNG 저장 수만 제한하고 metric 계산 범위는 제한하지 않는다.

실행될 명령표만 확인하려면 데이터와 GPU 없이 다음을 사용할 수 있다.

```bash
DRY_RUN=1 bash scripts/full.sh
```

## 중단 후 재개와 결과 보호

학습은 매 epoch 종료 시 `last.pt`를 원자적으로 갱신한다. 중단 후 같은 run을 epoch 경계에서 재개한다.

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/full.sh 2>&1 | tee -a logs/full.log
```

resume 시 model, optimizer, scheduler, AMP scaler, RNG, history뿐 아니라 config, 상대 data identity,
전체 data SHA-256, source tree hash와 GPU protocol을 교차검증한다. 일치하지 않으면 조용히 다른 실험을
이어 붙이지 않고 중단한다.

기존 calibration 또는 평가 결과도 자동 덮어쓰지 않는다. 의도적으로 calibration만 다시 만들 때는
`OVERWRITE_CALIBRATION=1`을 명시할 수 있지만, 이미 완료된 평가 artifact는 보존하거나 config의
`eval.output_dir`를 새 경로로 바꿔 별도 run으로 실행해야 한다.

주요 출력은 다음과 같다.

```text
runs/eventhdr_asgcn/
├── config.json
├── history.json
├── last.pt                       # 학습 재개용 전체 상태
├── best.pt                       # ANN inference용 clean checkpoint
└── best_snn.pt                   # 보정된 SNN graph encoder checkpoint

runs/eventhdr_official_eval_ann/ann/
runs/eventhdr_official_eval_snn/snn_literal_eq15_T{4,8,16,32}/
runs/eventhdr_official_eval_snn/snn_standard_if_T{4,8,16,32}/
runs/eventaid_r_external_ann/ann/
runs/eventaid_r_external_snn/snn_literal_eq15_T{4,8,16,32}/
runs/eventaid_r_external_snn/snn_standard_if_T{4,8,16,32}/
```

각 평가 폴더에는 `metrics.json`, `frames.csv`, `predictions/`, `benchmark.json`이 생긴다. 품질 지표는
PSNR, Gaussian SSIM, RMSE, temporal L1과 micro/H5-or-ZIP-group macro 집계를 포함한다. benchmark는
data I/O와 host-to-device 이동을 제외한 model compute latency, FPS, percentile, graph 처리량, spike
rate와 GPU peak memory를 기록한다. PyTorch GPU latency를 FPGA/ASIC latency나 에너지로 해석하면 안
된다.

## SLURM/PBS scheduler

클러스터 batch job은 저장소 root에서 제출한다. SLURM에서 학습→전체 calibration 의존성은 다음과
같이 건다.

```bash
train_id=$(sbatch --parsable --export=ALL,PROJECT_ROOT="$PWD" server/train.sbatch)
cal_id=$(sbatch --parsable --dependency=afterok:${train_id} \
  --export=ALL,PROJECT_ROOT="$PWD" server/calibrate.sbatch)
```

ANN 두 평가와 SNN 전체 행렬을 dependency로 제출한다.

```bash
for config in configs/hdr_ann.json configs/aid_ann.json; do
  sbatch --dependency=afterok:${train_id} \
    --export=ALL,PROJECT_ROOT="$PWD",CONFIG_PATH="$config",CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt,INFERENCE_MODE=ann \
    server/eval.sbatch
done

for config in configs/hdr_snn.json configs/aid_snn.json; do
  for dynamics in literal_eq15 standard_if; do
    for timestep in 4 8 16 32; do
      sbatch --dependency=afterok:${cal_id} \
        --export=ALL,PROJECT_ROOT="$PWD",CONFIG_PATH="$config",CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS="$dynamics",SIMULATION_STEPS="$timestep" \
        server/eval.sbatch
    done
  done
done
```

학습 resume job에는 `RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt"`를 `--export`에 추가한다.
PBS/Torque용 동등 wrapper는 `server/train.pbs`, `server/calibrate.pbs`, `server/eval.pbs`이며
`qsub -W depend=afterok:<job-id>`로 같은 의존성을 건다. `#SBATCH`/`#PBS`의 GPU, memory, walltime,
queue, account는 학교 scheduler 정책에 맞게 조정해야 한다. 자세한 서버 운용은
[docs/SERVER.md](docs/SERVER.md), 실험 정의는 [docs/EXPERIMENT.md](docs/EXPERIMENT.md)에 있다.

## 현재 검증 상태와 한계

- 코드의 unit/integration test와 Linux 의존성 검사는 구성되어 있지만, EventHDR+EventAid-R 전체
  실데이터를 사용한 CUDA 학습·전체 행렬 실행, A6000/A100 peak memory·runtime·latency 실측은 아직
  수행하지 않았다. 따라서 README는 실행 절차를 보장하는 코드 경로이지 측정 완료 보고서가 아니다.
- 공개 자료에 EventHDR H5↔physical-scene 완전 대응표가 없어 공식 train/eval root 이상의
  scene-disjoint 주장을 하지 않는다.
- 원 논문의 동적 asynchronous K-hop update, pooling/classifier, energy model은 포함하지 않는다.
- 반도체 RTL/FPGA/ASIC, event compression/transport, 실제 전력·에너지 측정은 후속 과제 범위다.
- recurrent batch size는 1이고 resume granularity는 epoch 단위다. 전체 실행 시간과 저장 공간은
  서버 GPU, filesystem, dataset decode 속도에 따라 달라진다.

코드 전체 스냅샷은 [code_summary.md](code_summary.md), 인수인계와 연구상 주의점은
[hand_off.md](hand_off.md)를 참조한다.
