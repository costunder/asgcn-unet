# ASGCN-inspired Event-to-Frame 실험

[![CI](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml)

EventHDR로 학습하고 EventHDR 공식 eval과 EventAid-R에서 평가하는 event-to-frame 연구
프로토타입이다. GitHub clone 후 MobaXterm/SSH로 접속한 Linux GPU 서버에서 설치·검사·학습·평가할
수 있게 구성했다.

정확한 기술 범위는 다음과 같다.

- encoder: ASGCN에서 착안한 causal edge-conditioned graph network
- decoder: residual U-Net + analog ConvGRU
- ANN 경로: 학습과 기본 평가
- `snn` 경로: 보정된 graph activation의 spike-rate 양자화 proxy
- 데이터: EventHDR H5 직접 읽기, EventAid-R ZIP 직접 읽기

원 논문의 B-spline ASGCN을 그대로 재현한 코드는 아니며, `snn` 경로도 membrane timestep을
전개하는 완전한 IF/LIF SNN이 아니다. 현재 저장소는 반도체 RTL/FPGA/ASIC, 전력·에너지 측정,
이벤트 전송 protocol을 포함하지 않는다.

## 데이터와 실험 역할

| 단계 | 데이터 | 목적 |
|---|---|---|
| ANN 학습 | EventHDR train | 복원 weight 최적화 |
| 모델 선택 | EventHDR holdout | 물리 scene split 확정 후 file-balanced macro SSIM 기준 `best.pt` 선택 |
| SNN 보정 | EventHDR train | file-balanced BN folding·threshold 계산 |
| 내부시험 | EventHDR 공식 eval | 학습 완료 후 고정 평가 |
| 외부시험 | EventAid-R 14장면 | 학습·보정 없이 일반화 평가 |

두 데이터의 target은 모두 `[0,1]` luminance에 동일한
`log1p(5000*x)/log1p(5000)` 변환을 적용한다. 이는 수치 output domain을 맞추는 조치이며,
서로 다른 센서의 radiometric response가 동일하다는 뜻은 아니다.

```text
events [N, x,y,t,p]
  -> bounded causal graph
  -> ASGCN-inspired graph encoder (ANN 또는 calibrated rate proxy)
  -> feature rasterization
  -> residual U-Net + ConvGRU
  -> luminance frame
  -> PSNR / Gaussian SSIM / RMSE / temporal_l1 / latency
```

## 본학습 전 차단 장치

- recurrent validation은 file/scene별 가능한 한 균등한 quota의 연속 window를 채점한다. 본학습은
  최대 500개, smoke는 32개를 채점하며, window 앞 group당 최대 64 frame(smoke는 8)을 별도로
  replay해 ConvGRU 상태를 예열한다. non-recurrent validation에는 이 context replay가 없다.
- calibration은 각 파일 시간축에서 등간격으로 뽑는다. recurrent benchmark는 균형 연속 window와
  group당 최대 32개의 unmeasured predecessor를 쓰고, non-recurrent benchmark는 time-spread sample을 쓴다.
- 장면·프레임 연속 번호·해상도 중 하나라도 끊기면 recurrent state와 temporal metric을 초기화한다.
- checkpoint 선택은 frame 수가 많은 scene에 치우치는 micro 값이 아니라 macro SSIM을 사용한다.
- EventHDR/EventAid-R의 timestamp, event index, 좌표, polarity와 배열 구조를 검사한다.
- `snn` 평가는 `simulation_steps >= 1`, BN folding, 1개 이상의 calibration sample을 강제한다.
- 보정 checkpoint에는 optimizer/scaler/RNG/history를 넣지 않아 training resume과 분리한다.
- checkpoint는 CPU에서 읽고 model만 GPU로 옮겨 평가 peak memory에서 optimizer state를 제외한다.
- `best.pt`는 ANN inference용 clean checkpoint이고 optimizer/RNG/history는 `last.pt`에만 둔다.
- exact resume의 checkpoint protocol은 절대 root/mtime 대신 상대 파일 identity와 train/validation
  원본 전체의 SHA-256, 기존 `best.pt`를 검증한다. 계산한 hash의 재사용 정책은 아래에 별도로 적었다.
- 임시 파일 번호 split으로 40 epoch 본학습을 실행하지 못하게 막는다. 물리 scene 분할을 확정하고
  manifest의 `status`를 `final`로 바꿔야 한다.

## MobaXterm/Linux GPU 서버 실행

MobaXterm은 SSH/SFTP 클라이언트다. 아래 명령은 접속한 Linux 서버의 저장소 루트에서 실행한다.

### 1. 비공개 저장소 clone

서버 SSH key를 GitHub에 등록했다면:

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction
```

HTTPS를 쓰면 GitHub 비밀번호 대신 접근 token이 필요하다.

```bash
git clone https://github.com/costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction
```

### 2. 서버와 CUDA wheel 확인

```bash
git --version
python3.12 --version
python3.12 -c "import venv, ensurepip; print('venv/ensurepip: OK')"
curl --version | head -n 1
tmux -V
nvidia-smi
ldd --version | head -n 1
```

`venv`/`ensurepip`은 가상환경 생성에, `curl`은 EventAid-R downloader에 필요하다. `tmux`는 직접
GPU 서버에서 장시간 학습할 때만 필요하며, scheduler만 쓰는 서버에서는 생략할 수 있다. 명령이
없으면 서버 관리자에게 해당 모듈 또는 OS package 이름을 확인한다.

기본 재현 프로필은 다음을 고정한다.

- Python 3.12
- torch 2.13.0
- `constraints/py312.txt`의 NumPy/h5py/Pillow/tqdm/dev 의존성
- CUDA build는 서버 드라이버와 호환되는 공식 PyTorch wheel index

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 **torch 2.13.0을 실제로
제공하는** CUDA index를 선택한다. 예시 URL을 그대로 복사하지 말고 해당 서버에서 확인한다.

```bash
cp .env.example .env
read -r -p "Official PyTorch wheel index URL: " TORCH_INDEX_URL
export TORCH_INDEX_URL

bash scripts/setup.sh
source .venv/bin/activate
python scripts/check_env.py --lock constraints/py312.txt
python -m pip check
python -m pytest -q
```

명령행에서 export한 값은 `.env`보다 우선한다. 서버에 Python 3.12가 없다면 `py312` constraints를
억지로 사용하지 말고, 서버의 Python/CUDA 조합에서 별도 lock을 검증한 뒤 설정해야 한다.

`PROJECT_EXTRAS=dev`가 기본이다. LPIPS는 기본 평가에 필요하지 않으며, 사용할 때만 torch와 맞는
torchvision을 먼저 확인한 뒤 `python -m pip install -e '.[eval]'`로 추가한다.

로그인 노드에서 GPU가 숨겨지는 클러스터도 있다. 실제 GPU allocation 안에서는 반드시:

```bash
python scripts/check_env.py --require-cuda --lock constraints/py312.txt
```

을 통과시킨다.

### 3. 데이터 배치

```text
data/
├── EventHDR/
│   ├── train/1.h5 ... 51.h5
│   └── eval/*.h5                 # 공식 eval 19개
└── EventAid-R/
    └── R-*.zip                   # 전체 14개 scene
```

EventHDR는 [공식 저장소](https://github.com/yunhao-zou/EventHDR)의 배포 링크에서 받은 뒤
MobaXterm SFTP 또는 shared storage symlink로 배치한다. 자동 downloader는 제공하지 않는다.

EventAid-R은 작은 `R-bear`로 먼저 loader를 확인하고, 최종 외부평가 전에만 전체를 받는다.

```bash
bash scripts/get_aid.sh R-bear
bash scripts/get_aid.sh --all
```

ZIP은 압축을 풀지 않는다. 두 데이터 합계 약 50.4GB이며 가상환경·checkpoint·결과를 포함해
70GB 이상의 여유 공간을 권장한다.

파일 수와 manifest 누락을 확인한다.

```bash
python scripts/check_env.py --require-full-data
```

빠른 구조 검사:

```bash
asgcn-recon inspect --config configs/hdr_train.json --samples 2
asgcn-recon inspect --config configs/hdr_ann.json --samples 2
asgcn-recon inspect --config configs/aid_ann.json --samples 2
```

본실험 전에는 모든 selected sample을 실제로 decode해 좌표·timestamp까지 검사한다. 시간이 걸리는
의도적인 전체 검사다.

```bash
asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all
```

### 4. scene split 확정

`manifests/eventhdr_split.json`은 현재 `status: provisional`이다. 파일 번호만으로 나눈 임시
holdout이므로 본학습용이 아니다. 동일 물리 장면의 파일을 한 group으로 묶어 train/validation이
겹치지 않게 두 목록을 수정하고 다음 값을 바꾼다.

```json
"status": "final"
```

공식 scene 대응표를 확보하지 못한 상태에서 임의로 `final`로 바꾸면 안 된다. 현재 상태에서
`configs/hdr_train.json`을 실행하면 의도적으로 중단된다.

### 5. A100/A6000 real-data smoke

`configs/hdr_smoke.json`은 임시 split 사용을 명시적으로 허용하는 최대 32 train sample과 32 scored
validation sample, 1 epoch 점검용이다. validation에는 group당 최대 8개의 unscored context frame이
추가될 수 있다. 결과는 논문 성능으로 보고하지 않는다.

```bash
mkdir -p logs
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt
bash scripts/train.sh configs/hdr_smoke.json 2>&1 | tee logs/smoke.log
```

`runs/smoke/history.json`의 `gpu_memory.peak_allocated_mib`와 `peak_reserved_mib`를 확인한다. 먼저
A100 10GB에서 기본 설정 그대로 통과시킨 뒤 A6000에서 본학습 예상 시간을 재는 순서가 안전하다.
이 저장소에서는 아직 두 GPU의 실측 결과를 제공하지 않는다.

### 6. EventHDR ANN 본학습

scene split이 `final`이고 smoke가 통과한 뒤 실행한다.

```bash
tmux new-session -s asgcn -c "$PWD" \
  "bash -lc 'source .venv/bin/activate && bash scripts/train.sh configs/hdr_train.json'"
```

분리: `Ctrl-b`, `d`

재접속: `tmux attach -t asgcn`

epoch 단위 resume:

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/train.sh configs/hdr_train.json
```

출력은 `runs/eventhdr_asgcn/{config.json,history.json,best.pt,last.pt}`다.
`--resume` 없이 시작할 때 이 폴더에 기존 결과가 있으면 덮어쓰지 않고 중단하므로, 기존 run을
재개하거나 `output.run_dir`가 다른 config를 사용한다.

### 7. ANN → rate-proxy 보정

EventHDR train의 여러 파일과 시간대를 균형 있게 사용한다. EventAid-R은 보정에 사용하지 않는다.

```bash
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500
```

### 8. 내부·외부 평가

```bash
INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/hdr_ann.json runs/eventhdr_asgcn/best.pt \
  2>&1 | tee logs/hdr_ann.log

INFERENCE_MODE=snn SIMULATION_STEPS=16 RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/hdr_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/hdr_snn.log

INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/aid_ann.json runs/eventhdr_asgcn/best.pt \
  2>&1 | tee logs/aid_ann.log

INFERENCE_MODE=snn SIMULATION_STEPS=16 RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/aid_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/aid_snn.log
```

`evaluate`는 end-to-end model forward latency를, `benchmark`는 데이터 I/O와 host-to-device 이동을
제외한 model compute latency를 기록한다. rate proxy는 timestep loop가 아니므로 T에 따른 GPU
latency를 neuromorphic 하드웨어 latency나 에너지로 해석하면 안 된다.

## Scheduler 실행

SLURM도 저장소 루트에서 제출한다. compute node에서는 Slurm spool 사본이 실행될 수 있으므로
스크립트가 `SLURM_SUBMIT_DIR` 또는 명시적인 `PROJECT_ROOT`를 확인한다.

```bash
sbatch server/train.sbatch
sbatch --export=ALL,PROJECT_ROOT="$PWD",CONFIG_PATH=configs/aid_ann.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt server/eval.sbatch
```

PBS/Torque는 저장소 루트에서 제출한다. `select/ngpus/queue/account` 명칭은 학교 설정에 맞춰
`server/*.pbs` 헤더를 수정해야 한다.

```bash
qsub server/train.pbs
qsub -v PROJECT_ROOT="$PWD",CONFIG_PATH=configs/aid_snn.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,\
SIMULATION_STEPS=16 server/eval.pbs
```

학교 wrapper가 `ssai_agpu -g=1`처럼 interactive allocation을 제공하면 allocation 안에서 일반
Bash 명령을 실행한다.

```bash
ssai_agpu -g=1
source .venv/bin/activate
python scripts/check_env.py --require-cuda --lock constraints/py312.txt
bash scripts/train.sh configs/hdr_smoke.json
```

자세한 내용은 [서버 가이드](docs/SERVER.md)와 [실험 프로토콜](docs/EXPERIMENT.md)에 있다.

## 결과와 지표

평가 폴더에는 `metrics.json`, `frames.csv`, `predictions/`가 생긴다.

- 품질: PSNR, Gaussian-window SSIM, RMSE, 선택적 LPIPS
- temporal: 같은 scene·해상도의 연속 sequence frame만 사용하는 `temporal_l1`
- 집계: micro, scene macro, per-scene
- 지연: mean, p50/p90/p95/p99, FPS, RTF, deadline miss ratio
- graph: events/s, node/edge 수, rate-proxy firing rate
- GPU: peak allocated/reserved memory

첫 학습 시 EventHDR train/validation 원본 전체를 읽어 SHA-256을 계산한다. 같은 경로의 resume은 run
폴더의 `.data_hash_cache.json`에서 size/mtime/ctime이 모두 같은 파일의 기존 full hash를 재사용한다.
절대경로나 filesystem mtime/ctime은 checkpoint protocol에 들어가지 않으며, 데이터 경로가 달라도
상대 파일 identity와 byte가 같으면 재개할 수 있다. 원본을 교체·복원했거나 다시 전수 hash하려면 config에서
`train.rehash_data=true`로 바꾼다.

SSIM 구현은 11×11, σ=1.5 Gaussian valid window를 사용한다. 기존 논문과 수치를 직접 비교할 때는
그 논문의 crop, border, color space, data range, SSIM package까지 동일하게 맞춰야 한다.

## Windows 개발

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints\py312.txt -e ".[dev]"
.\.venv\Scripts\python.exe scripts\check_env.py --lock constraints\py312.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\get_aid.ps1 -Destination .\data\EventAid-R -Scenes R-bear
.\.venv\Scripts\asgcn-recon.exe inspect --config configs\aid_ann.json --samples 2
```

`py -3.12`가 없으면 Python 3.12를 설치하거나 해당 interpreter의 절대 경로로 venv를 만든다.

## 남은 연구 한계

- EventHDR 물리 scene 대응표를 확보해 provisional manifest를 확정해야 한다.
- 전체 공식 데이터와 A100/A6000에서 full training·CUDA AMP·peak memory·latency를 아직 실측하지
  않았다.
- recurrent batch size는 1이고 frame마다 state를 detach하므로 GPU 활용률과 장기 BPTT가 제한된다.
- graph 후보 생성은 vectorized됐지만 decoder와 sample 처리는 여전히 serial하다.
- LPIPS의 CUDA/torchvision 조합과 공식 metric implementation은 별도 고정이 필요하다.
- 실제 timestep SNN, 연산량/에너지 모델, event compression/transport, accelerator hardware는 후속
  구현 범위다.

코드 전체 스냅샷과 더 세부적인 인수인계는 `code_summary.md`, `hand_off.md`를 참조한다.
