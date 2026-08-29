# ASGCN paper-core 기반 Event-to-Frame 실험

[![CI](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml)

EventHDR로 학습하고 EventHDR 공식 eval과 EventAid-R에서 평가하는 event-to-frame 연구
프로토타입이다. GitHub clone 후 MobaXterm/SSH로 접속한 Linux GPU 서버에서 설치·검사·학습·평가할
수 있게 구성했다.

정확한 기술 범위는 다음과 같다.

- encoder: uniform event sampling + undirected radius graph + B-spline graph convolution
- decoder: residual U-Net + analog ConvGRU
- ANN 경로: 학습과 기본 평가
- `snn` 경로: BN folding·parameter normalization 뒤 논문 식 (15)–(17)을 timestep별로 전개하는 IF graph encoder
- 데이터: EventHDR H5 직접 읽기, EventAid-R ZIP 직접 읽기

graph/SNN core는 AAAI 논문에 공개된 sampling, radius graph, B-spline aggregation, BN folding,
ANN→SNN normalization과 IF membrane 식을 구현한다. 다만 2026-08-29 기준 공개 배포된 저자
코드는 확인할 수 없었고, 논문에 없는 spline·좌표 세부값은 명시적 가정이며, 원 논문의 분류 head
대신 복원 decoder를 붙였다.
논문의 layer-wise lambda와 feature-wise threshold 결합도 단일하게 규정되지 않아 이 저장소는
feature-wise lambda를 적용하고 정규화 뒤 unit threshold를 쓰는 선택을 명시했다.
따라서 “공식 ASGCN 완전 재현”이 아니라 paper-core 기반 연구 프로토타입이다. 근거와 경계는
[ASGCN 구현 범위](docs/ASGCN.md)에 정리했다. 반도체 RTL/FPGA/ASIC, 전력·에너지 측정, 이벤트 전송
protocol은 포함하지 않는다.

기본 `snn_dynamics=literal_eq15`는 논문의 `+h(t-1)` self-feedback까지 그대로 실행한다. 이 식은
표준 ANN→SNN rate-conversion IF와 수학적으로 맞지 않는 공개 모호성이 있으므로, 저장소는 이를
숨기지 않고 `standard_if` 대조군과 장기-timestep 회귀 테스트를 함께 둔다. 기본 결과를 단순히
“ANN activation을 정확히 근사한 firing rate”라고 주장하지 않는다.

## 데이터와 실험 역할

| 단계 | 데이터 | 목적 |
|---|---|---|
| ANN 학습 | EventHDR train | 복원 weight 최적화 |
| 모델 선택 | EventHDR holdout | 최종 physical-scene/provisional file group-balanced macro SSIM 기준 `best.pt` 선택 |
| SNN 보정 | EventHDR train | 같은 group 기준 BN folding·feature threshold·parameter normalization |
| 내부시험 | EventHDR 공식 eval | 학습 완료 후 고정 평가 |
| 외부시험 | EventAid-R 14장면 | 학습·보정 없이 일반화 평가 |

두 데이터의 target은 모두 `[0,1]` luminance에 동일한
`log1p(5000*x)/log1p(5000)` 변환을 적용한다. 이는 수치 output domain을 맞추는 조치이며,
서로 다른 센서의 radiometric response가 동일하다는 뜻은 아니다.
EventAid-R은 event block `i`를 다음 GT `i+1`과 짝짓는 `target_offset=1`을 명시적 정렬
가정으로 사용한다. 이 값은 ASGCN 논문 값이 아니며 다른 정렬과 비교할 때 별도 run으로 기록한다.

```text
events [N, x,y,t,p]
  -> deterministic spatial crop
  -> adaptive integer-stride max_events cap (reconstruction/server safety)
  -> fixed factor-R sampling + undirected radius graph
  -> B-spline graph encoder (ANN 또는 calibrated literal-Eq15/standard-IF)
  -> feature rasterization
  -> residual U-Net + ConvGRU
  -> luminance frame
  -> PSNR / Gaussian SSIM / RMSE / temporal_l1 / latency
```

## 본학습 전 차단 장치

- recurrent validation은 group별(final은 physical scene, provisional은 H5 file) 가능한 한 균등한
  quota의 연속 window를 채점한다. 본학습은
  최대 500개, smoke는 32개를 채점하며, window 앞 group당 최대 64 frame(smoke는 8)을 별도로
  replay해 ConvGRU 상태를 예열한다. non-recurrent validation에는 이 context replay가 없다.
- calibration은 각 group의 index 범위에서 등간격으로 뽑는다. recurrent benchmark는 균형 연속 window와
  group당 최대 32개의 unmeasured predecessor를 쓰고, non-recurrent benchmark는 time-spread sample을 쓴다.
- calibration에서 모든 ReLU가 0인 feature는 0 또는 epsilon으로 나누지 않고 unit lambda를 사용하며,
  dead-channel 수를 checkpoint에 기록한다. 이는 논문 미공개 경우에 대한 구현 가정이다.
- gradient centralization은 spline/root에서는 마지막 output axis를, Conv/Linear에서는 첫 output
  axis를 제외한 차원에 적용한다. 이 axis 규칙도 저자 코드로 확인된 공식값이 아니다.
- 장면·프레임 연속 번호·해상도 중 하나라도 끊기면 recurrent state와 temporal metric을 초기화한다.
- checkpoint 선택은 frame 수가 많은 scene에 치우치는 micro 값이 아니라 macro SSIM을 사용한다.
- EventHDR/EventAid-R의 timestamp, event index, 좌표, polarity와 배열 구조를 검사한다.
- random crop은 안정적인 scene+source-file identity로 결정한다. 같은 연속 sequence의 모든 frame은
  동일한 sensor ROI를 써 ConvGRU state와 temporal loss를 정렬하고, worker 수와 resume에도 동일하다.
  epoch별로 ROI가 바뀌는 증강은 아니다.
- `snn` 평가는 `simulation_steps >= 1`, BN folding, parameter normalization과 모든 graph layer에서
  최소 1개의 유효한 non-empty calibration observation을 강제한다.
- SNN checkpoint metadata와 각 graph layer의 persistent BN-fold/Eq. (6) flag, 변환 뒤 정확한 unit
  threshold를 교차검증한다.
- `last.pt`, `best.pt`, `best_snn.pt`는 모두 model tensor byte의 SHA-256을 저장하며, 평가·보정·resume
  전에 digest를 다시 계산해 finite 값의 조용한 변조도 거부한다.
- radius graph가 `max_graph_edges=2,000,000`을 넘으면 edge를 몰래 버리지 않고 OOM 전에 실패한다.
- Eq. (6)으로 변환한 `best_snn.pt`를 ANN 모드로 읽는 것도 거부한다. ANN에는 변환 전 `best.pt`를 쓴다.
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
- Linux에서 이 locked torch 2.13.0 profile은 glibc 2.28 이상

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

`scripts/setup.sh`은 locked torch 2.13.0과 Linux glibc 2.28 미만 조합을 venv 생성·download 전에
중단하고, `scripts/check_env.py --lock constraints/py312.txt`도 같은 조건을 fail-fast한다. 해당
서버에서는 무작정 source build하지 말고 더 최신인 학교 module 또는 container를 사용한다.

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
asgcn-recon inspect --config configs/aid_smoke.json --samples 2 --validate-all

# 최종 14-scene 외부평가 직전에만 실행
bash scripts/get_aid.sh --all
```

`aid_smoke.json`은 일부 ZIP을 허용하는 비보고용 loader 점검 설정이다. `aid_ann.json`과
`aid_snn.json`은 manifest의 정확한 14개 ZIP이 모두 없으면 중단한다.

ZIP은 압축을 풀지 않는다. 두 데이터 합계 약 50.4GB이며 가상환경·checkpoint·결과를 포함해
70GB 이상의 여유 공간을 권장한다.

최종 내부·외부 평가 전에 전체 파일 수와 manifest 누락을 확인한다.

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

`manifests/eventhdr_split.json`은 현재 legacy `train_files`/`val_files`만 가진
`status: provisional` 임시 holdout이라 본학습용이 아니다. 최종 manifest는 동일 물리 장면의 H5를
하나의 group으로 묶고, 겹치지 않는 scene ID를 train/validation에 배정하는 다음 schema를 써야 한다.

```json
{
  "status": "final",
  "scene_groups": {
    "night-drive": ["1.h5", "2.h5"],
    "day-drive": ["48.h5", "49.h5"]
  },
  "train_scenes": ["night-drive"],
  "val_scenes": ["day-drive"]
}
```

예시는 schema 설명일 뿐 실제 scene 대응표가 아니다. 공식 대응표를 확보하지 않은 채
`status`만 `final`로 바꾸거나 legacy 파일 목록만 유지하면 loader가 거부한다. 현재 상태에서
`configs/hdr_train.json`을 실행해도 의도적으로 중단된다. `manifests/eventhdr_smoke.json`의 legacy
파일 목록 schema는 비보고용 provisional smoke에만 허용한다.
final manifest는 `data/EventHDR/train` 아래의 모든 H5를 정확히 한 scene에 포함하고 모든 scene을
train 또는 validation에 배정해야 한다. 누락 파일, 미선언 파일, 중복 소유권이 하나라도 있으면
본학습 전에 중단한다.

### 5. A100/A6000 real-data smoke

`configs/hdr_smoke.json`은 `manifests/eventhdr_smoke.json`의 `1.h5`, `2.h5`를 train으로,
`48.h5`, `49.h5`를 validation으로 쓰는 1 epoch 점검용이다. 최대 32 train sample과 32 scored
validation sample을 쓰며, group당 최대 8개의 unscored context frame이 추가될 수 있다. 이 run의
content fingerprint는 네 smoke-manifest H5만 SHA-256으로 읽고 전체 EventHDR/EventAid-R를 hash하지
않는다. 임시 파일 split 결과는 논문 성능으로 보고하지 않는다.

smoke와 본학습은 논문의 Adam+gradient centralization, 초기 learning rate `1e-3`, L2 weight decay
`5e-3`를 쓴다. 논문이 정확한 decay epoch와 gamma를 공개하지 않아 `[20,30]`과 `0.1`은 40-epoch
복원 실험용 명시적 가정으로 고정했다. smoke는 1 epoch라 milestone에 도달하지 않는다.

```bash
mkdir -p logs
python scripts/check_env.py --require-cuda --require-eventhdr-smoke \
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

### 7. ANN → IF-SNN 보정·변환

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

INFERENCE_MODE=snn SIMULATION_STEPS=16 SNN_DYNAMICS=literal_eq15 RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/hdr_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/hdr_snn.log

# 같은 calibrated checkpoint의 standard-IF 대조군
INFERENCE_MODE=snn SIMULATION_STEPS=16 SNN_DYNAMICS=standard_if RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/hdr_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/hdr_standard_if.log

INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/aid_ann.json runs/eventhdr_asgcn/best.pt \
  2>&1 | tee logs/aid_ann.log

INFERENCE_MODE=snn SIMULATION_STEPS=16 SNN_DYNAMICS=literal_eq15 RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/aid_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/aid_snn.log
```

`evaluate`는 end-to-end model forward latency를, `benchmark`는 데이터 I/O와 host-to-device 이동을
제외한 model compute latency를 기록한다. `snn` graph encoder는 T번 IF membrane timestep을 실제로
전개하고 고정 B-spline basis는 한 번만 계산해 재사용하지만 residual U-Net과 ConvGRU는 analog다.
이 PyTorch GPU latency를 neuromorphic 하드웨어
latency나 에너지로 해석하면 안 된다.

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
SIMULATION_STEPS=16,SNN_DYNAMICS=literal_eq15 server/eval.pbs
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

각 config의 `eval.output_dir` 아래에 inference별 하위 폴더가 생긴다.

```text
<output_dir>/ann/{metrics.json,frames.csv,predictions/,benchmark.json}
<output_dir>/snn_literal_eq15_T16/{metrics.json,frames.csv,predictions/,benchmark.json}
<output_dir>/snn_standard_if_T16/{metrics.json,frames.csv,predictions/,benchmark.json}
```

같은 mode/dynamics/T의 결과가 이미 있으면 덮어쓰지 않고 중단한다. 기존 결과를 이동·보존하거나
새 `eval.output_dir`를 사용한다. `benchmark.json`은 compute-only 지표이고 나머지 artifact는
`evaluate`가 기록한다. prediction PNG 이름에는 평가 순번과 전체 sample ID hash를 넣어 서로 다른
ID의 slug 충돌과 Windows 금지 문자를 피한다.

- 품질: PSNR, Gaussian-window SSIM, RMSE, 선택적 LPIPS
- temporal: 같은 scene·해상도의 연속 sequence frame만 사용하는 `temporal_l1`
- 집계: micro, group macro, per-group. final holdout은 physical scene, provisional/EventHDR 공식
  eval은 H5 파일, EventAid-R은 `R-*.zip` scene을 group key로 쓴다. JSON의 기존 필드명은
  호환성을 위해 `macro`/`per_scene`으로 유지한다.
- 지연: mean, p50/p90/p95/p99, FPS, RTF, deadline miss ratio
- graph: raw events/s, retained events/s, graph nodes/s, 평균 node/edge, isolate 비율/max degree,
  layer별 `총 spike / 총 neuron-step` IF firing rate
- GPU: peak allocated/reserved memory

첫 학습 시 선택된 manifest의 EventHDR train/validation 원본을 읽어 SHA-256을 계산한다. 따라서
smoke는 `1.h5`, `2.h5`, `48.h5`, `49.h5`만, 본학습은 최종 `eventhdr_split.json`에 든 파일 전체를
hash한다. 같은 경로의 resume은 run 폴더의 `.data_hash_cache.json`에서 size/mtime/ctime이 모두 같은
파일의 기존 full hash를 재사용한다.
절대경로나 filesystem mtime/ctime은 checkpoint protocol에 들어가지 않으며, 데이터 경로가 달라도
상대 파일 identity와 byte가 같으면 재개할 수 있다. 원본을 교체·복원했거나 다시 전수 hash하려면 config에서
`train.rehash_data=true`로 바꾼다.

고정 평가는 coverage도 강제한다. EventHDR 공식 eval config는 H5 정확히 19개를, EventAid-R 최종
config는 `manifests/eventaid_r.json`과 이름이 일치하는 ZIP 정확히 14개를 요구한다. 일부
EventAid-R만 확인할 때는 최종 config가 아니라 `configs/aid_smoke.json`을 사용한다.

SSIM 구현은 11×11, σ=1.5 Gaussian valid window를 사용한다. 기존 논문과 수치를 직접 비교할 때는
그 논문의 crop, border, color space, data range, SSIM package까지 동일하게 맞춰야 한다.

benchmark의 `raw_events_per_second`는 crop/cap 전 source interval event 수를,
`retained_events_per_second`는 spatial crop과 adaptive `max_events` cap 뒤 수를,
`graph_nodes_per_second`는 model의 추가 고정 factor `R`까지 적용한 graph node 수를 각각 측정된
model compute time으로 나눈 값이다. 따라서 dataset read를 포함한 ingest throughput이 아니다.
`retention_ratio`는 retained/raw 합계 비율이며 `events_per_second`는 하위 호환용으로만 남긴
deprecated alias로서 항상 `retained_events_per_second`와 같다. 각 sample metadata에는
`raw_event_count`, `cropped_event_count`, `retained_event_count`, `dataset_sampling_factor`도 기록한다.

## Windows 개발

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints\py312.txt -e ".[dev]"
.\.venv\Scripts\python.exe scripts\check_env.py --lock constraints\py312.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\get_aid.ps1 -Destination .\data\EventAid-R -Scenes R-bear
.\.venv\Scripts\asgcn-recon.exe inspect --config configs\aid_smoke.json --samples 2
```

`py -3.12`가 없으면 Python 3.12를 설치하거나 해당 interpreter의 절대 경로로 venv를 만든다.

## 남은 연구 한계

- EventHDR 물리 scene 대응표를 확보해 provisional manifest를 확정해야 한다.
- 전체 공식 데이터와 A100/A6000에서 full training·CUDA AMP·peak memory·latency를 아직 실측하지
  않았다.
- recurrent batch size는 1이고 frame마다 state를 detach하므로 GPU 활용률과 장기 BPTT가 제한된다.
- radius graph 생성은 chunked all-pairs 계산이며 2,000,000 directed-edge fail-fast guard가 있어도
  worst-case 계산은 O(N²)이다. decoder와 sample 처리도 여전히 serial하다.
- LPIPS의 CUDA/torchvision 조합과 공식 metric implementation은 별도 고정이 필요하다.
- 동적 asynchronous K-hop update, 논문의 pooling/classifier, 연산량·에너지 모델,
  event compression/transport와 accelerator hardware는 후속 구현 범위다.

코드 전체 스냅샷과 더 세부적인 인수인계는 `code_summary.md`, `hand_off.md`를 참조한다.
