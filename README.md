# ASGCN-U-Net: Event-to-Frame 복원 실험

[![CI](https://github.com/costunder/asgcn-unet/actions/workflows/ci.yml/badge.svg)](https://github.com/costunder/asgcn-unet/actions/workflows/ci.yml)

EventHDR 전체 공개 배포본으로 학습하고 EventHDR 공식 eval과 EventAid-R 전체에서 평가하는
event-to-frame 연구 코드다. MobaXterm으로 Linux GPU 서버에 SSH 접속한 뒤 clone, 설치, 데이터
구축, GPU 사전검증, 학습, ANN→SNN 보정, 전체 평가를 한 번에 재현할 수 있다.

## 빠른 시작 (MobaXterm)

MobaXterm으로 **본인 Linux 서버 계정**에 접속한 뒤 아래 순서로 실행한다. Conda가 설치된 서버
기준이다. 명령이 오류로 끝나면 다음 단계로 넘어가지 않는다. GPU를 scheduler로 할당받는 서버라면
3번의 CUDA 검사와 5번 실행은 GPU allocation 안에서 한다.

### 1. GitHub 로그인 — 처음 한 번

Private 저장소를 내려받기 위한 로그인이다. SSH 키 생성·등록은 필요 없다.
`asgcn` Conda 환경이 이미 있으면 첫 줄은 생략하고 활성화만 한다. 기존 환경 삭제를 묻는다면 취소한다.

```bash
conda create -n asgcn --override-channels -c conda-forge python=3.12 gh git
conda activate asgcn
gh auth login --hostname github.com --git-protocol https --web
```

설치 확인에는 `y`, Git 인증 질문에는 `Yes`를 선택한다. 일회용 코드가 나오면 안내에 따라 Enter를
누르고, **PC 브라우저**에서 [github.com/login/device](https://github.com/login/device)를 열어 그 코드를
입력·승인한다. 서버에서 브라우저를 열지 못한다는 메시지가 나와도 PC에서 진행하면 된다.
**서버 터미널에 로그인 성공이 표시되고 프롬프트로 돌아온 뒤** 2번으로 간다.

이미 해당 GitHub 계정으로 인증했다면 `gh auth status`로 확인하고 로그인은 생략할 수 있다.
이 로그인은 저장소 전용 읽기 키보다 권한 범위가 넓다. 같은 OS 계정을 다른 사람이 사용하는 경우에는
진행하지 않는다. credential store가 없으면 인증이 평문 파일에 저장될 수 있다. 저장 위치 확인과
로그아웃은 [서버 안내](docs/SERVER.md#1-private-repository-로그인과-설치)를 참고한다.
([GitHub CLI 공식 로그인 문서](https://cli.github.com/manual/gh_auth_login))

### 2. 코드 받기

프로젝트를 둘 위치에서 실행한다. 현재 위치 아래 `asgcn-unet/` 폴더가 생긴다.

```bash
gh auth setup-git --hostname github.com &&
git clone https://github.com/costunder/asgcn-unet.git &&
cd asgcn-unet
```

`gh auth setup-git`은 HTTPS Git 인증에 방금 로그인한 GitHub CLI를 사용하도록 설정한다.
([공식 설명](https://cli.github.com/manual/gh_auth_setup-git))
같은 이름의 폴더가 이미 있으면 삭제하거나 덮어쓰지 말고, 기존 clone을 확인하거나 다른 위치에서
진행한다. 이후 명령은 모두 저장소 root에서 실행한다.

### 3. 환경 설치

```bash
nvidia-smi
df -h .
test -e .env || cp .env.example .env
bash scripts/setup.sh &&
source .venv/bin/activate &&
python scripts/check_env.py --require-cuda --lock constraints/py312.txt
```

Python 3.12·PyTorch 2.13.0 고정 환경을 프로젝트의 `.venv`에 설치한다. Conda는 Python과 GitHub CLI
제공용이고 실제 학습은 `.venv`를 쓴다. 기존 `.env`는 보존하므로 이전 설정이 있으면 먼저 확인한다.
Linux glibc 2.28 이상과 `curl`이 필요하다. 기본 PyTorch wheel과 서버 드라이버가 맞지 않거나 CUDA
검사가 실패하면 학습하지 말고 [서버 환경 안내](docs/SERVER.md#1-private-repository-로그인과-설치)를
확인한다. 설치가 됐다는 것만으로 GPU 학습이 검증된 것은 아니다.

### 4. 두 데이터셋 준비

EventAid-R 전체 14개 ZIP은 자동 다운로드한다. 중단되면 같은 명령으로 이어받는다.

```bash
bash scripts/get_aid.sh --all
mkdir -p data/_archives
```

EventHDR은 [공식 OneDrive 폴더](https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N)에서
**train과 eval 폴더를 각각 ZIP으로 다운로드**한다. MobaXterm 왼쪽 SFTP에서 이 저장소의
`data/_archives/`로 올린다. 서버 파일명을 `train.zip`, `eval.zip`으로 맞춘 뒤 실행한다.

```bash
bash scripts/get_hdr.sh --archive data/_archives/train.zip --split train &&
bash scripts/get_hdr.sh --archive data/_archives/eval.zip --split eval &&
python scripts/check_env.py --require-full-data --lock constraints/py312.txt
```

EventHDR의 브라우저 다운로드·업로드는 수동 단계다. 이미 받은 H5 폴더나 통합 ZIP이 있다면
[다른 가져오기 방법](docs/SERVER.md#2-전체-데이터-배치)을 사용한다.
최종 데이터는 약 50.4GB지만 ZIP 원본·가상환경·학습 결과 공간은 별도로 필요하다.

### 5. 전체 학습·보정·평가

GPU가 할당된 터미널에서 실행한다. 기본 설정은 **40 epoch 전체 학습**이며 smoke test가 아니다.

```bash
source .venv/bin/activate
mkdir -p logs
set -o pipefail
bash scripts/run.sh all 2>&1 | tee logs/run.log
```

자동 순서: 전체 데이터 검사 → 최고 밀도 CUDA 사전검증 → ANN 학습 → SNN 보정 → 두 데이터셋 전체
ANN/SNN 평가. 결과는 `runs/`, 실행 로그는 `logs/run.log`에 저장된다.
다운로드와 학습에는 서버·네트워크에 따른 시간이 걸린다.

연결이 끊겨도 계속 실행하려면 위 블록을 **`tmux new-session -s asgcn`으로 연 세션 안에서**
실행한다(`tmux` 설치 필요). 분리는 `Ctrl-b`, `d`, 재접속은 `tmux attach -t asgcn`이다.
SLURM/PBS 서버는 [scheduler 안내](#slurmpbs-scheduler)를 따른다.
중단 후에는 처음부터 `all`을 반복하지 말고 [재개 절차](#중단-후-재개와-결과-보호)를 따른다.

서버 재접속 후에는 기존 저장소로 이동하고 `conda activate asgcn`, `source .venv/bin/activate`만
실행한다. 로그인·clone·환경 생성을 매번 반복하지 않는다.

## 실험 범위

기본 실험 범위는 다음과 같다.

- EventHDR `train/1.h5`–`51.h5` 전체로 40 epoch ANN 학습
- EventHDR `eval/1.h5`–`19.h5` 전체로 마지막 epoch에 한 번 내부 평가
- 학습 전 EventHDR train 전체 graph topology를 조사하고 최고 밀도 표본에서 CUDA 학습 step 측정
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

`configs/train.json`의 `validate_every: null`은 EventHDR eval을 **40번째(마지막) epoch에 단 한 번만**
평가한다는 뜻이다. eval loss는 gradient에 들어가지 않고 여러 epoch 중 checkpoint를 고르는 데도
사용되지 않는다. 호환성을 위해 파일명은 `best.pt`지만, 이 설정에서는 마지막 epoch에서 한 번
계산한 finite macro SSIM을 가진 모델이다. EventAid-R은 그 이후 외부 일반화 평가에만 사용한다.

두 loader는 config에 명시된 `target_normalization.mode=integer_dtype_max`로 정수 target을
`[0,1]`에 놓은 뒤 동일한 `log1p(5000*x)/log1p(5000)` tone mapping을 적용한다. float target은
`known_scale` 또는 `already_normalized`를 명시해야 하며 NaN/Inf와 범위 위반은 즉시 거부한다.
frame별 percentile 보정은 `percentile_debug_only`와 `debug_only=true`를 함께 지정한 비보고용
진단에서만 열린다. EventAid-R은 event block `i`를 다음 GT `i+1`과 짝짓는 `target_offset: 1`을
사용한다. offset은 bool·실수가 아닌 정확한 정수만 받는다. 이는 이 저장소의 정렬 가정이지 ASGCN 논문
값은 아니다. EventHDR `images/image<index>`는 numeric suffix가 유일해야 하며 문자열순이 아닌 숫자순으로
읽는다. EventAid-R full inspect는 event text의 원 timestamp min/max와 `timestamps.txt` interval의 span
ratio·offset·범위 이탈 수를 기록하되, 공식 14 ZIP의 단위가 실측으로 확인되기 전에는 이를 임의로 hard
fail 조건으로 만들지 않는다.

## 전체 실행에서 하는 일

`scripts/run.sh all`은 다음 순서를 fail-fast로 실행한다.

1. `check`: CUDA·locked dependency·전체 파일 coverage를 확인하고 두 데이터의 모든 sample을 decode
2. `profile`: EventHDR train graph를 전수 조사하고 edge 수 상위 표본 3개에서 CUDA forward/backward,
   optimizer step, peak allocated/reserved VRAM과 step time 측정
3. `train`: 검증된 `runs/profile.json`을 현재 config·data·source·CUDA runtime에 다시 결합한 뒤
   EventHDR train 전체 40 epoch ANN 학습
4. `calibrate`: EventHDR train의 모든 calibration sample로 `best_snn.pt` 생성
5. `eval`: EventHDR eval과 EventAid-R 전체에 대해 ANN 1회 및
   `literal_eq15`/`standard_if` × `T=4,8,16,32` 평가·benchmark

기본 `PROFILE_TOP_DENSITY=10`은 edge 수 상위 10개 interval의 topology를 상세 기록하고,
`PROFILE_SAMPLES=3`은 그중 3개를 실제 학습 step으로 측정한다. 이 profile은 기록된 GPU에서 선택된
최고 밀도 표본이 통과했다는 실측 gate이지 전체 40 epoch의 모든 미래 step에 대한 절대 VRAM 보증은
아니다. profile artifact에도 `absolute_vram_guarantee=false`가 고정된다.

전체 품질 평가는 모든 sample을 사용한다. 기본 `BENCHMARK_STEPS=100`은 GPU compute latency를 재는
별도 timing 반복 수이며 품질 평가를 100 sample로 줄이는 설정이 아니다. `save_predictions: 20`도
PNG 저장 수만 제한하고 metric 계산 범위는 제한하지 않는다.
보고 가능한 `metrics.json`은 `eval.max_samples=null`인 quality evaluation에서만 생성된다. 값이
설정된 부분 quality 실행은 기본적으로 시작 전에 거부하며, 합성 테스트용 명시적 non-reporting
우회로 실행해도 `report_eligible=false`와 그 이유가 기록된다. 표본 수를 갖는 compute-only
benchmark는 이 quality 계약과 별도로 sampling provenance를 기록한다.

실행될 명령표만 확인하려면 데이터와 GPU 없이 다음을 사용할 수 있다. 실제 실행을 대신하지 않는다.

```bash
DRY_RUN=1 bash scripts/run.sh all
```

## 중단 후 재개와 결과 보호

학습은 매 epoch 종료 시 `last.pt`를 원자적으로 갱신한다. 중단 후 같은 run을 epoch 경계에서 재개한다.

```bash
RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  bash scripts/run.sh train 2>&1 | tee -a logs/train.log
```

`run.sh`은 `check`, `profile`, `train`, `calibrate`, `eval`, `all` stage를 각각 실행할 수 있다. `train`은
기본적으로 통과한 `runs/profile.json`을 요구하고, 이를 현재 config·전체 train data digest·source·GPU
runtime에 다시 결합한다. 학습 재개가 끝나면 `bash scripts/run.sh calibrate`, 이어서
`bash scripts/run.sh eval`을 실행한다. 기존 artifact를 발견해도 자동으로 건너뛰거나 덮어쓰지 않으므로,
부분 실행을 복구할 때 완료된 stage를 임의로 다시 실행하지 않는다. 합성 fixture용
`--allow-unverified-preflight` 우회는 checkpoint에 비보고용으로 영구 기록되고 `all` 및 scheduler
wrapper에서는 허용되지 않는다.

resume 시 model, optimizer, scheduler, AMP scaler, RNG, history뿐 아니라 config, 상대 data identity,
전체 data SHA-256, source tree hash와 GPU protocol을 교차검증한다. `validate_every: null`인 run은
계획한 terminal epoch도 protocol에 봉인하므로 마지막 평가를 마친 뒤 epochs만 늘려 같은 run을
재개할 수 없다. 연장 학습은 새 output directory의 새 protocol로 시작한다. 계약이 일치하지 않으면
조용히 다른 실험을 이어 붙이지 않고 중단한다.
EventHDR 보고용 ANN은 문법적으로 유효한 반복 `best_validation_macro_ssim` checkpoint도 허용하지 않고,
계획 epoch에서 완료된 `single_final_epoch` terminal validation을 반드시 요구한다.

calibration은 clean `ann_inference` checkpoint만 받고, 학습 당시 EventHDR train 전체 byte digest,
transform, final split manifest와 source tree가 현재 실행과 일치해야 한다. 또한 manifest가 가리키는
모든 training sample을 dataset index `0..N-1` 순서로 정확히 한 번씩 사용해야 `sealed=true`가 된다.
일부 sample만 지정하면 기본 실행은 변환 전에 실패한다. 테스트용 `--allow-unsealed-calibration`으로만
부분 보정을 만들 수 있고, 결과에는 불일치 이유와 `report_eligible=false`가 영구 기록된다.
이 override를 명시한 사실 자체가 taint이므로 sample이 우연히 전체여도 `sealed=false`다.
validation protocol v7은 train/validation index의 sample 수·group별 수·ordered identity SHA-256도
봉인하며, 보고용 ANN은 `max_train_samples=max_val_samples=null`인 전체 학습·검증만 허용한다.
SNN model state에는 실제 시도 수 `calibration_attempts`, 32-byte
`calibration_commitment_digest`, `calibration_commitment_sealed`를 persistent buffer로 저장한다.
protocol·선택 수·sampling과 valid/minimum/dead-channel summary core의 commitment를 metadata summary,
layer별 valid count와 `calibration_activation_max`(관측 raw maximum),
`normalization_scale`(식 (6)의 effective scale), `dead_channel_mask`까지 대조한다. dead channel은 raw
maximum과 mask에는 그대로 남고 effective scale만 1이므로 저장 후 strict reload에서도 summary가
변하지 않는다. 따라서
부분 보정 tensor에 전체 보정 metadata만 이식하거나 선택 metadata만 사후 재라벨링한 checkpoint는
load 단계에서 거부된다.
기존 calibration 또는 평가 결과도 자동 덮어쓰지 않는다. 의도적으로 calibration만 다시 만들 때는
`OVERWRITE_CALIBRATION=1`을 명시할 수 있지만, 이미 완료된 평가 artifact는 보존하거나 config의
`eval.output_dir`를 새 경로로 바꿔 별도 run으로 실행해야 한다.

주요 출력은 다음과 같다.

```text
runs/train/
├── config.json
├── history.json
├── last.pt                       # 학습 재개용 전체 상태
├── best.pt                       # ANN inference용 clean checkpoint
└── best_snn.pt                   # 보정된 SNN graph encoder checkpoint

runs/profile.json                 # 전체 topology + 최고 밀도 CUDA 학습-step 사전검증

runs/status/
├── check.json
├── profile.json
├── train.json
├── calibrate.json
└── eval.json                     # RUNNING/COMPLETED/FAILED stage 상태

runs/eval/hdr/ann/
runs/eval/hdr/snn_literal_eq15_T{4,8,16,32}/
runs/eval/hdr/snn_standard_if_T{4,8,16,32}/
runs/eval/aid/ann/
runs/eval/aid/snn_literal_eq15_T{4,8,16,32}/
runs/eval/aid/snn_standard_if_T{4,8,16,32}/
```

각 평가 폴더에는 `metrics.json`, `frames.csv`, `predictions/`, `benchmark.json`이 생긴다. 보고용 ANN은
검증된 CUDA preflight를 통과한 clean `ann_inference` lineage를, 보고용 SNN은 그 ANN에서 생성한
`calibration_protocol.sealed=true` lineage를 요구한다. config/model/checkpoint file·tensor,
현재 eval data content·transform·manifest·coverage·sampling, source/runtime/precision 계약과 SHA-256을
`metrics.json`과 `benchmark.json`에 기록한다. 평가 시에도 calibration의 transform·final manifest·전체
sample identity/sampling·runtime을 ANN train index commitment와 비교하고, source ANN의 epoch·selection·
terminal validation을 포함한 reporting contract도 다시 검증한다. 합성 테스트에서만 쓰는 명시적 unsealed-checkpoint
우회는 결과의 `report_eligible=false`와 이유를 영구 기록하며 public wrapper에는 노출하지 않는다.
EventHDR lineage는 planned/completed/checkpoint epoch가 같고 validation selection이
`single_final_epoch_macro_ssim`인 경우만 보고 가능하다.
source ANN의 training dataset type도 EventHDR로 고정하고 training protocol의 seed·optimizer·scheduler·
loss·data order·AMP·runtime을 public training config와 교차검증한다. 평가 dataset은 명시적
`expected_file_count`와 final/fixed manifest의 정확한 파일 집합을 가져야 하며, EventHDR quality/benchmark는
현재 file bytes·transform·manifest가 source ANN validation 계약과 같아야 한다. protocol SHA-256에는
실제 inference mode, SNN T와 effective dynamics도 포함된다.
quality evaluation은 추가로 `eval.max_samples=null`을 요구하므로 부분 `metrics.json`이 보고 가능 상태로
표시되지 않는다. benchmark의 고정 warmup/측정 step은 품질 표본 제한과 다른 compute protocol이다.
JSON/CSV와
checkpoint는 temporary file을 완성한 뒤 원자적으로 교체하며 JSON은 NaN/Infinity를 허용하지 않는다.
SHA-256 seal은 재현 identity와 우발적·부분적 metadata 변조 탐지용이지 전자서명이 아니다. 악의적 주체가
checkpoint와 모든 내부 hash를 함께 다시 작성하는 상황까지 인증하려면 산출물 hash를 별도 immutable
실험 원장이나 서명 저장소에 보존해야 한다.
평가 직후 target, prediction, metric, graph diagnostic, latency와 `dt_us`의 비정상 수치도 거부한다.
품질 지표는 PSNR(완전 일치 시 120 dB 상한), Gaussian SSIM, RMSE, temporal L1과
micro/H5-or-ZIP-group macro 집계를 포함한다. `configs/hdr.json`과 `configs/aid.json`은 품질 비교의
기본 precision을 FP32/TF32 off로 명시하며, `amp_fp16`과 `bf16`을 선택한 별도 run은 요청값·실제
autocast dtype·parameter dtype·TF32 적용 여부를 결과에 기록한다. benchmark는 data I/O와
host-to-device 이동을 제외한 model compute latency, FPS, percentile, graph 처리량, spike rate와 GPU
peak memory를 기록한다. 이 compute-only memory 범위에는 GPU event input과 model 실행은 포함하지만
품질 계산에만 쓰는 ground-truth target은 CPU에 유지해 제외한다. PyTorch GPU latency를 FPGA/ASIC
latency나 에너지로 해석하면 안 된다.

## SLURM/PBS scheduler

클러스터 batch job은 저장소 root에서 제출한다. SLURM에서 profile→학습→전체 calibration 의존성은 다음과
같이 건다.

```bash
profile_id=$(sbatch --parsable --export=PROJECT_ROOT="$PWD" server/profile.sbatch)
train_id=$(sbatch --parsable --dependency=afterok:${profile_id} \
  --export=PROJECT_ROOT="$PWD" server/train.sbatch)
cal_id=$(sbatch --parsable --dependency=afterok:${train_id} \
  --export=PROJECT_ROOT="$PWD" server/calibrate.sbatch)
```

ANN 두 평가와 SNN 전체 행렬을 dependency로 제출한다.

```bash
for config in configs/hdr.json configs/aid.json; do
  sbatch --dependency=afterok:${train_id} \
    --export=PROJECT_ROOT="$PWD",CONFIG_PATH="$config",CHECKPOINT_PATH=runs/train/best.pt,INFERENCE_MODE=ann \
    server/eval.sbatch
done

for config in configs/hdr.json configs/aid.json; do
  for dynamics in literal_eq15 standard_if; do
    for timestep in 4 8 16 32; do
      sbatch --dependency=afterok:${cal_id} \
        --export=PROJECT_ROOT="$PWD",CONFIG_PATH="$config",CHECKPOINT_PATH=runs/train/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS="$dynamics",SIMULATION_STEPS="$timestep" \
        server/eval.sbatch
    done
  done
done
```

SLURM `--export`에는 각 job이 실제로 쓰는 변수만 명시한다. login shell 전체를 전달하면 token, proxy,
credential 같은 무관한 환경변수도 compute node와 job 환경에 복제될 수 있으므로 `ALL`은 사용하지 않는다.
학습 resume job에는 `RESUME_CHECKPOINT="$PWD/runs/train/last.pt"`를 `--export`에 추가한다.
PBS/Torque용 동등 wrapper는 `server/profile.pbs`, `server/train.pbs`, `server/calibrate.pbs`,
`server/eval.pbs`이며
`qsub -W depend=afterok:<job-id>`로 같은 의존성을 건다. `#SBATCH`/`#PBS`의 GPU, memory, walltime,
queue, account는 학교 scheduler 정책에 맞게 조정해야 한다. 자세한 서버 운용은
[docs/SERVER.md](docs/SERVER.md), 실험 정의는 [docs/EXPERIMENT.md](docs/EXPERIMENT.md)에 있다.
wrapper log는 기본적으로 hostname/job ID를 생략하고 config/checkpoint는 basename만 남긴다. 로컬 진단에서
정확한 host와 경로가 꼭 필요할 때만 `INCLUDE_PRIVATE_HOST_PROVENANCE=1`을 제출 변수에 추가하고, 그 log는
공개하거나 첨부하지 않는다. 다만 Slurm의 `slurm-...-<job-id>.out/.err`와 PBS의
`<job-name>.o<job-id>` 같은 raw scheduler log는 **파일명 자체에 job ID가 있으므로 그대로 공개하지 않는다.**
공개 전에는 기본 provenance로 생성한 raw log를 `logs/public/train.stdout.log` 같은 중립 파일명으로 복사하고,
저장소 밖 로컬 denylist를 사용해 그 공개 후보의 내용도 로컬에서 검사한다. scan 실패 파일과
`INCLUDE_PRIVATE_HOST_PROVENANCE=1`로 만든 opt-in log는 중립명으로 바꿔도 비공개다.

```bash
python scripts/scan_private_text.py logs/public/train.stdout.log \
  --root "$PWD" --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

## 현재 검증 상태와 한계

- 저장소의 개인정보 gate는 모든 Git tracked UTF-8 text와 생성된 `code_summary.md`를 검사한다.
  generic Unix/macOS/Windows home path와 labelled identity를 기본 탐지한다. 실제 계정·host marker는
  저장소 밖 로컬 denylist 또는 로컬 환경변수 `PRIVATE_MARKERS_B64`로만 주입하며, GitHub secret·변수·
  workflow·log·artifact로 전송하지 않는다. 로컬 release gate는 `--all-tracked --all-history`와
  `--require-external-patterns`를 함께 사용해 빈 denylist와 shallow history를 거부한다. 문자열 결합·
  constant f-string과 Base64 표현도 복원 검사하며 탐지 로그에는 marker 내용을 출력하지 않는다.
  CI는 실제 marker 없이 generic current-tree/history 검사와 clean provenance를 확인한다. CI 통과는
  로컬 실제-marker 검사 통과를 대신하지 않는다. history 검사는 모든 local ref에서 도달 가능한 unique
  Git blob을 대상으로 하며, 현재 checkout 검사와 별개다. history 열거는 신·구 Git에서 공통인
  `git rev-list --objects --all`의 LF 레코드를 사용하며 `-z` 지원 차이에 의존하지 않는다. 경로 출력은
  진단용 hint이고 실제 blob 내용은 object ID로 읽는다.
- `code_summary.md`는 `python scripts/build_code_summary.py`로 생성한 결정적 전체 text snapshot이다.
  파일별 SHA-256, snapshot SHA-256, 포함 파일 수와 생성 당시 commit/tree/branch/dirty provenance를
  기록하고 CI에서 `--check`로 working tree와의 일치를 확인한다. dirty working tree의 snapshot은
  superseded commit을 가리키지 않도록 commit/tree를 `null`로 기록하며 snapshot SHA-256을 검증
  identity로 사용한다. 배포 때는 코드·문서·검증 수치 수정과 history 정리를 마쳐 최종 sanitized source
  commit을 확정하고, clean source에서 summary를 재생성한 뒤 summary-only commit을 만든다.
  `--check --require-clean-provenance`는 source commit이 현재 HEAD의 ancestor이고 그 이후 summary
  이외의 tracked source가 바뀌지 않았는지 확인한다. 생성 뒤 문서 수정이나 history rewrite가 필요하면
  source commit 확정부터 다시 수행한다. 배포 결과는 본문을 고쳐 적기보다 최종 SHA의 Actions와 별도
  배포 기록으로 확인한다.
- 2026-08-30 Windows CPU 검증은 **267 passed, 1 skipped**다. skip 1건은 OS의 symlink 생성 권한이
  없는 경우의 shared-storage link test다. shell entrypoint 15개는 MSYS Bash에서 각각 구문 검사했으며,
  실제 Linux Git 2.47.3 실행 결과는 아니다. 전체 검증 명령은 `hand_off.md`에 기록한다. 이 로컬 결과는
  원격 history/과거 CI 정리나 배포 성공을 증명하지 않는다. 원격 배포 여부는 대상 SHA의 release gate와
  GitHub Actions 결과로 별도 확인한다.
- 코드의 unit/integration test와 Linux 의존성 검사는 구성되어 있지만, EventHDR+EventAid-R 전체
  실데이터를 사용한 `runs/profile.json`, CUDA 40-epoch 학습·전체 행렬 실행, A6000/A100 peak
  memory·runtime·latency artifact는 이 로컬 검증에서 생성하지 않았다. 따라서 README는 실행 절차
  설명이지 측정 완료 보고서가 아니다.
- 공개 자료에 EventHDR H5↔physical-scene 완전 대응표가 없어 공식 train/eval root 이상의
  scene-disjoint 주장을 하지 않는다.
- 원 논문의 동적 asynchronous K-hop update, pooling/classifier, energy model은 포함하지 않는다.
- 반도체 RTL/FPGA/ASIC, event compression/transport, 실제 전력·에너지 측정은 후속 과제 범위다.
- recurrent batch size는 1이고 resume granularity는 epoch 단위다. 전체 실행 시간과 저장 공간은
  서버 GPU, filesystem, dataset decode 속도에 따라 달라진다.

코드 전체 스냅샷은 [code_summary.md](code_summary.md), 인수인계와 연구상 주의점은
[hand_off.md](hand_off.md)를 참조한다.
