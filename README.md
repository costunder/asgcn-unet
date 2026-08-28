# ASGCN Event-to-Frame 실험

[![CI](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/costunder/asgcn-event-reconstruction/actions/workflows/ci.yml)

EventHDR로 학습하고 EventAid-R에서 외부 일반화를 평가하는 재현 가능한 ASGCN 기반
event-to-frame 연구 코드다. Windows 로컬 개발뿐 아니라 **GitHub clone → MobaXterm/SSH →
Linux GPU 서버 또는 SLURM 실행**을 지원한다.

이 프로젝트는 두 데이터셋만 사용한다.

- **EventHDR**: 실제 이벤트에서 HDR 프레임을 복원하는 주 학습/내부 검증 데이터
- **EventAid-R**: 장면이 겹치지 않는 외부 일반화 평가 데이터

분류용 ASGCN의 `이벤트 -> 시공간 그래프 -> sparse graph convolution` 부분을 유지하고,
MLP 분류기를 `graph feature rasterization -> 경량 U-Net decoder`로 교체했다. 원본 이벤트
좌표 `(x, y, t, p)`를 직접 사용하며 voxel 파일을 미리 만들지 않는다. EventAid-R ZIP도
압축을 풀지 않고 직접 읽기 때문에 약 24.68GB의 중복 저장을 피한다.

## 실험 역할

| 단계 | 데이터 | 목적 |
|---|---|---|
| 학습 | EventHDR train | 이벤트 그래프에서 HDR intensity frame 복원 |
| 모델 선택 | EventHDR train의 명시적 holdout | PSNR/SSIM 기준 checkpoint 선택 |
| 내부시험 | EventHDR 공식 eval | 학습·모델 선택이 끝난 뒤 1회 평가 |
| 외부 평가 | EventAid-R 전체 14장면 | 고속/비선형 운동에서 일반화 확인 |
| 시스템 평가 | 두 데이터 모두 | batch 1 latency, p50/p95, FPS, event throughput |

두 데이터의 영상 특성이 다르므로 기본 실험은 모두 **1채널 luminance**와 `[0,1]` 범위로
통일한다. EventHDR 결과에는 configurable log tone mapping을 적용해 HDR의 밝은 영역이 손실을
지배하지 않게 한다.

## MobaXterm/Linux GPU 서버: 처음부터 끝까지

MobaXterm은 SSH 접속 프로그램이고 아래 명령은 **접속한 Linux 서버**에서 실행한다. 서버에는
Git, `curl`, Python 3.10 이상과 `venv`, NVIDIA 드라이버가 필요하다. 전체 데이터는 약
50.4GB이고 가상환경·checkpoint·결과를 포함해 70GB 이상의 여유 공간을 권장한다.

사용자가 직접 결정해야 하는 항목은 두 가지다.

1. 비공개 GitHub 저장소에 접근할 SSH key 또는 HTTPS token
2. 서버 드라이버에 맞는 PyTorch wheel과 EventHDR 공식 파일의 다운로드·업로드

### 1. 저장소 clone

GitHub에 서버의 SSH key를 등록했다면:

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction
```

HTTPS를 사용한다면 다음 명령에서 GitHub 비밀번호가 아니라 token으로 인증한다.

```bash
git clone https://github.com/costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction
```

### 2. Python·PyTorch 환경 구성

먼저 GPU와 기본 명령을 확인한다.

```bash
git --version
curl --version
python3 --version
nvidia-smi
```

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 서버 드라이버와 호환되는
wheel index를 확인한다. 아래 블록을 그대로 실행하고, 별도 index가 필요 없으면 질문에서
Enter만 누른다. 입력값은 셸 redirection으로 오인되지 않도록 스크립트에 안전하게 전달된다.

```bash
read -r -p "PyTorch wheel index URL (Enter=PyPI default): " TORCH_INDEX_URL
export TORCH_INDEX_URL
export PROJECT_EXTRAS="dev,eval"
export REQUIRE_CUDA=0

bash scripts/setup.sh
source .venv/bin/activate
python scripts/check_env.py
python -m pytest -q
```

설치 스크립트는 `.venv` 생성, 의존성 설치와 데이터·실행 폴더 생성을 수행한다. 로그인 노드에서
GPU가 숨겨지는 클러스터도 있으므로 `REQUIRE_CUDA=0`은 설치 단계에만 사용한다. 실제 GPU node
또는 SLURM allocation에서는 반드시 다음 검사를 통과해야 한다.

```bash
source .venv/bin/activate
python scripts/check_env.py --require-cuda
```

반복 설치용 값을 파일로 보존하려면 `.env.example`을 `.env`로 복사해 같은 값을 적으면 된다.

### 3. 데이터 준비

최종 구조는 다음과 같아야 한다.

```text
data/
├── EventHDR/
│   ├── train/1.h5 ... 51.h5
│   └── eval/*.h5
└── EventAid-R/
    ├── R-ball.zip
    ├── R-bear.zip
    └── ... R-wall.zip
```

EventHDR는 배포처가 OneDrive이므로 자동 downloader를 제공하지 않는다. [EventHDR 공식
저장소](https://github.com/yunhao-zou/EventHDR)에서 train 51개와 공식 eval 19개 H5를 받은 뒤,
MobaXterm 왼쪽 SFTP 패널로 각각 `data/EventHDR/train/`, `data/EventHDR/eval/`에 업로드한다.
공유 스토리지에 이미 있다면 이 디렉터리로 심볼릭 링크해도 된다.

```bash
find data/EventHDR/train -maxdepth 1 -type f -name '*.h5' | wc -l
find data/EventHDR/eval -maxdepth 1 -type f -name '*.h5' | wc -l
```

기대 출력은 각각 `51`, `19`다. 기본 manifest는 `1.h5`–`47.h5`를 학습, `48.h5`–`51.h5`를
검증에 사용하며, 개선된 `inspect`가 두 split의 누락 파일을 모두 검사한다.

EventAid-R은 먼저 작은 `R-bear`만 받아 로더를 확인한다. ZIP은 압축 해제하지 않는다.

```bash
bash scripts/get_aid.sh R-bear
python -m asgcn_recon.cli inspect \
  --config configs/aid_ann.json --samples 2
```

최종 외부평가 전에는 전체 14개 장면을 받는다. 이미 받은 유효 ZIP은 자동으로 건너뛴다.

```bash
bash scripts/get_aid.sh --all
```

두 공식 데이터의 파일 구조와 샘플 로딩을 CLI로 검사한다.

```bash
python -m asgcn_recon.cli inspect \
  --config configs/hdr_train.json --samples 2
python -m asgcn_recon.cli inspect \
  --config configs/hdr_ann.json --samples 2
python -m asgcn_recon.cli inspect \
  --config configs/aid_ann.json --samples 2
```

### 4. EventHDR ANN 학습

SSH가 끊겨도 유지되도록 `tmux` 안에서 실행한다.

```bash
tmux new-session -s asgcn -c "$PWD" \
  "bash -lc 'source .venv/bin/activate && bash scripts/train.sh configs/hdr_train.json'"
```

분리는 `Ctrl-b`, `d`, 재접속은 `tmux attach -t asgcn`이다. 중단된 epoch 이후부터 재개하려면:

```bash
source .venv/bin/activate
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/train.sh configs/hdr_train.json
```

학습 결과는 `runs/eventhdr_asgcn/{config.json,history.json,best.pt,last.pt}`에 저장된다.

### 5. ANN→SNN 보정

EventHDR train만 사용해 graph encoder의 BatchNorm을 folding하고 채널별 activation threshold를
구한다. EventAid-R은 보정에 사용하지 않는다.

```bash
source .venv/bin/activate
python -m asgcn_recon.cli calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500
```

현재 SNN 경로는 graph encoder ReLU의 spike-rate 근사이며 U-Net·ConvGRU decoder는 analog로
유지되는 hybrid ANN–SNN이다.

### 6. EventHDR·EventAid-R ANN/SNN 평가

`eval.sh`는 품질 평가 후 I/O를 제외한 latency benchmark까지 연속 실행한다. 모든 console
출력도 보존한다.

```bash
mkdir -p logs

INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh \
  configs/hdr_ann.json runs/eventhdr_asgcn/best.pt \
  2>&1 | tee logs/eventhdr_ann.log

INFERENCE_MODE=snn SIMULATION_STEPS=16 RUN_BENCHMARK=1 \
  bash scripts/eval.sh \
  configs/hdr_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/eventhdr_snn_t16.log

INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh \
  configs/aid_ann.json runs/eventhdr_asgcn/best.pt \
  2>&1 | tee logs/eventaid_r_ann.log

INFERENCE_MODE=snn SIMULATION_STEPS=16 RUN_BENCHMARK=1 \
  bash scripts/eval.sh \
  configs/aid_snn.json runs/eventhdr_asgcn/best_snn.pt \
  2>&1 | tee logs/eventaid_r_snn_t16.log
```

ANN/SNN별 출력 디렉터리가 분리되어 서로 덮어쓰지 않는다.

```text
runs/
├── eventhdr_official_eval_ann/
├── eventhdr_official_eval_snn/
├── eventaid_r_external_ann/
└── eventaid_r_external_snn/
```

각 디렉터리에는 `metrics.json`, `frames.csv`, `predictions/`가 생긴다. benchmark JSON은
console에 출력되며 위 명령의 `logs/*.log`에도 남는다. 품질 평가만 실행하려면
`RUN_BENCHMARK=0`을 사용한다.

SLURM, Docker, 외부 데이터 심볼릭 링크와 장애 대응은 [Linux GPU 서버
가이드](docs/SERVER.md), 데이터 역할과 ablation은 [실험 문서](docs/EXPERIMENT.md)를 따른다.

## Windows 개발 환경

PowerShell에서는 환경을 설치한 뒤 배치한 공식 데이터를 CLI로 검사한다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,eval]"
.\scripts\get_aid.ps1 -Destination .\data\EventAid-R -Scenes R-bear
.\.venv\Scripts\python.exe -m asgcn_recon.cli inspect --config configs\hdr_train.json --samples 2
.\.venv\Scripts\python.exe -m asgcn_recon.cli inspect --config configs\hdr_ann.json --samples 2
.\.venv\Scripts\python.exe -m asgcn_recon.cli inspect --config configs\aid_ann.json --samples 2
```

## 비교 실험

최소한 다음 ablation을 고정한다.

1. `graph_layers=0`: 그래프 메시지 전달 없이 이벤트를 바로 rasterize
2. `graph_layers=6`: ASGCN-ANN 기본 모델
3. `inference_mode=snn`: activation calibration 후 IF spike-rate 추론
4. `max_events=4096/8192/16384`: 정확도-지연시간 trade-off
5. `causal_candidates=16/32/64`: 그래프 연결 밀도 trade-off

최종 표에는 PSNR/SSIM/LPIPS와 함께 평균 latency만 쓰지 말고 **p50, p95, FPS,
events/s, peak GPU memory**를 함께 기록한다.

## 주의

- EventAid-R은 공식적으로 평가 벤치마크 성격이 강하므로 기본 설정에서 학습에 섞지 않는다.
- EventHDR 배포본은 train H5가 51개지만 논문의 장면 수와 일치하지 않고 공식 대응표가 없다.
  현재 `manifests/eventhdr_split.json`은 파일 단위 holdout이며, 동일 물리 장면 파일을 확인한 뒤
  반드시 group 단위로 갱신해야 한다. 공식 eval 19개 H5는 checkpoint 선택에 사용하지 않는다.
- EventAid-R의 `event/NNNNNN.txt`는 `timestamp x y polarity`이며, 이벤트 구간은
  `GT_i -> GT_(i+1)` 사이이므로 정답은 **다음 번호 PNG**다. 동일 번호 연결은 off-by-one이다.
- EventAid-R FPS는 논문 수치를 hard-code하지 않고 각 ZIP의 `timestamps.txt` 차이로 계산한다.
- EventHDR H5의 각 이미지 `event_idx`까지의 이벤트를 해당 GT와 연결한다.
- 전체 다운로드 약 50.4GB 외에 checkpoint/캐시 공간이 필요하다. 이 구현은 별도 voxel/graph
  cache를 만들지 않아 100GB 저장공간 안에서 운용하도록 설계했다.
- `data/`, `runs/`, H5/ZIP/checkpoint는 Git에 올라가지 않는다. GitHub Actions는 공식 데이터를
  내려받지 않고 `tests/` 내부 fixture로 Windows와 Linux, Python 3.10–3.12 단위 테스트를 수행한다.
