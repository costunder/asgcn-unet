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

## Linux GPU 서버 빠른 시작

```bash
git clone https://github.com/costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

# 서버 드라이버에 맞는 값은 PyTorch 공식 설치 선택기에서 확인한다.
export TORCH_INDEX_URL=<PYTORCH_WHEEL_INDEX_URL>
export PROJECT_EXTRAS=dev,eval
bash scripts/setup_server.sh
source .venv/bin/activate

python scripts/check_environment.py --require-cuda
python -m pytest -q
python -m asgcn_recon.smoke --workspace /tmp/asgcn-smoke
```

MobaXterm, 외부 데이터 심볼릭 링크, tmux, Docker, SLURM과 장애 대응은
[Linux GPU 서버 가이드](docs/SERVER_SETUP.md)에 정리했다. 데이터 역할, ANN→SNN 순서와
필수 ablation은 [실험 프로토콜](docs/EXPERIMENT_PROTOCOL.md)을 따른다.

## Windows 개발 환경

PowerShell에서:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

이 PC처럼 `py` 명령이 없으면 Codex 번들 Python 또는 설치된 Python 3.10 이상으로 같은 명령을
실행하면 된다. CUDA PyTorch를 사용할 경우에는 먼저 PyTorch 공식 설치 명령으로 CUDA wheel을
설치한 뒤 `pip install -e ".[dev]"`를 실행한다.

## 데이터 배치

```text
data/
  EventHDR/
    train/*.h5
    eval/*.h5
  EventAid-R/
    R-ball.zip
    R-bear.zip
    ...
    R-wall.zip
```

EventHDR는 [공식 OneDrive](https://github.com/yunhao-zou/EventHDR)의 `train`과 `eval` H5를
각 폴더에 둔다. EventAid-R은 다음 명령으로 장면별 ZIP을 재개 가능한 방식으로 받는다.

```powershell
.\scripts\download_eventaid_r.ps1 -Destination .\data\EventAid-R
```

처음에는 작은 `R-bear`만 받아 파이프라인을 점검할 수 있다.

```powershell
.\scripts\download_eventaid_r.ps1 -Destination .\data\EventAid-R -Scenes R-bear
```

받아 둔 공식 H5/ZIP으로 실제 모델 forward까지 확인한다.

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_real_samples.py
```

## 빠른 검증

실제 데이터 없이도 H5와 ZIP의 공식 구조를 본뜬 작은 자료로 전체 경로를 검사한다.

```powershell
.\scripts\smoke_test.ps1
```

## 실행

데이터 구조 검사:

```powershell
asgcn-recon inspect --config configs/eventhdr_train.json
asgcn-recon inspect --config configs/eventaid_r_eval.json
```

EventHDR 학습:

```powershell
asgcn-recon train --config configs/eventhdr_train.json
```

중단된 서버 학습 재개:

```bash
asgcn-recon train --config configs/eventhdr_train.json \
  --resume runs/eventhdr_asgcn/last.pt
```

EventHDR 공식 eval 최종시험:

```powershell
asgcn-recon evaluate --config configs/eventhdr_eval.json `
  --checkpoint runs/eventhdr_asgcn/best.pt
```

EventAid-R 외부 평가:

```powershell
asgcn-recon evaluate --config configs/eventaid_r_eval.json `
  --checkpoint runs/eventhdr_asgcn/best.pt
```

ANN 학습이 끝난 뒤 graph encoder의 BatchNorm을 folding하고 EventHDR train activation으로
채널별 IF threshold를 보정한다. EventAid-R은 이 보정에 사용하지 않는다.

```powershell
asgcn-recon calibrate --config configs/eventhdr_train.json `
  --checkpoint runs/eventhdr_asgcn/best.pt `
  --output runs/eventhdr_asgcn/best_snn.pt --samples 500

asgcn-recon evaluate --config configs/eventaid_r_eval.json `
  --checkpoint runs/eventhdr_asgcn/best_snn.pt `
  --inference-mode snn --simulation-steps 16
```

지연시간/FPS 측정:

```powershell
asgcn-recon benchmark --config configs/eventaid_r_eval.json `
  --checkpoint runs/eventhdr_asgcn/best.pt --warmup 10 --steps 100
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
- `data/`, `runs/`, H5/ZIP/checkpoint는 Git에 올라가지 않는다. GitHub Actions는 합성 H5/ZIP으로
  Windows와 Linux, Python 3.10–3.12에서 설치·단위 테스트·end-to-end smoke를 수행한다.
