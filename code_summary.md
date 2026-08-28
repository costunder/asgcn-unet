# 코드 원문 스냅샷

- 기준 커밋: 8c935001b0c71ce0ed7966ef0c6c1622a3fadb65
- 생성일: 2026-08-29 (Asia/Seoul)
- 포함 범위: 기준 커밋에서 Git이 추적하는 텍스트 파일 49개 전체
- 제외 범위: data/, runs/, .venv/, checkpoint, H5/ZIP, 캐시, 생성된 요약 문서 자체
- 사용 목적: 다른 ChatGPT/검토자가 파일 경로와 원문을 한 문서에서 교차 검증

아래 각 섹션은 `# 상대경로` 다음에 해당 파일의 원문 전체를 담는다.

# .dockerignore

~~~~text
/.git
/.github
/.venv
.pytest_cache
.ruff_cache
__pycache__
*.pyc
*.h5
*.zip
*.pt
*.pth
/data
/runs
/logs
*.hwp
~~~~

# .editorconfig

~~~~text
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.py]
indent_style = space
indent_size = 4

[*.{json,yml,yaml}]
indent_style = space
indent_size = 2

[*.ps1]
end_of_line = crlf
~~~~

# .env.example

~~~~dotenv
# Host paths mounted by Docker Compose.
DATA_DIR=./data
RUNS_DIR=./runs

# Optional PyTorch wheel index selected for the server driver.
# Obtain the current value from https://pytorch.org/get-started/locally/
# Example only: https://download.pytorch.org/whl/cu128
TORCH_INDEX_URL=

# Server-side installer overrides.
PYTHON_BIN=python3
VENV_DIR=.venv
PROJECT_EXTRAS=dev,eval
REQUIRE_CUDA=0
~~~~

# .gitattributes

~~~~text
* text=auto
*.py text eol=lf
*.sh text eol=lf
*.sbatch text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.json text eol=lf
*.md text eol=lf
*.ps1 text eol=crlf
Dockerfile text eol=lf
Makefile text eol=lf
.dockerignore text eol=lf
.editorconfig text eol=lf
.env.example text eol=lf
.gitattributes text eol=lf
.gitignore text eol=lf
*.hwp binary
*.h5 binary
*.zip binary
~~~~

# .github/ISSUE_TEMPLATE/bug_report.yml

~~~~yaml
name: 버그 보고
description: 재현 가능한 오류나 잘못된 결과를 보고합니다.
title: "[Bug]: "
labels:
  - bug
body:
  - type: markdown
    attributes:
      value: |
        민감한 정보나 대용량 데이터 파일은 첨부하지 마세요. 가능한 경우 합성 스모크 데이터로 재현해 주세요.

  - type: textarea
    id: summary
    attributes:
      label: 문제 설명
      description: 발생한 문제와 기대했던 동작을 설명해 주세요.
      placeholder: 무엇이 잘못되었으며 원래 어떻게 동작해야 하나요?
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: 재현 방법
      description: 실행한 명령과 최소 재현 절차를 순서대로 적어 주세요.
      placeholder: |
        1. ...
        2. ...
        3. ...
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: 오류 로그
      description: 전체 traceback 또는 관련 로그를 붙여 넣어 주세요.
      render: shell

  - type: dropdown
    id: dataset
    attributes:
      label: 관련 데이터 경로
      options:
        - 합성 스모크 데이터
        - EventHDR
        - EventAid-R
        - 데이터와 무관함
        - 확인하지 못함
    validations:
      required: true

  - type: input
    id: os
    attributes:
      label: 운영체제
      placeholder: 예) Ubuntu 22.04, Windows 11
    validations:
      required: true

  - type: input
    id: python
    attributes:
      label: Python 및 PyTorch 버전
      placeholder: 예) Python 3.11.9, PyTorch 2.4.0+cu121
    validations:
      required: true

  - type: input
    id: hardware
    attributes:
      label: 하드웨어
      description: CPU, GPU 모델과 GPU 사용 시 CUDA 버전을 적어 주세요.
      placeholder: 예) RTX 4090, CUDA 12.1

  - type: textarea
    id: config
    attributes:
      label: 설정 및 추가 정보
      description: 사용한 config, commit SHA, 수정 사항 등 원인 파악에 필요한 내용을 적어 주세요.

  - type: checkboxes
    id: checks
    attributes:
      label: 확인 사항
      options:
        - label: 최신 기본 브랜치에서도 문제가 재현됩니다.
          required: true
        - label: 이슈에 개인 정보, 인증 정보 또는 비공개 데이터가 포함되지 않았습니다.
          required: true
~~~~

# .github/pull_request_template.md

~~~~markdown
## 변경 내용

- 무엇을 변경했고 왜 필요한지 적어 주세요.

## 검증

- [ ] `python -m ruff check .`
- [ ] `python -m pytest -q`
- [ ] 데이터 로더/config 변경 시 공식 데이터에서 `asgcn-recon inspect`를 실행했습니다.

## 실험 영향

- [ ] 데이터 로더 또는 데이터 정렬 규칙을 변경하지 않았습니다.
- [ ] 모델 구조, 손실 함수 또는 평가 지표 변경 사항을 문서화했습니다.
- [ ] 설정이나 재현 방법이 달라진 경우 README와 config를 갱신했습니다.
- [ ] 대용량 데이터, 체크포인트, 실행 결과를 커밋하지 않았습니다.

## 추가 정보

관련 이슈, 실험 결과 또는 호환성 관련 참고 사항이 있다면 적어 주세요.
~~~~

# .github/workflows/ci.yml

~~~~yaml
name: CI

on:
  push:
  pull_request:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Ruff
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - name: Check out repository
        uses: actions/checkout@v6

      - name: Set up Python 3.12
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml

      - name: Install editable development package
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"

      - name: Run Ruff
        run: python -m ruff check .

      - name: Validate Linux shell entrypoints
        run: bash -n scripts/*.sh server/*.sbatch

  test:
    name: ${{ matrix.os }} / Python ${{ matrix.python-version }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 45
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - name: Check out repository
        uses: actions/checkout@v6

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml

      - name: Install editable development package
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"

      - name: Run unit and end-to-end tests
        run: python -m pytest -q
~~~~

# .gitignore

~~~~text
/.venv/
.env
.idea/
.vscode/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
/build/
/dist/
*.egg-info/
/data/
/runs/
/logs/
slurm-*.out
slurm-*.err
*.pt
*.pth
*.onnx

# Local project source documents may contain non-public application details.
*.hwp
~~~~

# compose.yaml

~~~~yaml
services:
  experiment:
    build:
      context: .
      args:
        TORCH_INDEX_URL: ${TORCH_INDEX_URL:-}
    image: asgcn-event-reconstruction:local
    working_dir: /workspace
    volumes:
      - ${DATA_DIR:-./data}:/workspace/data:ro
      - ${RUNS_DIR:-./runs}:/workspace/runs
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["inspect", "--config", "configs/hdr_train.json"]
~~~~

# configs/aid_ann.json

~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventaid_r_zip",
    "root": "data/EventAid-R",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "target_offset": 1,
    "tone_map": "none"
  },
  "model": {
    "hidden_dim": 64,
    "graph_layers": 6,
    "causal_candidates": 32,
    "spatial_radius": 0.12,
    "temporal_radius": 0.30,
    "raster_downsample": 4,
    "decoder_channels": 48,
    "output_channels": 1,
    "recurrent": true
  },
  "eval": {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventaid_r_external_ann"
  }
}
~~~~

# configs/aid_snn.json

~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventaid_r_zip",
    "root": "data/EventAid-R",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "target_offset": 1,
    "tone_map": "none"
  },
  "model": {
    "hidden_dim": 64,
    "graph_layers": 6,
    "causal_candidates": 32,
    "spatial_radius": 0.12,
    "temporal_radius": 0.30,
    "raster_downsample": 4,
    "decoder_channels": 48,
    "output_channels": 1,
    "recurrent": true
  },
  "eval": {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventaid_r_external_snn"
  }
}
~~~~

# configs/hdr_ann.json

~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/eval",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "frame_stride": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "hidden_dim": 64,
    "graph_layers": 6,
    "causal_candidates": 32,
    "spatial_radius": 0.12,
    "temporal_radius": 0.30,
    "raster_downsample": 4,
    "decoder_channels": 48,
    "output_channels": 1,
    "recurrent": true
  },
  "eval": {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventhdr_official_eval_ann"
  }
}
~~~~

# configs/hdr_snn.json

~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/eval",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "frame_stride": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "hidden_dim": 64,
    "graph_layers": 6,
    "causal_candidates": 32,
    "spatial_radius": 0.12,
    "temporal_radius": 0.30,
    "raster_downsample": 4,
    "decoder_channels": 48,
    "output_channels": 1,
    "recurrent": true
  },
  "eval": {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventhdr_official_eval_snn"
  }
}
~~~~

# configs/hdr_train.json

~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/train",
    "val_root": "data/EventHDR/train",
    "split_manifest": "manifests/eventhdr_split.json",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": [256, 256],
    "frame_stride": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "hidden_dim": 64,
    "graph_layers": 6,
    "causal_candidates": 32,
    "spatial_radius": 0.12,
    "temporal_radius": 0.30,
    "raster_downsample": 4,
    "decoder_channels": 48,
    "output_channels": 1,
    "recurrent": true
  },
  "train": {
    "epochs": 40,
    "batch_size": 1,
    "num_workers": 4,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "learning_rate": 0.0002,
    "weight_decay": 0.000001,
    "grad_clip": 1.0,
    "amp": true,
    "log_every": 20,
    "validate_every": 1,
    "resume": null,
    "max_train_samples": null,
    "max_val_samples": 500,
    "loss_weights": {
      "charbonnier": 1.0,
      "ssim": 0.2,
      "gradient": 0.1,
      "temporal": 0.2
    }
  },
  "output": {
    "run_dir": "runs/eventhdr_asgcn",
    "save_predictions": 8
  }
}
~~~~

# CONTRIBUTING.md

~~~~markdown
# 기여 가이드

버그 수정과 재현성 개선을 환영합니다. 기본 단위 테스트는 공식 데이터셋이나 GPU를 요구하지
않으며 `tests/` 내부 fixture만 사용해야 합니다.

## 개발 환경

Python 3.10 이상을 사용합니다. Linux/macOS에서는 다음과 같이 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell에서는 다음 명령을 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

GPU 학습을 개발하는 경우 환경에 맞는 CUDA PyTorch를 먼저 설치한 뒤 editable install을
실행하세요. 기본 테스트와 CI는 CPU만 사용합니다.

## 변경 전 검증

Pull request를 열기 전에 저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m ruff check .
python -m pytest -q
```

Windows와 Linux의 Python 3.10, 3.11, 3.12 조합은 GitHub Actions에서 `tests/` 내부 fixture만
사용해 검사합니다. 데이터 로더나 config 경로를 변경했다면 공식 데이터를 별도로 배치한 뒤
다음 CLI 검사도 수행합니다.

```bash
asgcn-recon inspect --config configs/hdr_train.json --samples 2
asgcn-recon inspect --config configs/hdr_ann.json --samples 2
asgcn-recon inspect --config configs/aid_ann.json --samples 2
```

## Pull request 원칙

- 한 PR에는 가능한 한 하나의 논리적 변경만 포함합니다.
- 모델, 손실 함수, 그래프 생성 또는 데이터 정렬을 변경하면 근거와 예상 영향을 설명합니다.
- 새로운 기능에는 `tests/` 내부 fixture를 사용하는 최소 단위 테스트를 추가합니다.
- CLI나 설정 형식이 바뀌면 README와 예제 config도 함께 갱신합니다.
- EventHDR 공식 eval이나 EventAid-R을 학습·튜닝에 사용하지 않습니다.
- 포매터가 아닌 Ruff 검사 결과를 기준으로 기존 코드 스타일을 유지합니다.

## 데이터와 생성물

공식 데이터셋, 압축 파일, checkpoint, 학습 로그와 대규모 출력은 Git에 커밋하지 않습니다.
재현에 필요한 메타데이터와 `tests/` 내부의 작은 fixture만 저장하세요. 버그 보고에 실제 데이터
일부가 필요하다면 먼저 배포 라이선스와 개인 정보 포함 여부를 확인하고, 가능하면 이를 대신하는
최소 fixture를 제공하세요.

## 데이터 로더 변경 시 주의사항

- EventAid-R 이벤트 구간의 target은 동일 번호가 아니라 다음 번호의 GT입니다.
- 프레임 속도를 상수로 가정하지 말고 제공된 timestamp 차이를 사용합니다.
- EventHDR train holdout과 공식 eval의 역할을 섞지 않습니다.
- 데이터 형식 검증 실패는 가능한 한 파일명과 기대한 구조를 포함한 명확한 오류로 보고합니다.
~~~~

# Dockerfile

~~~~dockerfile
FROM python:3.12-slim

ARG TORCH_INDEX_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_INDEX_URL" ]; then \
         python -m pip install torch --index-url "$TORCH_INDEX_URL"; \
       else \
         python -m pip install torch; \
       fi \
    && python -m pip install -e ".[eval]"

COPY configs ./configs
COPY manifests ./manifests
COPY scripts ./scripts
COPY server ./server

RUN mkdir -p /workspace/data /workspace/runs

ENTRYPOINT ["asgcn-recon"]
CMD ["--help"]
~~~~

# docs/EXPERIMENT.md

~~~~markdown
# 재현 가능한 실험 프로토콜

## 데이터 역할

| 단계 | 데이터 | 허용되는 사용 |
|---|---|---|
| 학습 | EventHDR train | weight 최적화와 augmentation |
| 검증 | EventHDR train holdout | checkpoint 선택과 hyperparameter 결정 |
| 내부 최종시험 | EventHDR 공식 eval | 학습 완료 후 품질·지연 평가 |
| 외부 일반화 | EventAid-R | 학습·BN·threshold 보정 없이 최종 평가만 수행 |

EventHDR 공개 train H5와 논문의 물리 장면 사이 공식 대응표가 없으므로 현재 manifest는
파일 단위 임시 분할이다. 대응표를 확보하면 동일 장면이 양쪽에 걸리지 않도록 group split으로
교체해야 한다.

## 기준 실행 순서

```bash
# 1. 데이터 구조
asgcn-recon inspect --config configs/hdr_train.json

# 2. ANN 학습
asgcn-recon train --config configs/hdr_train.json

# 3. EventHDR train으로만 BN folding 및 SNN threshold calibration
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500

# 4. ANN/SNN 내부시험
asgcn-recon evaluate \
  --config configs/hdr_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16

# 5. 잠근 외부시험
asgcn-recon evaluate \
  --config configs/aid_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16
```

## 기록 항목

- 품질: PSNR, SSIM, RMSE, 선택적으로 LPIPS
- 지연: mean, p50, p90, p95, p99, max, FPS
- 스트리밍: 실제 timestamp 기반 RTF와 deadline miss ratio
- 그래프: 평균 event/node/edge 수
- SNN: simulation step과 layer 평균 firing rate
- 시스템: GPU 모델, PyTorch/CUDA/cuDNN, peak allocated GPU memory

`evaluate`는 품질과 frame별 CSV를 저장한다. `benchmark`는 파일 I/O를 timer 밖에 두고 지정한
warmup 반복을 제외한 연산 latency를 측정한다.

## 필수 비교 실험

1. `graph_layers=0`과 ASGCN graph encoder
2. ANN과 SNN `T=4/8/16/32`
3. `max_events=4096/8192/16384`
4. `causal_candidates=16/32/64`
5. ConvGRU 사용/미사용

모든 비교는 같은 split, seed, tone mapping, crop과 평가 해상도를 사용한다. EventAid-R 결과로
설정이나 threshold를 다시 선택하면 외부시험이 아니므로 별도 탐색 실험으로 표시해야 한다.
~~~~

# docs/SERVER.md

~~~~markdown
# Linux GPU 서버 실행 가이드

MobaXterm은 서버 자체가 아니라 SSH 접속 클라이언트다. 아래 명령은 MobaXterm 터미널로
Linux 서버에 로그인한 뒤 저장소 루트에서 실행한다.

## 1. 요구 사항

- Git
- Python 3.10 이상과 `venv`
- GPU 학습 시 NVIDIA 드라이버와 CUDA를 지원하는 PyTorch wheel
- 데이터 약 50.4GB와 checkpoint를 위한 별도 여유 공간

시스템 CUDA Toolkit과 PyTorch wheel의 CUDA runtime은 같은 개념이 아니다. 서버 드라이버가
지원하는 wheel은 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
확인한다. 로그인 노드에서는 GPU가 숨겨지고 SLURM 작업 안에서만 보이는 서버도 많다.

## 2. Clone과 설치

```bash
git clone https://github.com/costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

# 공식 선택기가 제시한 CUDA wheel index가 있다면 입력하고, 없으면 Enter를 누른다.
read -r -p "PyTorch wheel index URL (Enter=PyPI default): " TORCH_INDEX_URL
export TORCH_INDEX_URL
export PROJECT_EXTRAS=dev,eval
bash scripts/setup.sh
source .venv/bin/activate
```

CPU 전용 설치나 PyPI 기본 wheel을 사용할 때는 `TORCH_INDEX_URL`을 비워 둔다. GPU가 반드시
보여야 하는 compute node에서 설치를 검사하려면 다음과 같이 실행한다.

```bash
python scripts/check_env.py --require-cuda
python -m pytest -q
```

환경 진단 결과에는 Python/PyTorch/CUDA/cuDNN/GPU 이름, 데이터 파일 수, 출력 폴더 쓰기 권한과
남은 디스크 공간이 포함된다.

## 3. 데이터 연결

대용량 데이터는 Git 저장소 밖의 공유 스토리지에 보관하고 심볼릭 링크로 연결하는 편이 좋다.
아래 `rmdir`는 설치 스크립트가 만든 **빈 디렉터리만** 제거하며, 파일이 들어 있으면 안전하게
실패한다.

```bash
rmdir data/EventHDR/train data/EventHDR/eval data/EventHDR data/EventAid-R
ln -s /shared/datasets/EventHDR data/EventHDR
ln -s /shared/datasets/EventAid-R data/EventAid-R
```

기대 구조:

```text
data/
├── EventHDR/
│   ├── train/*.h5
│   └── eval/*.h5
└── EventAid-R/
    └── R-*.zip
```

공식 EventHDR 파일을 배치한 뒤 train과 eval 구조를 CLI로 검사한다.

```bash
asgcn-recon inspect --config configs/hdr_train.json --samples 2
asgcn-recon inspect --config configs/hdr_ann.json --samples 2
```

EventAid-R는 ZIP을 풀지 않고 직접 읽는다. 로더만 확인할 때는 작은 장면 하나만 받는다.

```bash
bash scripts/get_aid.sh R-bear
asgcn-recon inspect --config configs/aid_ann.json --samples 2
```

전체 14개 장면이 필요할 때만 `bash scripts/get_aid.sh --all`을 사용한다.

## 4. 단일 GPU 서버

SSH 연결이 끊겨도 학습이 유지되도록 `tmux` 안에서 실행한다.

```bash
tmux new -s asgcn
bash scripts/train.sh configs/hdr_train.json
```

분리: `Ctrl-b`, `d`
재접속: `tmux attach -t asgcn`

중단된 학습은 optimizer, AMP scaler, RNG와 epoch가 들어 있는 `last.pt`에서 재개한다.

```bash
python -m asgcn_recon.cli train \
  --config configs/hdr_train.json \
  --resume runs/eventhdr_asgcn/last.pt
```

평가:

```bash
bash scripts/eval.sh \
  configs/hdr_ann.json \
  runs/eventhdr_asgcn/best.pt
```

EventAid-R 외부평가는 첫 번째 인자만 `configs/aid_ann.json`으로 바꾼다. SNN 평가는 결과
덮어쓰기를 막기 위해 각각 `configs/hdr_snn.json`, `configs/aid_snn.json`을 사용한다.

## 5. SLURM 클러스터

기본 예제는 GPU 1개, CPU 8개, 메모리 32GB를 요청한다. 클러스터 정책에 맞게 `#SBATCH`
값을 수정한다.

```bash
sbatch server/train.sbatch

sbatch --export=ALL,CONFIG_PATH=configs/aid_ann.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt server/eval.sbatch
```

로그는 기본적으로 `slurm-<job-name>-<job-id>.out/.err`에 기록되고 Git에서는 제외된다.

## 6. Docker

서버에 Docker와 NVIDIA Container Toolkit이 구성된 경우:

```bash
docker build \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t asgcn-event-reconstruction .

docker compose run --rm experiment \
  inspect --config configs/hdr_train.json
```

`compose.yaml`은 `DATA_DIR`을 읽기 전용 `/workspace/data`, `RUNS_DIR`을 쓰기 가능한
`/workspace/runs`로 연결한다. 데이터와 checkpoint는 이미지에 포함되지 않는다.

## 7. 흔한 문제

- `torch.cuda.is_available() == false`: CPU wheel 설치 여부, NVIDIA 드라이버, SLURM GPU 할당,
  `CUDA_VISIBLE_DEVICES`를 확인한다.
- H5/ZIP을 못 찾음: 명령을 어느 폴더에서 실행해도 checked-in config 경로는 저장소 기준으로
  해석되지만, 외부 config는 해당 config 파일 위치를 기준으로 해석된다.
- `No selected .h5`: `manifests/eventhdr_split.json`에 적힌 train/validation 파일이 실제로
  존재하는지 확인한다.
- 메모리 부족: `max_events`, `causal_candidates`, `decoder_channels` 순서로 줄인다.
- SSH 종료로 작업 중단: interactive shell 대신 tmux 또는 SLURM을 사용한다.
~~~~

# Makefile

~~~~makefile
PYTHON ?= .venv/bin/python

.PHONY: setup doctor test lint inspect train

setup:
	bash scripts/setup.sh

doctor:
	$(PYTHON) scripts/check_env.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

inspect:
	$(PYTHON) -m asgcn_recon.cli inspect --config configs/hdr_train.json

train:
	$(PYTHON) -m asgcn_recon.cli train --config configs/hdr_train.json
~~~~

# manifests/eventaid_r.json

~~~~json
{
  "source": "https://sites.google.com/view/eventaid-benchmark",
  "displayed_total_gb": 24.68024,
  "files": [
    {"scene":"R-ball","size":"5.64GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AOa2ZA-ZO27IWfKur1SU_Fw/EventAid-R/R-ball.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-bear","size":"25.81MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AKpsAqhaQuAlRGnY2FhAKqE/EventAid-R/R-bear.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-blocks","size":"365.33MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AL4wHVG3Df2bbfNCghWyJaM/EventAid-R/R-blocks.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-box","size":"119.89MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AGvV-9JnNptwOaI-KVIUSK0/EventAid-R/R-box.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-building","size":"3.03GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/ADhYTCmHnEkAwfhHaEWgzCk/EventAid-R/R-building.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-outdoor","size":"19.07MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AKP0sOL3pENW3wtyNRAX26k/EventAid-R/R-outdoor.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-playball","size":"2.47GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AF8acN06ce8FSBdw159Znk8/EventAid-R/R-playball.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-room1","size":"869.19MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AKuYq0bLROmpyoeqj7zbvSI/EventAid-R/R-room1.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-room2","size":"1.07GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/ADW-5jFnZ_JUkLGPEE1OrvI/EventAid-R/R-room2.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-sculpture","size":"526.32MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/ACFkV3fjjREExbn-KA9g-eA/EventAid-R/R-sculpture.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-toy","size":"104.63MB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AD7BEyswPqPVaMi9yRXf4jw/EventAid-R/R-toy.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-traffic","size":"6.96GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/APAzVk_7JK12KIm2KSqSWb8/EventAid-R/R-traffic.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-umbrella","size":"1.48GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AFHQxXlstNbGerXeh_sFujI/EventAid-R/R-umbrella.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"},
    {"scene":"R-wall","size":"2GB","url":"https://www.dropbox.com/scl/fo/7l4jytyqpvdf5w9x3t8zi/AFj-qvqZ7f8Ma9Q2tWXrb5k/EventAid-R/R-wall.zip?rlkey=aq4t4jg5xerfrhfkut6tddh5w&dl=1"}
  ]
}
~~~~

# manifests/eventhdr_split.json

~~~~json
{
  "note": "Provisional file-level holdout. Update after grouping files that depict the same physical scene.",
  "train_files": [
    "1.h5", "2.h5", "3.h5", "4.h5", "5.h5", "6.h5", "7.h5", "8.h5", "9.h5",
    "10.h5", "11.h5", "12.h5", "13.h5", "14.h5", "15.h5", "16.h5", "17.h5",
    "18.h5", "19.h5", "20.h5", "21.h5", "22.h5", "23.h5", "24.h5", "25.h5",
    "26.h5", "27.h5", "28.h5", "29.h5", "30.h5", "31.h5", "32.h5", "33.h5",
    "34.h5", "35.h5", "36.h5", "37.h5", "38.h5", "39.h5", "40.h5", "41.h5",
    "42.h5", "43.h5", "44.h5", "45.h5", "46.h5", "47.h5"
  ],
  "val_files": ["48.h5", "49.h5", "50.h5", "51.h5"]
}
~~~~

# pyproject.toml

~~~~toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "asgcn-reconstruction"
version = "0.1.0"
description = "ASGCN-style event-to-frame reconstruction for EventHDR and EventAid-R"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "h5py>=3.11,<4",
  "numpy>=1.26,<3",
  "Pillow>=10.2,<13",
  "torch>=2.3,<3",
  "tqdm>=4.66,<5",
]

[project.optional-dependencies]
eval = ["lpips>=0.1.4"]
dev = ["pytest>=8.0", "ruff>=0.6"]

[project.scripts]
asgcn-recon = "asgcn_recon.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"
~~~~

# README.md

~~~~markdown
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
~~~~

# requirements.txt

~~~~text
-e .[dev]
~~~~

# scripts/check_env.py

~~~~python
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

import torch


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASGCN server readiness")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or project_root / "data").resolve()
    runs_root = (args.runs_root or project_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(data_root if data_root.exists() else project_root)

    cuda_available = torch.cuda.is_available()
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    report = {
        "project_root": str(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_root": str(data_root),
        "eventhdr_train_h5": _count_files(data_root / "EventHDR" / "train", "*.h5"),
        "eventhdr_eval_h5": _count_files(data_root / "EventHDR" / "eval", "*.h5"),
        "eventaid_r_zip": _count_files(data_root / "EventAid-R", "R-*.zip"),
        "runs_root": str(runs_root),
        "runs_writable": os.access(runs_root, os.W_OK),
        "disk_free_gib": round(disk.free / (1024**3), 2),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    problems: list[str] = []
    if not report["runs_writable"]:
        problems.append(f"Run directory is not writable: {runs_root}")
    if args.require_cuda and not cuda_available:
        problems.append("CUDA was required but torch.cuda.is_available() is false")
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
~~~~

# scripts/eval.sh

~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_ann.json}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}}"
INFERENCE_MODE="${INFERENCE_MODE:-ann}"
SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: evaluation config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "${INFERENCE_MODE}" != "ann" && "${INFERENCE_MODE}" != "snn" ]]; then
  echo "ERROR: INFERENCE_MODE must be ann or snn" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${REQUIRE_CUDA}" <<'PY'
import sys
import torch

required = sys.argv[1] == "1"
available = torch.cuda.is_available()
print(f"PyTorch {torch.__version__}; CUDA runtime={torch.version.cuda}; available={available}")
if available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif required:
    raise SystemExit("CUDA GPU is required. Set REQUIRE_CUDA=0 only for a deliberate CPU run.")
PY

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${CONFIG_PATH}" --samples "${INSPECT_SAMPLES}"
fi

echo "Evaluating ${CHECKPOINT_PATH} on ${CONFIG_PATH} (${INFERENCE_MODE})"
"${PYTHON_BIN}" -m asgcn_recon.cli evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --inference-mode "${INFERENCE_MODE}" \
  --simulation-steps "${SIMULATION_STEPS}"

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  echo "Running latency benchmark"
  "${PYTHON_BIN}" -m asgcn_recon.cli benchmark \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --warmup "${BENCHMARK_WARMUP}" \
    --steps "${BENCHMARK_STEPS}" \
    --inference-mode "${INFERENCE_MODE}" \
    --simulation-steps "${SIMULATION_STEPS}"
fi
~~~~

# scripts/get_aid.ps1

~~~~powershell
param(
    [string]$Destination = ".\data\EventAid-R",
    [string[]]$Scenes = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot "manifests\eventaid_r.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Destination))
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

$selected = $manifest.files
if ($Scenes.Count -gt 0) {
    $selected = $manifest.files | Where-Object { $Scenes -contains $_.scene }
    $missing = $Scenes | Where-Object { $_ -notin $selected.scene }
    if ($missing.Count -gt 0) {
        throw "Unknown scene(s): $($missing -join ', ')"
    }
}

Write-Host "EventAid-R destination: $destinationPath"
Write-Host "Selected scenes: $($selected.scene -join ', ')"
foreach ($item in $selected) {
    $target = Join-Path $destinationPath ($item.scene + ".zip")
    Write-Host "Downloading $($item.scene) ($($item.size))"
    & curl.exe -L --fail --retry 5 --retry-delay 3 --continue-at - --output $target $item.url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($item.scene)"
    }
}
Write-Host "Done. ZIP files are read directly; do not extract them."
~~~~

# scripts/get_aid.sh

~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/manifests/eventaid_r.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DESTINATION="${EVENTAID_ROOT:-${PROJECT_ROOT}/data/EventAid-R}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/get_aid.sh [options] [SCENE ...]

With no SCENE, only the small R-bear sample is downloaded. ZIP files stay
compressed because the loader reads them directly.

Options:
  -d, --destination DIR  Download directory (default: data/EventAid-R)
  --all                  Download all 14 scenes (~24.68 GB)
  -h, --help             Show this help

Examples:
  ./scripts/get_aid.sh
  ./scripts/get_aid.sh R-bear R-outdoor
  ./scripts/get_aid.sh --all
EOF
}

DOWNLOAD_ALL=0
SCENES=()
while (($#)); do
  case "$1" in
    -d|--destination)
      if (($# < 2)); then
        echo "ERROR: $1 requires a directory" >&2
        exit 2
      fi
      DESTINATION="$2"
      shift 2
      ;;
    --all)
      DOWNLOAD_ALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SCENES+=("$@")
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SCENES+=("$1")
      shift
      ;;
  esac
done

if ((DOWNLOAD_ALL == 1)) && ((${#SCENES[@]} > 0)); then
  echo "ERROR: use either --all or explicit scene names, not both" >&2
  exit 2
fi
if ((DOWNLOAD_ALL == 0)) && ((${#SCENES[@]} == 0)); then
  SCENES=(R-bear)
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${DESTINATION}"
DESTINATION="$(cd -- "${DESTINATION}" && pwd)"

REQUESTED=("${SCENES[@]}")
if ((DOWNLOAD_ALL == 1)); then
  REQUESTED=(__ALL__)
fi

echo "EventAid-R destination: ${DESTINATION}"
SELECTION_FILE="$(mktemp)"
trap 'rm -f -- "${SELECTION_FILE}"' EXIT

"${PYTHON_BIN}" - "${MANIFEST}" "${REQUESTED[@]}" >"${SELECTION_FILE}" <<'PY'
import json
import sys

manifest_path, *requested = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    files = json.load(stream)["files"]

by_name = {item["scene"]: item for item in files}
if requested == ["__ALL__"]:
    selected = files
else:
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit("Unknown scene(s): " + ", ".join(missing))
    selected = [by_name[name] for name in requested]

for item in selected:
    print(item["scene"], item["size"], item["url"], sep="\t")
PY

while IFS=$'\t' read -r scene size url; do
  [[ -n "${scene}" ]] || continue
  target="${DESTINATION}/${scene}.zip"

  if "${PYTHON_BIN}" - "${target}" <<'PY'
import sys
import zipfile

raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
PY
  then
    echo "Already downloaded and valid: ${target}"
    continue
  fi

  echo "Downloading ${scene} (${size})"
  curl --location --fail --retry 5 --retry-delay 3 --continue-at - \
    --output "${target}" "${url}"

  "${PYTHON_BIN}" - "${target}" <<'PY'
import sys
import zipfile

path = sys.argv[1]
if not zipfile.is_zipfile(path):
    raise SystemExit(f"Downloaded file is not a valid ZIP: {path}")
PY
  echo "Verified ZIP container: ${target}"
done <"${SELECTION_FILE}"

echo "Done. Keep the ZIP files compressed; the EventAid-R loader reads them directly."
~~~~

# scripts/setup.sh

~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

# Clone 후 한 번 실행하는 Linux 서버 설치 스크립트.
# https://pytorch.org/get-started/locally/ 에서 서버 드라이버에 맞는 wheel을 고른 뒤:
#   TORCH_INDEX_URL=<official-wheel-index> ./scripts/setup.sh
# 재현용으로 버전을 고정할 때:
#   TORCH_VERSION=<version> TORCH_INDEX_URL=<official-wheel-index> \
#     PROJECT_EXTRAS=eval ./scripts/setup.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  echo "Loading installer settings: ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
TORCH_VERSION="${TORCH_VERSION:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PROJECT_EXTRAS="${PROJECT_EXTRAS:-}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${PROJECT_ROOT}/${VENV_DIR}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required; found {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

# PIP_EXTRA_ARGS is intentionally optional. It is split on spaces, so paths with
# spaces should instead be configured through pip.conf.
PIP_ARGS=()
if [[ -n "${PIP_EXTRA_ARGS:-}" ]]; then
  read -r -a PIP_ARGS <<<"${PIP_EXTRA_ARGS}"
fi

TORCH_SPEC="torch"
if [[ -n "${TORCH_VERSION}" ]]; then
  TORCH_SPEC="torch==${TORCH_VERSION}"
fi

# Installing torch first preserves an explicitly chosen CUDA wheel when the
# editable project (which declares torch>=2.3) is installed below.
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  "${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" \
    --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
else
  "${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" "${TORCH_SPEC}"
fi

INSTALL_TARGET="${PROJECT_ROOT}"
if [[ -n "${PROJECT_EXTRAS}" ]]; then
  INSTALL_TARGET="${PROJECT_ROOT}[${PROJECT_EXTRAS}]"
fi
"${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" -e "${INSTALL_TARGET}"

mkdir -p \
  "${PROJECT_ROOT}/data/EventHDR/train" \
  "${PROJECT_ROOT}/data/EventHDR/eval" \
  "${PROJECT_ROOT}/data/EventAid-R" \
  "${PROJECT_ROOT}/runs"

"${VENV_PYTHON}" - "${REQUIRE_CUDA}" <<'PY'
import sys
import torch

required = sys.argv[1] == "1"
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available now: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif required:
    raise SystemExit(
        "CUDA is required but unavailable. Run this check inside a GPU allocation, "
        "and verify TORCH_INDEX_URL and the NVIDIA driver."
    )
else:
    print("NOTE: no GPU is visible in this shell; login nodes commonly hide GPUs.")
PY

echo
echo "Installation complete."
echo "Python: ${VENV_PYTHON}"
echo "Next: ./scripts/get_aid.sh R-bear"
echo "Then place EventHDR H5 files under data/EventHDR/train and data/EventHDR/eval."
~~~~

# scripts/train.sh

~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_train.json}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: training config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${REQUIRE_CUDA}" <<'PY'
import sys
import torch

required = sys.argv[1] == "1"
available = torch.cuda.is_available()
print(f"PyTorch {torch.__version__}; CUDA runtime={torch.version.cuda}; available={available}")
if available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif required:
    raise SystemExit("CUDA GPU is required. Set REQUIRE_CUDA=0 only for a deliberate CPU run.")
PY

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${CONFIG_PATH}" --samples "${INSPECT_SAMPLES}"
fi

echo "Starting EventHDR training with ${CONFIG_PATH}"
TRAIN_ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_CHECKPOINT}"
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
exec "${PYTHON_BIN}" -m asgcn_recon.cli train "${TRAIN_ARGS[@]}"
~~~~

# server/eval.sbatch

~~~~bash
#!/usr/bin/env bash
#SBATCH --job-name=asgcn-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_ann.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}"
export INFERENCE_MODE="${INFERENCE_MODE:-ann}"
export SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
export RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
export BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
export BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
nvidia-smi || true

srun bash "${PROJECT_ROOT}/scripts/eval.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}"
~~~~

# server/train.sbatch

~~~~bash
#!/usr/bin/env bash
#SBATCH --job-name=asgcn-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Cluster-specific modules remain opt-in:
#   sbatch --export=ALL,CUDA_MODULE=cuda/12.4 server/train.sbatch
if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_train.json}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_PATH}"
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  echo "Resume: ${RESUME_CHECKPOINT}"
fi
nvidia-smi || true

srun bash "${PROJECT_ROOT}/scripts/train.sh" "${CONFIG_PATH}"
~~~~

# src/asgcn_recon/__init__.py

~~~~python
"""ASGCN-style event-to-frame reconstruction."""

__version__ = "0.1.0"
~~~~

# src/asgcn_recon/cli.py

~~~~python
from __future__ import annotations

import argparse
import json
from typing import Any

from .data import build_dataset
from .engine import benchmark, calibrate, evaluate, train
from .utils import experiment_base_dir, load_json, resolve_experiment_paths, resolve_path


def _inspect_one_split(dataset: Any, samples: int) -> dict[str, Any]:
    details = []
    for index in range(min(samples, len(dataset))):
        item = dataset[index]
        details.append(
            {
                "sample_id": item["sample_id"],
                "events": int(item["events"].shape[0]),
                "target_shape": list(item["target"].shape),
                "sensor_size": list(item["sensor_size"]),
                "metadata": item["metadata"],
            }
        )
    result: dict[str, Any] = {"samples": len(dataset), "preview": details}
    if hasattr(dataset, "scene_info"):
        result["scenes"] = dataset.scene_info
    if hasattr(dataset, "files"):
        result["files"] = len(dataset.files)
    return result


def inspect_dataset(config: dict[str, Any], samples: int = 3) -> dict[str, Any]:
    data_config = config["dataset"]
    result: dict[str, Any] = {
        "dataset_type": data_config["type"],
        "root": data_config["root"],
    }
    if data_config["type"] == "eventhdr" and data_config.get("split_manifest"):
        split_details: dict[str, Any] = {}
        for split in ("train", "val"):
            dataset = build_dataset(data_config, split=split)
            try:
                split_details[split] = _inspect_one_split(dataset, samples)
            finally:
                if hasattr(dataset, "close"):
                    dataset.close()
        result["splits"] = split_details
        result["samples"] = sum(detail["samples"] for detail in split_details.values())
        return result

    dataset = build_dataset(data_config, split="eval")
    try:
        result.update(_inspect_one_split(dataset, samples))
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASGCN Event-to-Frame experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect", help="validate dataset structure")
    inspect_cmd.add_argument("--config", required=True)
    inspect_cmd.add_argument("--samples", type=int, default=3)

    train_cmd = subparsers.add_parser("train", help="train on EventHDR")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument(
        "--resume",
        help="resume from a checkpoint (relative paths are resolved from the repository root)",
    )

    eval_cmd = subparsers.add_parser("evaluate", help="evaluate quality and latency")
    eval_cmd.add_argument("--config", required=True)
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--inference-mode", choices=["ann", "snn"], default="ann")
    eval_cmd.add_argument("--simulation-steps", type=int, default=16)

    bench_cmd = subparsers.add_parser("benchmark", help="benchmark compute-only latency")
    bench_cmd.add_argument("--config", required=True)
    bench_cmd.add_argument("--checkpoint", required=True)
    bench_cmd.add_argument("--warmup", type=int, default=10)
    bench_cmd.add_argument("--steps", type=int, default=100)
    bench_cmd.add_argument("--inference-mode", choices=["ann", "snn"], default="ann")
    bench_cmd.add_argument("--simulation-steps", type=int, default=16)

    calibrate_cmd = subparsers.add_parser("calibrate", help="calibrate ANN-to-SNN thresholds")
    calibrate_cmd.add_argument("--config", required=True)
    calibrate_cmd.add_argument("--checkpoint", required=True)
    calibrate_cmd.add_argument("--output", required=True)
    calibrate_cmd.add_argument("--samples", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = resolve_path(args.config, ".")
    config = resolve_experiment_paths(load_json(config_path), config_path)
    base_dir = experiment_base_dir(config_path)
    if args.command == "inspect":
        result = inspect_dataset(config, args.samples)
    elif args.command == "train":
        resume = resolve_path(args.resume, base_dir) if args.resume else None
        result = {"best_checkpoint": str(train(config, resume_from=resume))}
    elif args.command == "evaluate":
        result = evaluate(
            config,
            resolve_path(args.checkpoint, base_dir),
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
        )
    elif args.command == "benchmark":
        result = benchmark(
            config,
            resolve_path(args.checkpoint, base_dir),
            warmup=args.warmup,
            steps=args.steps,
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
        )
    elif args.command == "calibrate":
        result = {
            "calibrated_checkpoint": str(
                calibrate(
                    config,
                    resolve_path(args.checkpoint, base_dir),
                    resolve_path(args.output, base_dir),
                    samples=args.samples,
                )
            )
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
~~~~

# src/asgcn_recon/data/__init__.py

~~~~python
from .eventaid_r import EventAidRZipDataset
from .eventhdr import EventHDRDataset
from .factory import build_dataset, collate_samples

__all__ = ["EventAidRZipDataset", "EventHDRDataset", "build_dataset", "collate_samples"]
~~~~

# src/asgcn_recon/data/common.py

~~~~python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int


def normalize_polarity(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32, copy=False)
    if p.size and p.min() >= 0:
        p = p * 2.0 - 1.0
    return np.where(p >= 0, 1.0, -1.0).astype(np.float32, copy=False)


def stratified_subsample(events: np.ndarray, max_events: int | None) -> np.ndarray:
    """Keep temporal coverage while bounding graph memory."""
    if max_events is None or max_events <= 0 or len(events) <= max_events:
        return events.astype(np.float32, copy=False)
    indices = np.linspace(0, len(events) - 1, max_events, dtype=np.int64)
    return events[indices].astype(np.float32, copy=False)


def choose_crop(
    image_height: int,
    image_width: int,
    crop_size: tuple[int, int] | None,
    random_crop: bool,
    rng: np.random.Generator,
) -> Crop:
    if crop_size is None:
        return Crop(0, 0, image_width, image_height)
    crop_h = min(int(crop_size[0]), image_height)
    crop_w = min(int(crop_size[1]), image_width)
    if random_crop:
        top = int(rng.integers(0, image_height - crop_h + 1))
        left = int(rng.integers(0, image_width - crop_w + 1))
    else:
        top = (image_height - crop_h) // 2
        left = (image_width - crop_w) // 2
    return Crop(left, top, crop_w, crop_h)


def crop_events(events: np.ndarray, crop: Crop) -> np.ndarray:
    if len(events) == 0:
        return events
    x, y = events[:, 0], events[:, 1]
    keep = (
        (x >= crop.left)
        & (x < crop.left + crop.width)
        & (y >= crop.top)
        & (y < crop.top + crop.height)
    )
    result = events[keep].copy()
    result[:, 0] -= crop.left
    result[:, 1] -= crop.top
    return result


def image_array_to_tensor(
    image: np.ndarray,
    target_channels: int = 1,
    tone_map: str = "none",
    tone_map_mu: float = 5000.0,
) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")

    original_dtype = image.dtype
    image = image.astype(np.float32, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        image /= float(np.iinfo(original_dtype).max)
    elif image.size and (float(np.nanmax(image)) > 1.0 or float(np.nanmin(image)) < 0.0):
        lo, hi = np.nanpercentile(image, [0.1, 99.9])
        image = (image - lo) / max(float(hi - lo), 1e-6)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)

    if target_channels == 1 and image.shape[-1] != 1:
        rgb = image[..., :3]
        image = (
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        )[..., None]
    elif target_channels == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > target_channels:
        image = image[..., :target_channels]

    if tone_map == "log":
        image = np.log1p(tone_map_mu * image) / np.log1p(tone_map_mu)
    elif tone_map not in {"none", "linear"}:
        raise ValueError(f"Unknown tone_map: {tone_map}")
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def make_sample(
    events: np.ndarray,
    target: torch.Tensor,
    sample_id: str,
    sensor_size: tuple[int, int],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Events must have shape Nx4 [x,y,t,p], got {events.shape}")
    event_tensor = torch.from_numpy(np.ascontiguousarray(events)).float()
    return {
        "events": event_tensor,
        "target": target,
        "sample_id": sample_id,
        "sensor_size": tuple(int(v) for v in sensor_size),
        "metadata": metadata or {},
    }
~~~~

# src/asgcn_recon/data/eventaid_r.py

~~~~python
from __future__ import annotations

import io
import os
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .common import (
    choose_crop,
    crop_events,
    image_array_to_tensor,
    make_sample,
    normalize_polarity,
    pil_to_array,
    stratified_subsample,
)

_EVENT_RE = re.compile(r"(?:^|/)event/(\d+)\.txt$", re.IGNORECASE)
_GT_RE = re.compile(r"(?:^|/)gt/(\d+)_img\.png$", re.IGNORECASE)


class EventAidRZipDataset(Dataset):
    """Read official EventAid-R scene ZIPs directly to avoid an extracted copy."""

    def __init__(
        self,
        root: str | Path,
        target_channels: int = 1,
        max_events: int | None = 8192,
        crop_size: list[int] | tuple[int, int] | None = None,
        target_offset: int = 1,
        tone_map: str = "none",
        tone_map_mu: float = 5000.0,
        random_crop: bool = False,
        seed: int = 2026,
    ) -> None:
        self.root = Path(root).expanduser()
        self.target_channels = int(target_channels)
        self.max_events = max_events
        self.crop_size = tuple(crop_size) if crop_size else None
        self.target_offset = int(target_offset)
        self.tone_map = tone_map
        self.tone_map_mu = float(tone_map_mu)
        self.random_crop = random_crop
        self.seed = int(seed)
        self._handles: dict[Path, zipfile.ZipFile] = {}
        self._owner_pid = os.getpid()
        self.zip_paths = sorted(self.root.glob("R-*.zip"))
        if not self.zip_paths:
            raise FileNotFoundError(f"No R-*.zip files found under {self.root}")
        self.samples, self.scene_info = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No paired EventAid-R samples found under {self.root}")

    @staticmethod
    def _read_shape(zf: zipfile.ZipFile, names: list[str]) -> tuple[int, int] | None:
        shape_name = next((name for name in names if name.lower().endswith("shape.txt")), None)
        if not shape_name:
            return None
        values = zf.read(shape_name).decode("utf-8", errors="replace").split()
        if len(values) < 2:
            return None
        width, height = int(values[0]), int(values[1])
        return height, width

    def _build_index(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        scene_info: dict[str, Any] = {}
        for path in self.zip_paths:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                events = {int(m.group(1)): name for name in names if (m := _EVENT_RE.search(name))}
                targets = {int(m.group(1)): name for name in names if (m := _GT_RE.search(name))}
                shape = self._read_shape(zf, names)
                timestamps_name = next(
                    (name for name in names if name.lower().endswith("timestamps.txt")), None
                )
                timestamps: list[int] = []
                if timestamps_name:
                    timestamps = [
                        int(value)
                        for value in zf.read(timestamps_name).decode("utf-8").split()
                    ]
                scene = path.stem
                scene_info[scene] = {"shape": shape, "frames": len(targets), "events": len(events)}
                for event_id in sorted(events):
                    target_id = event_id + self.target_offset
                    if target_id not in targets:
                        continue
                    samples.append(
                        {
                            "path": path,
                            "scene": scene,
                            "frame_id": event_id,
                            "event_name": events[event_id],
                            "target_name": targets[target_id],
                            "shape": shape,
                            "t0_us": timestamps[event_id - 1]
                            if 0 <= event_id - 1 < len(timestamps)
                            else None,
                            "t1_us": timestamps[event_id]
                            if 0 <= event_id < len(timestamps)
                            else None,
                        }
                    )
        return samples, scene_info

    def __len__(self) -> int:
        return len(self.samples)

    def _get_handle(self, path: Path) -> zipfile.ZipFile:
        process_id = os.getpid()
        if process_id != self._owner_pid:
            # Keep one independent archive descriptor per DataLoader worker.
            self._handles = {}
            self._owner_pid = process_id
        if path not in self._handles:
            self._handles[path] = zipfile.ZipFile(path)
        return self._handles[path]

    @staticmethod
    def _read_events(raw: bytes) -> np.ndarray:
        values = np.fromstring(raw.decode("ascii", errors="ignore"), sep=" ", dtype=np.float64)
        usable = (values.size // 4) * 4
        if usable == 0:
            return np.empty((0, 4), dtype=np.float32)
        rows = values[:usable].reshape(-1, 4)
        # Official text columns: timestamp, x, y, polarity.
        events = rows[:, [1, 2, 0, 3]]
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1.0)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        events[:, 3] = normalize_polarity(events[:, 3])
        return events

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        zf = self._get_handle(item["path"])
        events = self._read_events(zf.read(item["event_name"]))
        with Image.open(io.BytesIO(zf.read(item["target_name"]))) as image:
            target = image_array_to_tensor(
                pil_to_array(image),
                self.target_channels,
                tone_map=self.tone_map,
                tone_map_mu=self.tone_map_mu,
            )
        height, width = target.shape[-2:]
        if item["shape"] and item["shape"] != (height, width):
            raise ValueError(
                f"{item['scene']} shape.txt={item['shape']} but target={(height, width)}"
            )
        scene_seed = zlib.crc32(item["scene"].encode("utf-8"))
        rng = np.random.default_rng(self.seed + scene_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        events = stratified_subsample(events, self.max_events)
        sample_id = f"{item['scene']}/{item['frame_id']:06d}"
        return make_sample(
            events,
            target,
            sample_id,
            (crop.height, crop.width),
            {
                "dataset": "EventAid-R",
                "scene": item["scene"],
                "source": str(item["path"]),
                "t0_us": item["t0_us"],
                "t1_us": item["t1_us"],
                "dt_us": (item["t1_us"] - item["t0_us"])
                if item["t0_us"] is not None and item["t1_us"] is not None
                else None,
            },
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for handle in handles.values():
            try:
                handle.close()
            except OSError:
                pass
        handles.clear()

    def __del__(self) -> None:
        self.close()
~~~~

# src/asgcn_recon/data/eventhdr.py

~~~~python
from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

from .common import (
    choose_crop,
    crop_events,
    image_array_to_tensor,
    make_sample,
    normalize_polarity,
    stratified_subsample,
)


class EventHDRDataset(Dataset):
    """Read the official EventHDR HDF5 structure without preprocessing copies."""

    def __init__(
        self,
        root: str | Path,
        target_channels: int = 1,
        max_events: int | None = 8192,
        crop_size: list[int] | tuple[int, int] | None = None,
        frame_stride: int = 1,
        tone_map: str = "log",
        tone_map_mu: float = 5000.0,
        random_crop: bool = False,
        seed: int = 2026,
        allowed_files: list[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.target_channels = int(target_channels)
        self.max_events = max_events
        self.crop_size = tuple(crop_size) if crop_size else None
        self.frame_stride = max(1, int(frame_stride))
        self.tone_map = tone_map
        self.tone_map_mu = float(tone_map_mu)
        self.random_crop = random_crop
        self.seed = int(seed)
        self._handles: dict[Path, h5py.File] = {}
        self._owner_pid = os.getpid()
        discovered = sorted([*self.root.rglob("*.h5"), *self.root.rglob("*.hdf5")])
        if not discovered:
            raise FileNotFoundError(
                f"No EventHDR .h5/.hdf5 files found under {self.root}. "
                "Place the official files in this directory or update dataset.root."
            )
        self.files = discovered
        if allowed_files is not None:
            allowed = set(allowed_files)
            present = {path.name for path in discovered}
            missing = sorted(allowed - present)
            if missing:
                preview = ", ".join(missing[:8])
                suffix = " ..." if len(missing) > 8 else ""
                raise FileNotFoundError(
                    f"EventHDR split requires {len(allowed)} files but {len(missing)} are "
                    f"missing under {self.root}: {preview}{suffix}"
                )
            self.files = [path for path in self.files if path.name in allowed]
        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No valid EventHDR frames found under {self.root}")

    def _build_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.files:
            with h5py.File(path, "r") as h5:
                if "events" not in h5 or "images" not in h5:
                    continue
                image_keys = sorted(k for k in h5["images"] if k.startswith("image"))
                selected_start_idx = 0
                selected_start_timestamp: float | None = None
                for frame_index, key in enumerate(image_keys):
                    node = h5["images"][key]
                    end_idx = int(node.attrs.get("event_idx", selected_start_idx))
                    timestamp = float(node.attrs.get("timestamp", frame_index))
                    if (
                        frame_index % self.frame_stride == 0
                        and end_idx > selected_start_idx
                    ):
                        samples.append(
                            {
                                "path": path,
                                "image_key": key,
                                "start_idx": selected_start_idx,
                                "end_idx": end_idx,
                                "t0": selected_start_timestamp,
                                "timestamp": timestamp,
                            }
                        )
                        # With frame_stride > 1, aggregate every skipped event interval
                        # into the next selected output instead of silently discarding it.
                        selected_start_idx = end_idx
                        selected_start_timestamp = timestamp
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_handle(self, path: Path) -> h5py.File:
        process_id = os.getpid()
        if process_id != self._owner_pid:
            # DataLoader workers must never reuse HDF5 objects inherited through fork.
            self._handles = {}
            self._owner_pid = process_id
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        h5 = self._get_handle(item["path"])
        start, end = item["start_idx"], item["end_idx"]
        xs = np.asarray(h5["events/xs"][start:end], dtype=np.float32)
        ys = np.asarray(h5["events/ys"][start:end], dtype=np.float32)
        ts = np.asarray(h5["events/ts"][start:end], dtype=np.float64)
        ps = normalize_polarity(np.asarray(h5["events/ps"][start:end]))
        events = np.column_stack((xs, ys, ts, ps))
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1e-9)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        image = np.asarray(h5["images"][item["image_key"]])
        target = image_array_to_tensor(
            image,
            self.target_channels,
            tone_map=self.tone_map,
            tone_map_mu=self.tone_map_mu,
        )
        height, width = target.shape[-2:]
        scene_seed = zlib.crc32(str(item["path"]).encode("utf-8"))
        rng = np.random.default_rng(self.seed + scene_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        events = stratified_subsample(events, self.max_events)
        sample_id = f"{item['path'].stem}/{item['image_key']}"
        t0 = item["t0"]
        t1 = item["timestamp"]
        return make_sample(
            events,
            target,
            sample_id,
            (crop.height, crop.width),
            {
                "dataset": "EventHDR",
                "timestamp": t1,
                "t0": t0,
                "t1": t1,
                "dt_us": round((t1 - t0) * 1_000_000) if t0 is not None else None,
                "source": str(item["path"]),
                "scene": item["path"].stem,
            },
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for handle in handles.values():
            try:
                handle.close()
            except (OSError, RuntimeError):
                pass
        handles.clear()

    def __del__(self) -> None:
        self.close()
~~~~

# src/asgcn_recon/data/factory.py

~~~~python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .eventaid_r import EventAidRZipDataset
from .eventhdr import EventHDRDataset


def build_dataset(config: dict[str, Any], split: str = "train"):
    cfg = dict(config)
    dataset_type = cfg.pop("type")
    root = cfg.pop("root")
    cfg.pop("val_root", None)
    split_manifest = cfg.pop("split_manifest", None)
    cfg["random_crop"] = split == "train" and cfg.get("crop_size") is not None
    if dataset_type == "eventhdr":
        if split_manifest and split in {"train", "val", "calibration"}:
            manifest_path = Path(split_manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"EventHDR split manifest does not exist: {manifest_path}. "
                    "Paths in checked-in configs are resolved from the repository root."
                )
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            key = "val_files" if split == "val" else "train_files"
            if key not in manifest:
                raise KeyError(
                    f"EventHDR split manifest {manifest_path} has no '{key}' list "
                    f"for split='{split}'"
                )
            if not isinstance(manifest[key], list) or not manifest[key]:
                raise ValueError(
                    f"EventHDR split manifest {manifest_path} field '{key}' must be "
                    "a non-empty list of HDF5 filenames"
                )
            cfg["allowed_files"] = manifest[key]
        return EventHDRDataset(root=root, **cfg)
    if dataset_type == "eventaid_r_zip":
        return EventAidRZipDataset(root=root, **cfg)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Graphs and sensor resolutions are variable-sized; the model loops over this small list.
    return batch
~~~~

# src/asgcn_recon/engine.py

~~~~python
from __future__ import annotations

import copy
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset, collate_samples
from .losses import ReconstructionLoss
from .metrics import MetricAccumulator, frame_metrics, percentile
from .model import ASGCNReconstructor
from .utils import (
    atomic_torch_save,
    move_sample,
    resolve_device,
    save_image,
    save_json,
    set_seed,
    write_frame_csv,
)


def build_model(config: dict[str, Any]) -> ASGCNReconstructor:
    return ASGCNReconstructor(**config)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[ASGCNReconstructor, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config", fallback_model_config)
    model = build_model(model_config).to(device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _data_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
):
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_samples,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = (
            True if persistent_workers is None else bool(persistent_workers)
        )
        loader_options["worker_init_fn"] = _seed_worker
        if prefetch_factor is not None:
            if int(prefetch_factor) < 1:
                raise ValueError("prefetch_factor must be at least 1")
            loader_options["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_options)


def _loader_kwargs(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "persistent_workers": section.get("persistent_workers"),
        "prefetch_factor": section.get("prefetch_factor"),
    }


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch before the unified torch.amp API.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _validation_dataset(config: dict[str, Any]):
    data_config = copy.deepcopy(config["dataset"])
    data_config["root"] = data_config.get("val_root", data_config["root"])
    return build_dataset(data_config, split="val")


@torch.no_grad()
def validate(
    model: ASGCNReconstructor,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    current_scene = None
    recurrent_state = None
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        if len(batch) != 1:
            raise ValueError("Stateful validation currently requires batch_size=1")
        sample = move_sample(batch[0], device)
        scene = str(sample["metadata"].get("scene", "unknown"))
        if scene != current_scene:
            recurrent_state = None
            current_scene = scene
        prediction, diagnostics = model.forward_sample(sample, recurrent_state=recurrent_state)
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        target = sample["target"].unsqueeze(0)
        accumulator.update(scene, sample["sample_id"], frame_metrics(prediction, target))
    return accumulator.summary()["micro"]


def train(
    config: dict[str, Any], resume_from: str | Path | None = None
) -> Path:
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    train_config = config["train"]
    data_config = copy.deepcopy(config["dataset"])
    train_dataset = build_dataset(data_config, split="train")
    val_dataset = _validation_dataset(config)
    batch_size = int(train_config.get("batch_size", 1))
    if batch_size != 1 and config["model"].get("recurrent", True):
        raise ValueError("The recurrent experiment uses chronological batch_size=1")
    train_loader = _data_loader(
        train_dataset,
        batch_size,
        int(train_config.get("num_workers", 0)),
        device,
        shuffle=False,
        **_loader_kwargs(train_config),
    )
    val_loader = _data_loader(
        val_dataset,
        1,
        int(train_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(train_config),
    )

    resume_path = resume_from or train_config.get("resume")
    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        model, resume_checkpoint = load_model_checkpoint(
            resume_path, device, config["model"]
        )
    else:
        model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-6)),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    criterion = ReconstructionLoss(train_config.get("loss_weights"))
    temporal_weight = float(train_config.get("loss_weights", {}).get("temporal", 0.0))
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    best_ssim = float("-inf")
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_checkpoint is not None:
        if "optimizer" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has model weights but no optimizer state; "
                "it cannot be used for exact training resume"
            )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        _optimizer_to(optimizer, device)
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_ssim = float(
            resume_checkpoint.get(
                "best_ssim",
                resume_checkpoint.get("val", {}).get("ssim", float("-inf")),
            )
        )
        history = list(resume_checkpoint.get("history", []))
        _restore_rng_state(resume_checkpoint.get("rng_state"))

    epochs = int(train_config.get("epochs", 40))
    validate_every = max(1, int(train_config.get("validate_every", 1)))
    max_train_samples = train_config.get("max_train_samples")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        current_scene = None
        recurrent_state = None
        previous_prediction = None
        previous_target = None
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"train {epoch:03d}/{epochs:03d}")
        for step, batch in enumerate(progress):
            if max_train_samples is not None and seen >= int(max_train_samples):
                break
            if len(batch) != 1:
                raise ValueError("Stateful training currently requires batch_size=1")
            sample = move_sample(batch[0], device)
            scene = str(sample["metadata"].get("scene", "unknown"))
            if scene != current_scene:
                recurrent_state = None
                previous_prediction = None
                previous_target = None
                current_scene = scene
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction, diagnostics = model.forward_sample(
                    sample, recurrent_state=recurrent_state
                )
                target = sample["target"].unsqueeze(0)
                loss, loss_parts = criterion(prediction, target)
                if temporal_weight > 0 and previous_prediction is not None:
                    temporal = F.l1_loss(
                        prediction - previous_prediction,
                        target - previous_target,
                    )
                    loss = loss + temporal_weight * temporal
                    loss_parts["temporal"] = float(temporal.detach().cpu())
                    loss_parts["total"] = float(loss.detach().cpu())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()

            recurrent_state = diagnostics["recurrent_state"]
            if recurrent_state is not None:
                recurrent_state = recurrent_state.detach()
            previous_prediction = prediction.detach()
            previous_target = target.detach()
            running_loss += float(loss.detach().cpu())
            seen += 1
            if step % int(train_config.get("log_every", 20)) == 0:
                progress.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}", **loss_parts)

        should_validate = epoch % validate_every == 0 or epoch == epochs
        val_metrics = (
            validate(
                model,
                val_loader,
                device,
                max_samples=train_config.get("max_val_samples"),
            )
            if should_validate
            else {}
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val": val_metrics,
        }
        history.append(record)
        save_json(run_dir / "history.json", history)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": (
                resume_checkpoint.get("model_config", config["model"])
                if resume_checkpoint is not None
                else config["model"]
            ),
            "config": config,
            "val": val_metrics,
            "best_ssim": best_ssim,
            "history": history,
            "rng_state": _capture_rng_state(),
        }
        if val_metrics.get("ssim", float("-inf")) > best_ssim:
            best_ssim = val_metrics["ssim"]
            checkpoint["best_ssim"] = best_ssim
            atomic_torch_save(checkpoint, run_dir / "best.pt")
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        print(record)
    best_path = run_dir / "best.pt"
    if best_path.is_file():
        return best_path
    if start_epoch > epochs and resume_path is not None:
        return Path(resume_path)
    return run_dir / "last.pt"


def _maybe_lpips(enabled: bool, device: torch.device):
    if not enabled:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requested. Install with: pip install -e '.[eval]'") from exc
    return lpips.LPIPS(net="alex").to(device).eval()


@torch.no_grad()
def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    eval_config = config.get("eval", {})
    loader = _data_loader(
        dataset,
        1,
        int(eval_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(eval_config),
    )
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    lpips_model = _maybe_lpips(bool(eval_config.get("lpips", False)), device)
    accumulator = MetricAccumulator()
    frame_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    realtime_factors: list[float] = []
    current_scene = None
    recurrent_state = None
    output_dir = Path(eval_config.get("output_dir", "runs/evaluation"))
    save_limit = int(eval_config.get("save_predictions", 0))
    max_samples = eval_config.get("max_samples")
    saved = 0
    for index, batch in enumerate(tqdm(loader, desc=f"evaluate-{inference_mode}")):
        if max_samples is not None and index >= int(max_samples):
            break
        sample = move_sample(batch[0], device)
        scene = str(sample["metadata"].get("scene", "unknown"))
        if scene != current_scene:
            recurrent_state = None
            current_scene = scene
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction, diagnostics = model.forward_sample(
            sample,
            inference_mode=inference_mode,
            simulation_steps=simulation_steps,
            recurrent_state=recurrent_state,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        target = sample["target"].unsqueeze(0)
        metrics = frame_metrics(prediction, target, lpips_model)
        accumulator.update(scene, sample["sample_id"], metrics)
        dt_us = sample["metadata"].get("dt_us")
        rtf = latency_ms / (float(dt_us) / 1000.0) if dt_us else None
        if rtf is not None:
            realtime_factors.append(rtf)
        row = {
            "scene": scene,
            "sample_id": sample["sample_id"],
            **metrics,
            "latency_ms": latency_ms,
            "rtf": rtf,
            "events": int(sample["events"].shape[0]),
            "nodes": diagnostics["nodes"],
            "edges": diagnostics["edges"],
        }
        frame_rows.append(row)
        latencies.append(latency_ms)
        if saved < save_limit:
            safe_name = sample["sample_id"].replace("/", "_")
            save_image(output_dir / "predictions" / f"{safe_name}_pred.png", prediction)
            save_image(output_dir / "predictions" / f"{safe_name}_gt.png", target)
            saved += 1

    quality = accumulator.summary()
    latency = _latency_summary(latencies)
    latency["deadline_miss_ratio"] = (
        sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
        if realtime_factors
        else None
    )
    latency["rtf_p95"] = percentile(realtime_factors, 0.95) if realtime_factors else None
    result = {
        "dataset": config["dataset"]["type"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "quality": quality,
        "latency": latency,
    }
    save_json(output_dir / "metrics.json", result)
    write_frame_csv(output_dir / "frames.csv", frame_rows)
    return result


def _latency_summary(latencies: list[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {"frames": 0}
    mean = statistics.fmean(latencies)
    return {
        "frames": len(latencies),
        "mean_ms": mean,
        "p50_ms": percentile(latencies, 0.50),
        "p90_ms": percentile(latencies, 0.90),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "max_ms": max(latencies),
        "fps": 1000.0 / mean,
    }


@torch.no_grad()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warmup: int = 10,
    steps: int = 100,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if inference_mode == "snn" and simulation_steps < 1:
        raise ValueError("simulation_steps must be at least 1 for SNN inference")
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    model, _ = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    cuda_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    cuda_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    latencies: list[float] = []
    event_counts: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    firing_rates: list[float] = []
    realtime_factors: list[float] = []
    recurrent_state = None
    current_scene = None
    total = warmup + steps
    for iteration in range(total):
        if iteration > 0 and iteration % len(dataset) == 0:
            recurrent_state = None
            current_scene = None
        if iteration == warmup and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        raw = dataset[iteration % len(dataset)]  # I/O intentionally outside the timer.
        sample = move_sample(raw, device)
        scene = str(sample["metadata"].get("scene", "unknown"))
        if scene != current_scene:
            recurrent_state = None
            current_scene = scene
        if cuda_start is not None:
            cuda_start.record()
        else:
            start = time.perf_counter()
        _, diagnostics = model.forward_sample(
            sample,
            inference_mode=inference_mode,
            simulation_steps=simulation_steps,
            recurrent_state=recurrent_state,
        )
        if cuda_end is not None:
            cuda_end.record()
            cuda_end.synchronize()
            elapsed_ms = float(cuda_start.elapsed_time(cuda_end))
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if iteration >= warmup:
            latencies.append(elapsed_ms)
            event_counts.append(int(sample["events"].shape[0]))
            node_counts.append(diagnostics["nodes"])
            edge_counts.append(diagnostics["edges"])
            firing_rates.extend(
                float(value.detach().cpu())
                if torch.is_tensor(value)
                else float(value)
                for value in diagnostics["firing_rates"]
            )
            dt_us = sample["metadata"].get("dt_us")
            if dt_us:
                realtime_factors.append(elapsed_ms / (float(dt_us) / 1000.0))
    result: dict[str, Any] = {
        **_latency_summary(latencies),
        "events_per_second": sum(event_counts) / (sum(latencies) / 1000.0),
        "mean_nodes": statistics.fmean(node_counts),
        "mean_edges": statistics.fmean(edge_counts),
        "mean_firing_rate": statistics.fmean(firing_rates) if firing_rates else None,
        "deadline_miss_ratio": (
            sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
            if realtime_factors
            else None
        ),
        "rtf_p95": percentile(realtime_factors, 0.95) if realtime_factors else None,
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter",
        "io_excluded": True,
    }
    return result


@torch.no_grad()
def calibrate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples: int = 100,
) -> Path:
    device = resolve_device(config.get("device", "auto"))
    data_config = copy.deepcopy(config["dataset"])
    # Calibration is restricted to EventHDR train, never EventAid-R.
    if data_config["type"] != "eventhdr":
        raise ValueError("SNN calibration must use EventHDR training data")
    dataset = build_dataset(data_config, split="calibration")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    model.fold_batch_norm()
    model.encoder.reset_thresholds()
    for index in tqdm(range(min(samples, len(dataset))), desc="calibrate-SNN"):
        sample = move_sample(dataset[index], device)
        model.calibrate_sample(sample, momentum=-1.0)
    checkpoint["model"] = model.state_dict()
    checkpoint["snn_calibration_samples"] = min(samples, len(dataset))
    checkpoint["batch_norm_folded"] = True
    output_path = Path(output_path)
    atomic_torch_save(checkpoint, output_path)
    return output_path
~~~~

# src/asgcn_recon/graph.py

~~~~python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EventGraph:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


def _safe_batch_norm(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
    """Use running statistics when a graph has fewer than two real events."""
    if norm.training and values.shape[0] < 2:
        return F.batch_norm(
            values,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=0.0,
            eps=norm.eps,
        )
    return norm(values)


def prepare_event_nodes(
    events: torch.Tensor, sensor_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize raw [x,y,t,p] while retaining event order."""
    height, width = sensor_size
    if events.numel() == 0:
        return (
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
            torch.empty((0, 3), device=events.device, dtype=torch.float32),
        )
    events = events.float()
    x = events[:, 0] / max(width - 1, 1)
    y = events[:, 1] / max(height - 1, 1)
    t = events[:, 2]
    t = (t - t[0]) / (t[-1] - t[0]).abs().clamp_min(1e-6)
    p = torch.where(events[:, 3] >= 0, 1.0, -1.0)
    positions = torch.stack((x, y, t), dim=-1)
    node_features = torch.stack((x * 2 - 1, y * 2 - 1, t * 2 - 1, p), dim=-1)
    return node_features, positions


def build_causal_graph(
    positions: torch.Tensor,
    candidates: int = 32,
    spatial_radius: float = 0.12,
    temporal_radius: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect each event only to recent events, enabling streaming inference.

    Restricting candidates avoids the quadratic radius-graph materialization that is
    unsuitable for the target low-latency system.
    """
    n = positions.shape[0]
    device = positions.device
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    attr_parts: list[torch.Tensor] = []
    max_offset = min(max(0, int(candidates)), max(0, n - 1))
    for offset in range(1, max_offset + 1):
        src = torch.arange(0, n - offset, device=device)
        dst = src + offset
        delta = positions[dst] - positions[src]
        spatial = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        valid = (spatial <= spatial_radius) & (delta[:, 2] <= temporal_radius)
        # Appending empty tensors is cheap and avoids a device-to-host synchronization
        # from ``if valid.any()`` for every candidate offset on CUDA.
        kept_delta = delta[valid]
        src_parts.append(src[valid])
        dst_parts.append(dst[valid])
        attr_parts.append(
            torch.cat(
                (kept_delta, torch.linalg.vector_norm(kept_delta, dim=-1, keepdim=True)),
                dim=-1,
            )
        )

    # Self edges guarantee a defined degree for sparse non-empty crops.
    self_nodes = torch.arange(n, device=device)
    src_parts.append(self_nodes)
    dst_parts.append(self_nodes)
    attr_parts.append(torch.zeros((n, 4), device=device, dtype=positions.dtype))
    edge_index = torch.stack((torch.cat(src_parts), torch.cat(dst_parts)), dim=0)
    edge_attr = torch.cat(attr_parts, dim=0)
    return edge_index, edge_attr


def build_event_graph(
    events: torch.Tensor,
    sensor_size: tuple[int, int],
    candidates: int,
    spatial_radius: float,
    temporal_radius: float,
) -> EventGraph:
    node_features, positions = prepare_event_nodes(events, sensor_size)
    edge_index, edge_attr = build_causal_graph(
        positions,
        candidates=candidates,
        spatial_radius=spatial_radius,
        temporal_radius=temporal_radius,
    )
    return EventGraph(node_features, positions, edge_index, edge_attr)


class SplineMessageLayer(nn.Module):
    """Distance-conditioned graph aggregation inspired by ASGCN's B-spline kernel."""

    def __init__(self, channels: int, edge_dim: int = 4) -> None:
        super().__init__()
        self.message = nn.Linear(channels, channels, bias=False)
        self.self_projection = nn.Linear(channels, channels)
        self.edge_kernel = nn.Sequential(
            nn.Linear(edge_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm1d(channels)
        self.register_buffer("bn_bypassed", torch.tensor(False), persistent=True)
        self._bn_is_folded = False
        self.register_buffer("threshold", torch.ones(channels), persistent=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._bn_is_folded = bool(self.bn_bypassed.item())

    def preactivation(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        gates = self.edge_kernel(edge_attr)
        messages = self.message(x[src]) * gates
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, dst, messages)
        degree = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)
        output = self.self_projection(x) + aggregate
        return output if self._bn_is_folded else _safe_batch_norm(self.norm, output)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        if self._bn_is_folded:
            return
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm")
        variance = self.norm.running_var
        mean = self.norm.running_mean
        gamma = self.norm.weight
        beta = self.norm.bias
        scale = gamma / torch.sqrt(variance + self.norm.eps)
        self.message.weight.mul_(scale[:, None])
        self.self_projection.weight.mul_(scale[:, None])
        self.self_projection.bias.copy_((self.self_projection.bias - mean) * scale + beta)
        self.bn_bypassed.fill_(True)
        self._bn_is_folded = True

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.preactivation(x, edge_index, edge_attr)
        return torch.relu(z), z

    def rate_convert(
        self, z: torch.Tensor, simulation_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.numel() == 0:
            return z, torch.zeros((), device=z.device, dtype=z.dtype)
        threshold = self.threshold.to(dtype=z.dtype, device=z.device).clamp_min(1e-6)
        normalized = torch.clamp(torch.relu(z) / threshold, 0.0, 1.0)
        spike_count = torch.floor(normalized * simulation_steps + 1e-6)
        rate_output = spike_count * threshold / float(simulation_steps)
        # Keep diagnostics on-device so a timed CUDA forward has no hidden host sync.
        firing_rate = (spike_count / float(simulation_steps)).mean().detach()
        return rate_output, firing_rate


class ASGCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, graph_layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(4, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        self.register_buffer("input_bn_bypassed", torch.tensor(False), persistent=True)
        self._input_bn_is_folded = False
        self.layers = nn.ModuleList(
            [SplineMessageLayer(hidden_dim) for _ in range(int(graph_layers))]
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._input_bn_is_folded = bool(self.input_bn_bypassed.item())

    def forward_ann(
        self, graph: EventGraph, return_activations: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.input_linear(graph.node_features)
        if not self._input_bn_is_folded:
            h = _safe_batch_norm(self.input_norm, h)
        h = torch.relu(h)
        activations: list[torch.Tensor] = []
        for layer in self.layers:
            h, z = layer(h, graph.edge_index, graph.edge_attr)
            if return_activations:
                activations.append(torch.relu(z))
        return h, activations

    def forward_snn(
        self, graph: EventGraph, simulation_steps: int = 16
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.input_linear(graph.node_features)
        if not self._input_bn_is_folded:
            h = _safe_batch_norm(self.input_norm, h)
        h = torch.relu(h)
        firing_rates: list[torch.Tensor] = []
        for layer in self.layers:
            z = layer.preactivation(h, graph.edge_index, graph.edge_attr)
            h, firing_rate = layer.rate_convert(z, simulation_steps)
            firing_rates.append(firing_rate)
        return h, firing_rates

    @torch.no_grad()
    def update_thresholds(self, activations: list[torch.Tensor], momentum: float = -1.0) -> None:
        if len(activations) != len(self.layers):
            raise ValueError("Activation count does not match graph layer count")
        for layer, activation in zip(self.layers, activations, strict=True):
            if activation.numel() == 0:
                continue
            maxima = activation.amax(dim=0).clamp_min(1e-6)
            if momentum < 0:
                layer.threshold.copy_(torch.maximum(layer.threshold, maxima))
            else:
                layer.threshold.mul_(momentum).add_(maxima * (1.0 - momentum))

    @torch.no_grad()
    def reset_thresholds(self) -> None:
        for layer in self.layers:
            layer.threshold.fill_(1e-6)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.eval()
        if not self._input_bn_is_folded:
            variance = self.input_norm.running_var
            mean = self.input_norm.running_mean
            gamma = self.input_norm.weight
            beta = self.input_norm.bias
            scale = gamma / torch.sqrt(variance + self.input_norm.eps)
            self.input_linear.weight.mul_(scale[:, None])
            self.input_linear.bias.copy_((self.input_linear.bias - mean) * scale + beta)
            self.input_bn_bypassed.fill_(True)
            self._input_bn_is_folded = True
        for layer in self.layers:
            layer.fold_batch_norm()
~~~~

# src/asgcn_recon/losses.py

~~~~python
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import structural_similarity


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3):
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class ReconstructionLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        defaults = {"charbonnier": 1.0, "ssim": 0.2, "gradient": 0.1}
        self.weights = defaults | (weights or {})

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        terms: dict[str, torch.Tensor] = {
            "charbonnier": charbonnier_loss(prediction, target),
            "ssim": 1.0 - structural_similarity(prediction, target),
            "gradient": gradient_loss(prediction, target),
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        values: dict[str, Any] = {name: float(value.detach().cpu()) for name, value in terms.items()}
        values["total"] = float(total.detach().cpu())
        return total, values
~~~~

# src/asgcn_recon/metrics.py

~~~~python
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    min_side = min(prediction.shape[-2:])
    size = min(window_size, min_side)
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    padding = size // 2
    mu_x = F.avg_pool2d(prediction, size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, size, 1, padding) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean().clamp(-1.0, 1.0)


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=mse.device) / mse.clamp_min(1e-12))


@torch.no_grad()
def frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    result = {
        "psnr": float(peak_signal_to_noise_ratio(prediction, target).cpu()),
        "ssim": float(structural_similarity(prediction, target).cpu()),
        "rmse": float(torch.sqrt(F.mse_loss(prediction, target)).cpu()),
    }
    if lpips_model is not None:
        pred3 = prediction.repeat(1, 3, 1, 1) if prediction.shape[1] == 1 else prediction
        target3 = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        result["lpips"] = float(lpips_model(pred3 * 2 - 1, target3 * 2 - 1).mean().cpu())
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def update(self, scene: str, sample_id: str, metrics: dict[str, float]) -> None:
        self.frames.append({"scene": scene, "sample_id": sample_id, **metrics})

    def summary(self) -> dict[str, Any]:
        if not self.frames:
            return {"frames": 0, "micro": {}, "macro": {}, "per_scene": {}}
        names = [key for key in self.frames[0] if key not in {"scene", "sample_id"}]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for frame in self.frames:
            grouped[str(frame["scene"])].append(frame)
        per_scene = {
            scene: {
                "frames": len(items),
                **{name: sum(item[name] for item in items) / len(items) for name in names},
            }
            for scene, items in grouped.items()
        }
        micro = {name: sum(item[name] for item in self.frames) / len(self.frames) for name in names}
        macro = {
            name: sum(scene[name] for scene in per_scene.values()) / len(per_scene) for name in names
        }
        return {"frames": len(self.frames), "micro": micro, "macro": macro, "per_scene": per_scene}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
~~~~

# src/asgcn_recon/model.py

~~~~python
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .graph import ASGCNEncoder, EventGraph, build_event_graph


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.body(x))


class ConvGRUCell(nn.Module):
    """Causal analog state; only the graph front-end is ANN-to-SNN converted."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        merged = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(merged, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(merged, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        expected = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        if state is None or tuple(state.shape) != expected:
            state = torch.zeros(expected, device=x.device, dtype=x.dtype)
        reset, update = torch.sigmoid(self.gates(torch.cat((x, state), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * state), dim=1)))
        return (1.0 - update) * state + update * candidate


class RasterDecoder(nn.Module):
    def __init__(
        self, input_channels: int, base_channels: int, output_channels: int, recurrent: bool = True
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.enc1 = ResidualBlock(base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4), ResidualBlock(base_channels * 4)
        )
        self.recurrent = ConvGRUCell(base_channels * 4, base_channels * 4) if recurrent else None
        self.up2 = nn.Conv2d(base_channels * 6, base_channels * 2, 3, padding=1)
        self.dec2 = ResidualBlock(base_channels * 2)
        self.up1 = nn.Conv2d(base_channels * 3, base_channels, 3, padding=1)
        self.dec1 = ResidualBlock(base_channels)
        self.head = nn.Conv2d(base_channels, output_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int],
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        if self.recurrent is not None:
            state = self.recurrent(b, state)
            b = b + state
        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(self.up1(torch.cat((u1, e1), dim=1)))
        output = torch.sigmoid(self.head(u1))
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        return output, state


def rasterize_features(
    features: torch.Tensor,
    graph: EventGraph,
    sensor_size: tuple[int, int],
    downsample: int,
) -> torch.Tensor:
    height, width = sensor_size
    grid_h = max(1, (height + downsample - 1) // downsample)
    grid_w = max(1, (width + downsample - 1) // downsample)
    x = torch.clamp((graph.positions[:, 0] * width / downsample).long(), 0, grid_w - 1)
    y = torch.clamp((graph.positions[:, 1] * height / downsample).long(), 0, grid_h - 1)
    linear = y * grid_w + x
    raster = torch.zeros(
        (grid_h * grid_w, features.shape[-1]), device=features.device, dtype=features.dtype
    )
    raster.index_add_(0, linear, features)
    counts = torch.zeros((grid_h * grid_w, 1), device=features.device, dtype=features.dtype)
    counts.index_add_(
        0,
        linear,
        torch.ones(
            (linear.numel(), 1), device=features.device, dtype=features.dtype
        ),
    )
    raster = raster / counts.clamp_min(1.0)
    return raster.transpose(0, 1).reshape(1, features.shape[-1], grid_h, grid_w)


class ASGCNReconstructor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        graph_layers: int = 3,
        causal_candidates: int = 32,
        spatial_radius: float = 0.12,
        temporal_radius: float = 0.30,
        raster_downsample: int = 4,
        decoder_channels: int = 48,
        output_channels: int = 1,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ASGCNEncoder(hidden_dim, graph_layers)
        self.decoder = RasterDecoder(hidden_dim, decoder_channels, output_channels, recurrent)
        self.causal_candidates = int(causal_candidates)
        self.spatial_radius = float(spatial_radius)
        self.temporal_radius = float(temporal_radius)
        self.raster_downsample = int(raster_downsample)

    def _graph(self, sample: dict[str, Any]) -> EventGraph:
        return build_event_graph(
            sample["events"],
            sample["sensor_size"],
            self.causal_candidates,
            self.spatial_radius,
            self.temporal_radius,
        )

    def forward_sample(
        self,
        sample: dict[str, Any],
        inference_mode: str = "ann",
        simulation_steps: int = 16,
        return_activations: bool = False,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        graph = self._graph(sample)
        if inference_mode == "ann":
            features, activations = self.encoder.forward_ann(graph, return_activations)
            firing_rates: list[torch.Tensor] = []
        elif inference_mode == "snn":
            features, firing_rates = self.encoder.forward_snn(graph, simulation_steps)
            activations = []
        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}")
        raster = rasterize_features(
            features, graph, sample["sensor_size"], self.raster_downsample
        )
        prediction, next_state = self.decoder(raster, sample["sensor_size"], recurrent_state)
        diagnostics = {
            "nodes": int(graph.node_features.shape[0]),
            "edges": int(graph.edge_index.shape[1]),
            "firing_rates": firing_rates,
            "activations": activations,
            "recurrent_state": next_state,
        }
        return prediction, diagnostics

    def forward(
        self,
        batch: list[dict[str, Any]],
        inference_mode: str = "ann",
        simulation_steps: int = 16,
        recurrent_states: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        predictions, diagnostics = [], []
        if recurrent_states is None:
            recurrent_states = [None] * len(batch)
        for sample, state in zip(batch, recurrent_states, strict=True):
            prediction, detail = self.forward_sample(
                sample, inference_mode, simulation_steps, recurrent_state=state
            )
            predictions.append(prediction)
            diagnostics.append(detail)
        return predictions, diagnostics

    @torch.no_grad()
    def calibrate_sample(self, sample: dict[str, Any], momentum: float = -1.0) -> None:
        _, diagnostics = self.forward_sample(sample, return_activations=True)
        self.encoder.update_thresholds(diagnostics["activations"], momentum=momentum)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.encoder.fold_batch_norm()
~~~~

# src/asgcn_recon/utils.py

~~~~python
from __future__ import annotations

import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def experiment_base_dir(config_path: str | Path) -> Path:
    """Locate the checkout root that owns an experiment configuration.

    Checked-in configs use paths relative to the repository, not relative to the
    shell's current directory.  Falling back to the config directory also keeps
    standalone, externally supplied configs useful.
    """
    config_path = Path(config_path).expanduser().resolve()
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return config_path.parent


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(path))).expanduser()
    if not expanded.is_absolute():
        expanded = Path(base_dir) / expanded
    return expanded.resolve()


def resolve_experiment_paths(
    config: dict[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Return a copy with filesystem paths anchored to the checkout root."""
    resolved = copy.deepcopy(config)
    base_dir = experiment_base_dir(config_path)
    path_locations = (
        ("dataset", "root"),
        ("dataset", "val_root"),
        ("dataset", "split_manifest"),
        ("train", "resume"),
        ("output", "run_dir"),
        ("eval", "output_dir"),
    )
    for section, key in path_locations:
        value = resolved.get(section, {}).get(key)
        if value:
            resolved[section][key] = str(resolve_path(value, base_dir))
    return resolved


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(sample)
    result["events"] = sample["events"].to(device, non_blocking=True)
    result["target"] = sample["target"].to(device, non_blocking=True)
    return result


def save_image(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 1:
        image = Image.fromarray((array[0] * 255.0 + 0.5).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray(
            (array[:3].transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8), mode="RGB"
        )
    image.save(path)


def write_frame_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
~~~~

# tests/__init__.py

~~~~python
"""Test-only package; never installed with asgcn_recon."""
~~~~

# tests/fixtures.py

~~~~python
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def make_eventhdr(root: Path, frames: int = 4) -> Path:
    """Create the smallest useful EventHDR-shaped fixture under pytest tmp_path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test.h5"
    height, width = 32, 48
    events_per_frame = 96
    total = frames * events_per_frame
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as h5:
        h5.attrs["sensor_resolution"] = np.array([height, width], dtype=np.int32)
        h5.attrs["num_events"] = total
        h5.attrs["num_imgs"] = frames
        events = h5.create_group("events")
        events.create_dataset("xs", data=rng.integers(0, width, total, dtype=np.int16))
        events.create_dataset("ys", data=rng.integers(0, height, total, dtype=np.int16))
        events.create_dataset("ts", data=np.linspace(0.0, frames * 0.002, total))
        events.create_dataset("ps", data=rng.integers(0, 2, total, dtype=np.uint8))
        images = h5.create_group("images")
        images.attrs["num_images"] = frames
        yy, xx = np.mgrid[:height, :width]
        for index in range(frames):
            image = np.clip((xx + yy + index * 8) / (width + height + frames * 8), 0, 1)
            node = images.create_dataset(
                f"image{index:09d}", data=(image * 65535).astype(np.uint16)
            )
            node.attrs["event_idx"] = (index + 1) * events_per_frame
            node.attrs["timestamp"] = (index + 1) * 0.002
            node.attrs["size"] = [height, width]
            node.attrs["type"] = "hdr"
    return path


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def make_eventaid(root: Path, frames: int = 4) -> Path:
    """Create the smallest useful EventAid-R-shaped fixture under pytest tmp_path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "R-test.zip"
    height, width = 32, 48
    rng = np.random.default_rng(11)
    timestamps = [1_000_000 + index * 10_000 for index in range(frames)]
    yy, xx = np.mgrid[:height, :width]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("shape.txt", f"{width} {height}\n")
        zf.writestr("timestamps.txt", "\n".join(str(value) for value in timestamps) + "\n")
        for index in range(1, frames + 1):
            image = np.clip((xx + yy + index * 6) / (width + height + frames * 6), 0, 1)
            zf.writestr(f"gt/{index:06d}_img.png", _png_bytes(image * 255))
            t0 = timestamps[index - 1]
            rows = [
                (
                    f"{timestamp} {rng.integers(0, width)} {rng.integers(0, height)} "
                    f"{rng.integers(0, 2)}"
                )
                for timestamp in np.linspace(t0, t0 + 9_500, 80, dtype=np.int64)
            ]
            zf.writestr(f"event/{index:06d}.txt", "\n".join(rows) + "\n")
    return path
~~~~

# tests/test_e2e.py

~~~~python
from __future__ import annotations

import torch

from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.engine import benchmark, evaluate
from asgcn_recon.losses import ReconstructionLoss
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import atomic_torch_save
from tests.fixtures import make_eventaid, make_eventhdr


def test_train_eval_benchmark_contract(tmp_path):
    """Keep the full pipeline check test-only and confined to pytest's temp tree."""
    hdr_root = tmp_path / "hdr"
    aid_root = tmp_path / "aid"
    make_eventhdr(hdr_root)
    make_eventaid(aid_root)

    hdr = EventHDRDataset(hdr_root, max_events=32)
    aid = EventAidRZipDataset(aid_root, max_events=32)
    assert len(hdr) == 4
    assert len(aid) == 3

    model_config = {
        "hidden_dim": 8,
        "graph_layers": 2,
        "causal_candidates": 4,
        "spatial_radius": 1.0,
        "temporal_radius": 1.0,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": True,
    }
    model = ASGCNReconstructor(**model_config)
    sample = hdr[0]
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["nodes"] == 32

    checkpoint = tmp_path / "model.pt"
    atomic_torch_save(
        {"epoch": 0, "model": model.state_dict(), "model_config": model_config}, checkpoint
    )
    output_dir = tmp_path / "eval"
    config = {
        "seed": 7,
        "device": "cpu",
        "dataset": {
            "type": "eventaid_r_zip",
            "root": str(aid_root),
            "target_channels": 1,
            "max_events": 32,
            "crop_size": None,
            "target_offset": 1,
            "tone_map": "none",
        },
        "model": model_config,
        "eval": {
            "num_workers": 0,
            "max_samples": 2,
            "save_predictions": 1,
            "output_dir": str(output_dir),
        },
    }
    result = evaluate(config, checkpoint)
    timing = benchmark(config, checkpoint, warmup=1, steps=2)

    assert result["quality"]["frames"] == 2
    assert timing["frames"] == 2
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "frames.csv").is_file()
    assert len(list((output_dir / "predictions").glob("*_pred.png"))) == 1

    hdr.close()
    aid.close()
~~~~

# tests/test_pipeline.py

~~~~python
from __future__ import annotations

import json

import pytest
import torch

from asgcn_recon.cli import inspect_dataset
from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.data.factory import build_dataset
from asgcn_recon.engine import _data_loader, benchmark, train
from asgcn_recon.graph import build_causal_graph, prepare_event_nodes
from asgcn_recon.losses import ReconstructionLoss
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import (
    load_json,
    resolve_experiment_paths,
)
from tests.fixtures import make_eventaid, make_eventhdr


def test_eventhdr_loader(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=32)
    sample = dataset[0]
    assert sample["events"].shape == (32, 4)
    assert sample["target"].shape == (1, 32, 48)
    assert sample["events"][:, 2].min() >= 0
    assert sample["events"][:, 2].max() <= 1
    assert dataset[1]["metadata"]["dt_us"] == 2_000


def test_eventhdr_stride_aggregates_intervals(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=None, frame_stride=2)
    assert len(dataset) == 2
    assert dataset.samples[1]["end_idx"] - dataset.samples[1]["start_idx"] == 192
    assert dataset[1]["metadata"]["dt_us"] == 4_000


def test_eventaid_next_frame_alignment(tmp_path):
    make_eventaid(tmp_path / "eventaid")
    dataset = EventAidRZipDataset(tmp_path / "eventaid", max_events=32)
    assert len(dataset) == 3
    assert dataset.samples[0]["frame_id"] == 1
    assert dataset.samples[0]["target_name"].endswith("000002_img.png")
    assert dataset[0]["metadata"]["dt_us"] == 10_000


def test_causal_graph_has_no_future_sources():
    events = torch.tensor([[i, i, i, i % 2] for i in range(12)], dtype=torch.float32)
    _, positions = prepare_event_nodes(events, (16, 16))
    edge_index, _ = build_causal_graph(
        positions, candidates=4, spatial_radius=1.0, temporal_radius=1.0
    )
    assert torch.all(edge_index[0] <= edge_index[1])


def test_empty_event_interval_uses_zero_node_graph():
    sample = {
        "events": torch.empty((0, 4), dtype=torch.float32),
        "target": torch.zeros((1, 8, 8), dtype=torch.float32),
        "sensor_size": (8, 8),
        "sample_id": "empty/0",
        "metadata": {},
    }
    model = ASGCNReconstructor(
        hidden_dim=4,
        graph_layers=1,
        causal_candidates=2,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
    )
    prediction, diagnostics = model.forward_sample(sample)
    prediction.mean().backward()
    assert torch.isfinite(prediction).all()
    assert diagnostics["nodes"] == 0
    assert diagnostics["edges"] == 0


def test_model_forward_backward(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model = ASGCNReconstructor(
        hidden_dim=8,
        graph_layers=2,
        causal_candidates=4,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
    )
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert prediction.shape == (1, 1, 32, 48)
    assert diagnostics["edges"] >= diagnostics["nodes"]
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_bn_folding_and_snn_rate_path(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model = ASGCNReconstructor(
        hidden_dim=8,
        graph_layers=2,
        causal_candidates=4,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
        recurrent=False,
    ).eval()
    with torch.no_grad():
        ann_before, _ = model.forward_sample(sample)
        model.fold_batch_norm()
        ann_after, _ = model.forward_sample(sample)
        restored = ASGCNReconstructor(
            hidden_dim=8,
            graph_layers=2,
            causal_candidates=4,
            spatial_radius=1.0,
            temporal_radius=1.0,
            raster_downsample=4,
            decoder_channels=4,
            recurrent=False,
        ).eval()
        restored.load_state_dict(model.state_dict())
        ann_restored, _ = restored.forward_sample(sample)
        model.encoder.reset_thresholds()
        model.calibrate_sample(sample)
        snn_output, diagnostics = model.forward_sample(
            sample, inference_mode="snn", simulation_steps=8
        )
    assert torch.allclose(ann_before, ann_after, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ann_after, ann_restored, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(snn_output).all()
    assert len(diagnostics["firing_rates"]) == 2


def test_cpu_autocast_keeps_raster_dtypes_compatible(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=16)[0]
    model = ASGCNReconstructor(
        hidden_dim=4,
        graph_layers=1,
        causal_candidates=2,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction, _ = model.forward_sample(sample)
    assert prediction.dtype == torch.bfloat16
    assert torch.isfinite(prediction).all()


def test_config_paths_are_independent_of_shell_cwd(tmp_path):
    repository = tmp_path / "checkout"
    config_dir = repository / "configs"
    config_dir.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    config_path = config_dir / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "root": "data/train",
                    "split_manifest": "manifests/split.json",
                },
                "output": {"run_dir": "runs/example"},
            }
        ),
        encoding="utf-8",
    )
    config = resolve_experiment_paths(load_json(config_path), config_path)
    assert config["dataset"]["root"] == str((repository / "data/train").resolve())
    assert config["dataset"]["split_manifest"] == str(
        (repository / "manifests/split.json").resolve()
    )
    assert config["output"]["run_dir"] == str((repository / "runs/example").resolve())


def test_eventhdr_split_names_all_missing_files(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps({"train_files": ["1.h5", "2.h5"], "val_files": ["3.h5"]}),
        encoding="utf-8",
    )
    config = {
        "type": "eventhdr",
        "root": str(data_root),
        "split_manifest": str(manifest_path),
    }
    with pytest.raises(FileNotFoundError, match=r"1\.h5, 2\.h5"):
        build_dataset(config, split="train")


def test_inspect_training_config_validates_both_manifest_splits(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": ["test.h5"],
                "val_files": ["missing_validation.h5"],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root),
            "split_manifest": str(manifest_path),
        }
    }
    with pytest.raises(FileNotFoundError, match=r"missing_validation\.h5"):
        inspect_dataset(config, samples=1)


def _tiny_training_config(tmp_path, data_root):
    return {
        "seed": 17,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root),
            "target_channels": 1,
            "max_events": 16,
            "crop_size": [16, 16],
            "tone_map": "log",
        },
        "model": {
            "hidden_dim": 4,
            "graph_layers": 1,
            "causal_candidates": 2,
            "spatial_radius": 1.0,
            "temporal_radius": 1.0,
            "raster_downsample": 4,
            "decoder_channels": 4,
            "output_channels": 1,
            "recurrent": True,
        },
        "train": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "amp": False,
            "max_train_samples": 1,
            "max_val_samples": 1,
            "log_every": 100,
        },
        "output": {"run_dir": str(tmp_path / "run")},
    }


def test_training_checkpoint_can_resume_optimizer_and_epoch(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)
    first = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1
    assert "optimizer" in first and "scaler" in first and "rng_state" in first

    config["train"]["epochs"] = 2
    train(config, resume_from=tmp_path / "run/last.pt")
    resumed = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]


def test_benchmark_rejects_empty_measurement(tmp_path):
    with pytest.raises(ValueError, match="steps must be at least 1"):
        benchmark({}, tmp_path / "unused.pt", steps=0)


def test_hdf5_and_zip_loaders_are_multiprocess_safe(tmp_path):
    hdr = tmp_path / "hdr"
    eventaid = tmp_path / "eventaid"
    make_eventhdr(hdr)
    make_eventaid(eventaid)
    datasets = [
        EventHDRDataset(hdr, max_events=8),
        EventAidRZipDataset(eventaid, max_events=8),
    ]
    for dataset in datasets:
        loader = _data_loader(
            dataset,
            batch_size=1,
            num_workers=2,
            device=torch.device("cpu"),
            persistent_workers=False,
        )
        sample = next(iter(loader))[0]
        assert sample["events"].shape == (8, 4)
~~~~
