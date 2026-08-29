# .dockerignore

~~~~~~~~text
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
~~~~~~~~

# .editorconfig

~~~~~~~~text
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
~~~~~~~~

# .env.example

~~~~~~~~text
# Host paths mounted by Docker Compose.
DATA_DIR=./data
RUNS_DIR=./runs

# Optional PyTorch wheel index selected for the server driver.
# Obtain the current value from https://pytorch.org/get-started/locally/
TORCH_INDEX_URL=

# Server-side installer overrides. The checked-in lock profile uses Python 3.12.
PYTHON_BIN=python3.12
VENV_DIR=.venv
PROJECT_EXTRAS=dev
REQUIRE_CUDA=0

# Exact profile validated by local tests and the locked Linux CI job.
TORCH_VERSION=2.13.0
CONSTRAINTS_FILE=constraints/py312.txt
EXPECTED_PYTHON_MINOR=3.12
~~~~~~~~

# .gitattributes

~~~~~~~~text
* text=auto
*.py text eol=lf
*.sh text eol=lf
*.sbatch text eol=lf
*.pbs text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.json text eol=lf
*.txt text eol=lf
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
~~~~~~~~

# .github/ISSUE_TEMPLATE/bug_report.yml

~~~~~~~~yaml
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
~~~~~~~~

# .github/pull_request_template.md

~~~~~~~~markdown
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
~~~~~~~~

# .github/workflows/ci.yml

~~~~~~~~yaml
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
          cache-dependency-path: |
            pyproject.toml
            constraints/py312.txt

      - name: Install editable development package
        run: |
          python -m pip install -c constraints/py312.txt --upgrade pip setuptools wheel
          python -m pip install -c constraints/py312.txt -e ".[dev]"
          python scripts/check_env.py --lock constraints/py312.txt

      - name: Run Ruff
        run: python -m ruff check .

      - name: Validate Linux shell entrypoints
        run: bash -n scripts/*.sh server/*.sbatch server/*.pbs

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
~~~~~~~~

# .gitignore

~~~~~~~~text
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
asgcn-*.o*
asgcn-*.e*
.data_hash_cache.json
*.pt
*.pth
*.onnx

# Local project source documents may contain non-public application details.
*.hwp
~~~~~~~~

# compose.yaml

~~~~~~~~yaml
services:
  experiment:
    build:
      context: .
      args:
        TORCH_INDEX_URL: ${TORCH_INDEX_URL:-}
        TORCH_VERSION: ${TORCH_VERSION:-2.13.0}
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
~~~~~~~~

# configs/aid_ann.json

~~~~~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventaid_r_zip",
    "root": "data/EventAid-R",
    "file_manifest": "manifests/eventaid_r.json",
    "expected_file_count": 14,
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "target_offset": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "architecture_version": 2,
    "hidden_dim": 64,
    "graph_layers": 6,
    "graph_operator": "spline",
    "spline_backend": "torch",
    "spline_pseudo": "distance_over_radius",
    "spline_is_open": true,
    "event_sampling_factor": 1,
    "graph_radius": 0.08,
    "graph_position_dims": 3,
    "graph_chunk_size": 512,
    "max_graph_edges": 2000000,
    "spline_kernel_size": 5,
    "spline_degree": 1,
    "spline_root_weight": true,
    "snn_dynamics": "literal_eq15",
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
    "recurrent_context_frames": 32,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventaid_r_external_ann"
  }
}
~~~~~~~~

# configs/aid_snn.json

~~~~~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventaid_r_zip",
    "root": "data/EventAid-R",
    "file_manifest": "manifests/eventaid_r.json",
    "expected_file_count": 14,
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "target_offset": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "architecture_version": 2,
    "hidden_dim": 64,
    "graph_layers": 6,
    "graph_operator": "spline",
    "spline_backend": "torch",
    "spline_pseudo": "distance_over_radius",
    "spline_is_open": true,
    "event_sampling_factor": 1,
    "graph_radius": 0.08,
    "graph_position_dims": 3,
    "graph_chunk_size": 512,
    "max_graph_edges": 2000000,
    "spline_kernel_size": 5,
    "spline_degree": 1,
    "spline_root_weight": true,
    "snn_dynamics": "literal_eq15",
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
    "recurrent_context_frames": 32,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventaid_r_external_snn"
  }
}
~~~~~~~~

# configs/hdr_ann.json

~~~~~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/eval",
    "expected_file_count": 19,
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "frame_stride": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "architecture_version": 2,
    "hidden_dim": 64,
    "graph_layers": 6,
    "graph_operator": "spline",
    "spline_backend": "torch",
    "spline_pseudo": "distance_over_radius",
    "spline_is_open": true,
    "event_sampling_factor": 1,
    "graph_radius": 0.08,
    "graph_position_dims": 3,
    "graph_chunk_size": 512,
    "max_graph_edges": 2000000,
    "spline_kernel_size": 5,
    "spline_degree": 1,
    "spline_root_weight": true,
    "snn_dynamics": "literal_eq15",
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
    "recurrent_context_frames": 32,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventhdr_official_eval_ann"
  }
}
~~~~~~~~

# configs/hdr_snn.json

~~~~~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/eval",
    "expected_file_count": 19,
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "frame_stride": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "architecture_version": 2,
    "hidden_dim": 64,
    "graph_layers": 6,
    "graph_operator": "spline",
    "spline_backend": "torch",
    "spline_pseudo": "distance_over_radius",
    "spline_is_open": true,
    "event_sampling_factor": 1,
    "graph_radius": 0.08,
    "graph_position_dims": 3,
    "graph_chunk_size": 512,
    "max_graph_edges": 2000000,
    "spline_kernel_size": 5,
    "spline_degree": 1,
    "spline_root_weight": true,
    "snn_dynamics": "literal_eq15",
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
    "recurrent_context_frames": 32,
    "max_samples": null,
    "save_predictions": 20,
    "output_dir": "runs/eventhdr_official_eval_snn"
  }
}
~~~~~~~~

# configs/hdr_train.json

~~~~~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/train",
    "val_root": "data/EventHDR/eval",
    "split_manifest": "manifests/eventhdr_split.json",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": null,
    "frame_stride": 1,
    "tone_map": "log",
    "tone_map_mu": 5000.0
  },
  "model": {
    "architecture_version": 2,
    "hidden_dim": 64,
    "graph_layers": 6,
    "graph_operator": "spline",
    "spline_backend": "torch",
    "spline_pseudo": "distance_over_radius",
    "spline_is_open": true,
    "event_sampling_factor": 1,
    "graph_radius": 0.08,
    "graph_position_dims": 3,
    "graph_chunk_size": 512,
    "max_graph_edges": 2000000,
    "spline_kernel_size": 5,
    "spline_degree": 1,
    "spline_root_weight": true,
    "snn_dynamics": "literal_eq15",
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
    "optimizer": "adam_gc",
    "learning_rate": 0.001,
    "weight_decay": 0.005,
    "lr_milestones": [20, 30],
    "lr_gamma": 0.1,
    "grad_clip": 1.0,
    "amp": true,
    "log_every": 20,
    "validate_every": null,
    "resume": null,
    "max_train_samples": null,
    "max_val_samples": null,
    "validation_context_frames": 64,
    "rehash_data": false,
    "loss_weights": {
      "charbonnier": 1.0,
      "ssim": 0.2,
      "gradient": 0.1,
      "temporal": 0.2
    }
  },
  "output": {
    "run_dir": "runs/eventhdr_asgcn"
  }
}
~~~~~~~~

# constraints/py312.txt

~~~~~~~~text
# Reproducible core/dev profile validated with Python 3.12.13.
# Select the CUDA build of torch from the official PyTorch wheel index via
# TORCH_INDEX_URL; the public version remains fixed here.
filelock==3.32.4
fsspec==2026.7.0
h5py==3.16.0
iniconfig==2.3.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.2
packaging==26.3
pillow==12.3.0
pluggy==1.6.0
Pygments==2.21.0
pytest==9.1.1
ruff==0.16.5
sympy==1.14.0
torch==2.13.0
tqdm==4.70.0
typing_extensions==4.16.0
~~~~~~~~

# CONTRIBUTING.md

~~~~~~~~markdown
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
- EventHDR 공식 eval과 EventAid-R은 gradient update나 hyperparameter tuning에 사용하지 않습니다.
  기본 프로토콜은 official eval을 마지막 epoch에서 한 번만 계산하고 EventAid-R은 그 뒤 외부평가에만
  사용합니다.
- 포매터가 아닌 Ruff 검사 결과를 기준으로 기존 코드 스타일을 유지합니다.

## 데이터와 생성물

공식 데이터셋, 압축 파일, checkpoint, 학습 로그와 대규모 출력은 Git에 커밋하지 않습니다.
재현에 필요한 메타데이터와 `tests/` 내부의 작은 fixture만 저장하세요. 버그 보고에 실제 데이터
일부가 필요하다면 먼저 배포 라이선스와 개인 정보 포함 여부를 확인하고, 가능하면 이를 대신하는
최소 fixture를 제공하세요.

## 데이터 로더 변경 시 주의사항

- EventAid-R 이벤트 구간의 target은 동일 번호가 아니라 다음 번호의 GT입니다.
- 프레임 속도를 상수로 가정하지 말고 제공된 timestamp 차이를 사용합니다.
- EventHDR 공개 H5 번호를 물리 장면 ID로 주장하지 않습니다. 공식 eval은 마지막 epoch의 내부 결과,
  EventAid-R은 외부 일반화 결과로 구분합니다.
- 데이터 형식 검증 실패는 가능한 한 파일명과 기대한 구조를 포함한 명확한 오류로 보고합니다.
~~~~~~~~

# Dockerfile

~~~~~~~~dockerfile
FROM python:3.12-slim

ARG TORCH_INDEX_URL=""
ARG TORCH_VERSION="2.13.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY constraints ./constraints
COPY src ./src

RUN python -m pip install -c constraints/py312.txt --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_INDEX_URL" ]; then \
         python -m pip install -c constraints/py312.txt \
           "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"; \
       else \
         python -m pip install -c constraints/py312.txt "torch==$TORCH_VERSION"; \
       fi \
    && python -m pip install -c constraints/py312.txt -e . \
    && python -m pip check

COPY configs ./configs
COPY manifests ./manifests
COPY scripts ./scripts
COPY server ./server

RUN mkdir -p /workspace/data /workspace/runs

ENTRYPOINT ["asgcn-recon"]
CMD ["--help"]
~~~~~~~~

# docs/ASGCN.md

~~~~~~~~markdown
# ASGCN paper-core 구현 범위

이 저장소는 AAAI 2025 ASGCN 논문의 공개 수식에서 확인할 수 있는 event graph와 ANN→SNN
변환 핵심을 구현한 뒤, event-to-frame 복원용 decoder를 연결한 연구 코드다. 원 논문은 event
classification을 다루지만 이 프로젝트의 출력은 luminance frame이다. 따라서 이 코드는 저자 공식
구현, 원 논문의 classification pipeline, 또는 공식 성능표의 완전 재현본이 아니다.

근거로 삼은 공개 자료는 [AAAI 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32154)와
[공식 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)다. 확인 가능한 논문에는
저자 코드와 checkpoint 링크가 없으며, 복원 과제에 필요한 graph·spline·decoder 설정도 모두
주어지지 않는다. 문서와 결과에서는 반드시 `ASGCN paper-core 기반 복원 적응` 또는 `공개 수식
기반 구현`으로 표현하고, `저자 공식 코드`, `공식 ASGCN 완전 재현`, `논문 성능 재현`이라고
표현하지 않는다.

## 1. 논문 수식에서 가져온 core

현재 `architecture_version=2`의 graph encoder가 다음 항목을 구현한다.

1. event sequence를 고정 정수 sampling factor `R`로 균일하게 선택한다.
2. event를 node로 만들고, 좌표 거리 조건 `d(i,j) < D`를 만족하는 node 쌍을 잇는다.
3. scalar edge distance를 pseudo-coordinate로 쓰는 weighted B-spline graph convolution과 실제
   incoming degree 기반 mean aggregation을 적용한다.
4. ANN 경로에서는 graph affine update, BatchNorm, ReLU를 순서대로 실행한다.
5. ANN 학습 뒤 식 (13)–(14)의 BatchNorm folding과 식 (6)의 layer-wise parameter
   normalization을 적용한다.
6. SNN 경로에서는 별도 stochastic/rate input encoder 없이 IF membrane을 명시적 timestep으로
   전개한다. 초기 membrane은 threshold의 절반, spike amplitude는 threshold이며 발화 뒤 soft
   reset을 사용한다.

원 논문의 graph clustering, graph pooling, edge remapping과 classification head는 구현하지 않았다.
이 프로젝트는 graph encoder의 node feature를 raster로 바꿔 복원 decoder에 전달한다.

## 2. 논문에 없는 명시적 가정

논문만으로 결정할 수 없는 값은 config에 노출했고 checkpoint의 `model_config`에 보존한다. 기본
본실험 설정은 다음과 같다.

| 항목 | 저장소의 선택 | 해석 |
|---|---:|---|
| node feature | `[x,y,t,p]` | `x,y,t`는 `[0,1]`, polarity는 `-1/+1` |
| graph distance | 정규화된 `x,y,t` 3차원 | polarity는 기본 거리에서 제외 |
| radius | `D=0.08` | 원 논문의 복원 과제용 공식값이 아님 |
| edge pseudo | `u=d/D` | 원시 distance를 SplineConv의 `[0,1]` 정의역으로 재매개화 |
| graph width/depth | 64 features, 6 layers | 복원 과제 가정 |
| spline | scalar, open, degree 1, `K=5` | degree/kernel/open 여부는 논문 미공개 |
| update terms | mean message + root transform + bias | root/bias는 PyG 계열 동작을 참고한 선택 |
| paper sampling `R` | `event_sampling_factor=1` | dataset의 8,192-event cap과 별개 |
| raster | 4배 downsample grid에서 cell mean | 논문에 없는 event-to-frame bridge |
| decoder | base 48 residual U-Net + analog ConvGRU | 논문에 없는 복원 확장 |
| SNN dynamics | `literal_eq15` | 공개 식을 우선한 선택; 아래 모호성 참조 |
| edge guard | directed edge 2,000,000개 | 초과 시 graph를 자르지 않고 실패 |

dataset loader의 `max_events=8192`는 논문의 `R`이 아니다. crop 뒤 event가 cap을 넘으면
`np.linspace(0, N-1, 8192)`로 정확히 8,192개를 시간축 전체에서 결정적으로 선택하며 양 끝 event를
포함한다. 그 뒤 model의 고정 factor `R`이 `events[::R]`로 적용된다. 결과에는 raw/cropped/retained
event 수, dataset sampling ratio, model factor와 두 비율의 곱을 따로 기록한다.

## 3. exact cell radius graph

`build_radius_graph`는 dense `N×N` distance matrix를 만들지 않고 uniform-cell candidate search를
사용하지만, 만들어지는 graph의 의미는 brute-force strict radius graph와 같다.

- 선택한 `d`차원 좌표에서 cell 폭을 정확히 `D`로 둔다.
- 각 source node의 cell과 축별 offset `{-1,0,1}`의 조합인 `3^d` 인접 cell만 검색한다. cell 폭이
  `D`이므로 그 밖의 cell에는 `d < D`인 이웃이 존재할 수 없다.
- sorted cell hash의 범위를 `searchsorted`로 찾아 후보를 만들고, 모든 후보에 대해 Euclidean
  distance를 다시 계산한다.
- `source != destination`와 `distance < D`를 모두 만족할 때만 edge를 남긴다. 경계 `distance=D`는
  포함하지 않는다.
- 모든 node를 source로 처리하므로 각 유효한 무방향 쌍의 두 ordered direction을 모두
  materialize한다. self-loop와 중복 edge는 없다.
- 최종 edge는 `(source,destination)` 순서로 정렬하며 edge attribute는 `distance/D`다.

`graph_chunk_size=512`와 내부 candidate chunk 축소는 계산을 나누는 메모리 최적화일 뿐 neighbor를
근사하거나 누락하지 않는다. 다만 한 cell에 event가 몰리는 최악의 경우 후보 수와 실제 edge 수는
여전히 `O(N²)`가 될 수 있다. `max_graph_edges`는 이 경우 조용히 edge를 버리지 않고 오류를 내는
fail-fast 장치다. radius가 만든 graph를 강제로 연결하지 않으므로 isolated node는 root transform과
bias만 받으며 isolate ratio와 maximum degree가 결과에 기록된다.

## 4. degree-1 open B-spline과 계산 최적화

scalar pseudo-coordinate `u`에 대해 `scaled=u(K-1)`를 계산한다. degree 1이므로 edge마다 활성
basis는 인접한 두 control point뿐이고, 가중치는 `1-frac(scaled)`와 `frac(scaled)`다. 두 가중치의
합은 endpoint를 포함해 1이다. `u=1`에서는 마지막 control point의 가중치가 정확히 1이 되도록
endpoint 동작과 pseudo-coordinate gradient를 테스트로 고정했다.

각 `PaperSplineConv` layer의 계산은 다음과 같다.

```text
node projection for every control point: [N,Cin] x [K,Cin,Cout] -> [N,K,Cout]
edge gather: source node의 활성 control point 두 개만 선택
weighted message: 두 basis weight로 합산
aggregation: destination의 실제 incoming degree로 평균
update: mean message + optional root transform + bias
ANN activation: BatchNorm -> ReLU
```

edge마다 작은 matrix multiplication을 반복하지 않고 node를 `K`개 control point에 한 번씩 projection한
뒤 두 활성 항만 gather한다. 고정 graph의 basis index와 weight도 sample당 한 번 계산해 6개 graph
layer와 모든 IF timestep에서 재사용한다. 이는 연산 중복을 줄이는 exact optimization이며 graph나
spline 값을 바꾸지 않는다. 구현은 순수 PyTorch라 `torch-spline-conv` binary extension에
의존하지 않는다.

weight 초기화 bound는 `1/sqrt(K*Cin)`, root bound는 `1/sqrt(Cin)`으로 고정했다. open degree 1,
scalar pseudo-coordinate, mean aggregation, root weight와 bias 범위에서만 구현·테스트했으며 이를
ASGCN 저자의 미공개 hyperparameter와 동일하다고 주장하지 않는다.

## 5. ANN 학습과 ANN→SNN 변환

ANN graph layer는 B-spline affine update, BatchNorm, ReLU로 학습한다. 기본 optimizer는 Adam에
gradient centralization을 추가한 `adam_gc`이며 learning rate `1e-3`, weight decay `5e-3`, epoch
20/30의 MultiStepLR과 gamma `0.1`을 사용한다. milestone과 gamma, tensor별 centralization 축은
저자 코드로 검증된 공식값이 아니라 config 선택이다.

변환 순서는 다음과 같다.

1. 학습된 ANN checkpoint를 불러온다.
2. 각 graph layer의 BatchNorm scale을 spline kernel, root와 bias에 fold한다.
3. EventHDR train 전체에서 각 graph layer의 feature별 ReLU maximum `lambda_l`를 측정한다.
4. 식 (6)에 따라 kernel과 root에 `lambda_(l-1)/lambda_l`, bias에 `1/lambda_l`를 적용한다.
5. 첫 layer의 입력 scale `lambda_0`은 `[1,1,1,1]`로 둔다.
6. calibration에서 항상 0인 channel은 epsilon으로 폭증시키지 않고 unit scale을 사용한다.
7. 변환 뒤 threshold를 정확히 1로 두고, 마지막 spike rate에 `lambda_L`를 곱해 analog decoder의
   학습 단위로 복원한다.

각 layer의 persistent BN-fold/normalization flag, valid calibration count, dead-channel 수, threshold,
model tensor SHA-256과 checkpoint metadata를 load 시 교차검증한다. ANN checkpoint와 변환된 SNN
checkpoint의 inference mode를 서로 바꿔 사용하는 것도 거부한다.

## 6. IF dynamics와 식 (15)의 모호성

논문 식 (15)는 다음 self-feedback 항을 포함한다.

```text
v_tilde(t) = v(t-1) + c(t) + h(t-1)
```

`literal_eq15`는 이 `+h(t-1)`을 문자 그대로 실행한다. 각 timestep에서 threshold 이상이면 threshold
크기의 spike를 내고 membrane에서 그 값을 빼는 soft reset을 한다. 그러나 이전 spike의 재주입은
작은 양의 정전류도 첫 발화 뒤 장기 firing rate 1에 가깝게 만들 수 있어 표준 ANN rate-conversion
유도와 충돌한다. 저자 코드나 정정 자료가 없으므로 임의로 오타 처리하지 않았다.

- `literal_eq15`: 기본 경로. 공개 식 (15)–(17)을 문자 그대로 실행한다.
- `standard_if`: `+h(t-1)`을 제거한 대조군이다. 공식 ASGCN 설정이 아니다.

마지막 `lambda_L` 곱은 decoder 입력의 단위 변환이지, `literal_eq15`가 유한 timestep에서 ANN과
동등하다는 증명이 아니다. 두 dynamics는 같은 calibrated checkpoint로 비교할 수 있지만 결과는
반드시 별도 dynamics와 timestep으로 표기한다.

## 7. hybrid event-to-frame decoder

전체 forward는 다음과 같다.

```text
event interval [N,x,y,t,p]
  -> exact-size dataset cap
  -> fixed paper sampling factor R
  -> strict undirected radius graph
  -> B-spline graph encoder (ANN 또는 explicit IF-SNN)
  -> downsampled feature raster: pixel-cell별 node feature mean
  -> residual U-Net encoder
  -> bottleneck analog ConvGRU
  -> bilinear upsampling + skip connections
  -> sigmoid luminance frame, 원 sensor size로 interpolation
```

U-Net은 두 번 downsample하고 bottleneck에 residual block 두 개를 둔다. ConvGRU state는 같은 H5/ZIP
sequence group, 연속 sequence index, 같은 sensor shape일 때만 전달되고 불연속에서는 초기화된다.
학습 시 state와 이전 prediction은 frame마다 detach하므로 전체 sequence backpropagation이 아니라
frame 단위 truncated recurrence다. SNN 변환 대상은 graph encoder뿐이며 rasterization, U-Net,
ConvGRU와 output head는 모두 analog 연산이다.

EventHDR의 zero-event target interval도 버리지 않는다. 빈 event tensor는 zero-node/zero-edge graph와
zero raster를 만들고 recurrent decoder가 해당 target frame을 학습·평가하도록 한다. `frame_stride>1`을
사용하면 건너뛴 interval의 event를 다음 선택 frame까지 합치지만 기본 본실험은 `frame_stride=1`이다.

## 8. 구현하지 않은 범위와 주장 한계

- 새 event에 영향받는 K-hop subgraph만 갱신하는 asynchronous incremental 실행
- sliding window node 만료와 graph/SNN state의 하드웨어 친화적 관리
- 논문 식 (18)–(19)의 clustering, pooling, edge remapping
- 원 논문의 classification MLP, 원 데이터셋 전처리와 성능표
- 실제 DVS sensor ingest, event compression·전송 protocol
- FPGA/ASIC RTL, synthesis, 실제 latency·전력·에너지 측정
- analog U-Net/ConvGRU까지 포함한 완전한 spiking network

현재 구현으로 보고할 수 있는 것은 PyTorch GPU/CPU에서의 paper-core 기반 복원 품질, graph 통계,
compute latency와 firing-rate 통계다. 이를 저자 공식 재현, neuromorphic hardware latency, 반도체
전력·에너지 또는 통합 칩 구현 완료로 확대해 해석하면 안 된다.
~~~~~~~~

# docs/EXPERIMENT.md

~~~~~~~~markdown
# 본실험 프로토콜

이 문서는 `configs/hdr_train.json`, `configs/hdr_{ann,snn}.json`,
`configs/aid_{ann,snn}.json`과 현재 실행 코드를 기준으로 한다. 전체 기본 실행은
`bash scripts/full.sh`이며, 일부 파일이나 일부 frame만 사용한 결과는 본실험 결과로 합치지 않는다.
ASGCN 수식과 이 저장소의 과제용 확장 범위는 [ASGCN.md](ASGCN.md)를 함께 참고한다.

## 1. 연구 질문과 데이터 역할

1. EventHDR 실제 event로 luminance frame을 얼마나 잘 복원하는가?
2. 동일 ANN checkpoint를 보정한 뒤 `literal_eq15`와 `standard_if`가
   `T=4,8,16,32`에서 보이는 품질·지연 차이는 무엇인가?
3. 학습과 보정에 쓰지 않은 EventAid-R 14개 scene에서 성능이 얼마나 유지되는가?

| 단계 | 고정 데이터 | 코드상 역할 |
|---|---|---|
| 학습 | EventHDR 공식 train root의 `1.h5`–`51.h5` | 40 epoch weight 최적화 |
| 내부 평가 | EventHDR 공식 eval root의 `1.h5`–`19.h5` | 마지막 epoch에서만 ANN 평가하고 `best.pt` 생성 |
| ANN→SNN 보정 | EventHDR train 51개 H5의 모든 선택 frame | BN folding, activation maximum, 식 (6) 정규화 |
| 내부 비교 | 동일 EventHDR eval 19개 H5 | ANN과 두 IF dynamics × 네 T 비교 |
| 외부 비교 | EventAid-R manifest의 14개 ZIP | 고정 checkpoint의 외부 일반화 평가 |

`manifests/eventhdr_split.json`은 `status=final`,
`split_schema=official_separate_roots_v1`이다. train과 eval은 별도 root이고 숫자 basename이 겹치는
공식 배포 구조다. H5 하나를 recurrent state와 metric의 sequence group으로 사용할 뿐, 공개 자료에
물리 scene 대응표가 없으므로 두 root가 물리 scene-disjoint이거나 통계적으로 독립이라고 주장하지
않는다. JSON의 `macro`/`per_scene`도 EventHDR에서는 물리 scene이 아니라 H5 sequence-file 단위다.

`train.validate_every=null`이므로 1–39 epoch에는 내부 평가를 하지 않고 40번째 마지막 epoch에서만
EventHDR eval 전체를 평가한다. 따라서 현재 `best.pt`는 여러 epoch를 비교해 고른 checkpoint가 아니라
마지막 epoch checkpoint다. 파일명 `best.pt`와 `best_metric=macro_ssim`은 공통 checkpoint 계약을 위한
것이다. 이후 같은 eval root에서 생성하는 ANN/SNN 표는 내부 비교이며 독립 잠금시험으로 표현하지
않는다. EventAid-R 결과를 본 뒤 설정·보정·threshold를 바꾸면 그 결과 역시 외부시험으로 해석할 수
없다.

## 2. 고정 전처리

두 dataset의 target은 다음 순서로 `[0,1]` luminance domain에 놓는다.

```text
integer image / dtype maximum
  -> RGB이면 BT.709 luminance
  -> y = log1p(5000*x) / log1p(5000)
```

EventAid-R은 event block `i`와 GT `i+1`을 짝짓는 `target_offset=1`을 사용한다. 이는 이 과제를
위한 정렬 가정이며 다른 offset 결과와 섞지 않는다. 두 dataset의 sensor response와 exposure가 같다는
보장은 없으므로 절대 PSNR/SSIM을 동일 분포의 수치처럼 직접 비교하지 않는다.

event 전처리는 다음 순서다.

1. 설정된 sensor crop을 적용한다. 본실험 config는 `crop_size=null`이라 전체 sensor를 쓴다.
2. crop 뒤 event 수가 `max_events=8192`를 넘으면, 양 끝 timestamp를 포함하는 결정적 `linspace`
   index로 **정확히 8,192개**를 시간 전역에서 남긴다. 8,192개 이하면 그대로 둔다.
3. model의 논문식 균일 sampling factor `R=event_sampling_factor=1`을 적용한다.

두 번째 단계는 graph memory를 제한하기 위한 과제용 cap이며 ASGCN 논문의 공식 `R`이 아니다.
`raw_event_count`, `cropped_event_count`, `retained_event_count`, `dataset_sampling_ratio`를 분리해
기록한다. 동일 source sequence의 crop은 seed와 상대 sequence identity로 결정되므로 worker 수와
resume 여부에 따라 바뀌지 않는다.

EventHDR에서 연속 target 사이 event가 0개인 interval도 sample로 보존한다. EventAid-R의 빈 event
text도 빈 `[0,4]` tensor로 유지한다. 빈 sample은 0 node·0 edge graph와 zero raster를 거쳐 analog
decoder/ConvGRU로 처리되며 임의 event를 합성하지 않는다. 보정에서는 비어 있는 activation이 layer의
유효 calibration observation으로 집계되지 않고, 어느 graph layer든 non-empty observation이 0개면
변환을 거부한다.

## 3. graph와 모델 고정값

기본 graph node feature는 정규화한 `(x,y,t,polarity)`이고, 거리 계산은 `[0,1]`로 정규화한
`(x,y,t)` 3차원에서 한다. `graph_radius=0.08`보다 Euclidean 거리가 **엄격히 작은** 서로 다른 node를
연결하고 양 방향 directed edge를 저장한다. edge pseudo-coordinate는 `distance/radius` 한 값이다.

구현은 폭이 radius인 uniform cell과 인접 `3^3` cell로 후보를 찾은 뒤 exact Euclidean 조건을 다시
적용한다. 이는 근사 k-NN이나 edge truncation이 아니다. directed edge 수가
`max_graph_edges=2,000,000`을 넘으면 일부 edge를 버리지 않고 실패한다. isolated node도 유지하며
비율과 최대 degree를 결과에 기록한다.

고정 모델은 6-layer, hidden 64의 pure-PyTorch open degree-1 B-spline graph encoder, feature
rasterization, residual U-Net과 analog ConvGRU decoder다. ANN→SNN 변환과 IF timestep은 graph
encoder에만 적용된다. decoder는 두 모드 모두 analog이다.

## 4. 학습과 checkpoint 선택

본학습 설정은 다음과 같다.

- 40 epochs, chronological `batch_size=1`, shuffle 없음
- Adam + gradient centralization, learning rate `1e-3`, weight decay `5e-3`
- MultiStepLR milestones 20/30, gamma 0.1
- CUDA에서 AMP, gradient norm clip 1.0
- Charbonnier 1.0 + SSIM 0.2 + gradient 0.1 + temporal 0.2
- train/validation sample cap 없음, 마지막 epoch에서만 전체 EventHDR eval 평가

ConvGRU state와 temporal loss는 같은 H5 sequence에서 index가 1씩 이어지고 sensor shape가 같을 때만
이어진다. 경계에서는 state와 이전 prediction/target을 초기화한다. 내부 평가도 모든 frame을
사용하므로 별도 표본 추출은 없다.

training artifact는 `runs/eventhdr_asgcn/`에 기록한다.

- `config.json`: 실행 시 resolve된 전체 config
- `history.json`: epoch별 loss, 마지막 epoch validation, learning rate, CUDA peak memory
- `last.pt`: 매 epoch 끝에 저장하는 model/optimizer/scheduler/scaler/RNG 포함 재개 checkpoint
- `best.pt`: 현재 protocol에서는 마지막 epoch의 clean ANN inference checkpoint
- `.data_hash_cache.json`: full source hash 계산을 가속하는 로컬 cache

새 학습은 위 핵심 artifact가 이미 있는 run directory를 덮어쓰지 않는다. 중단된 run은 `last.pt`로
재개한다.

## 5. 전체 ANN→SNN 보정

기본 본실험은 EventHDR train의 모든 frame을 사용한다.

```bash
bash scripts/calibrate.sh \
  configs/hdr_train.json \
  runs/eventhdr_asgcn/best.pt \
  runs/eventhdr_asgcn/best_snn.pt
```

wrapper 기본값 `CALIBRATION_SAMPLES=all`은 모든 training frame을 선택한다. model을 eval mode로
전환해 graph BN을 convolution parameter에 fold하고, non-empty sample의 layer별 feature-wise ReLU
maximum을 측정한 뒤 식 (6) parameter normalization과 unit threshold를 적용한다. 출력
`best_snn.pt`는 optimizer와 training history를 제거한 SNN inference checkpoint이며 calibration
표본 수·유효 표본 수·dead channel 수·sampling summary와 model tensor SHA-256을 포함한다.

기존 출력은 기본적으로 보호한다. 명시적으로 다시 보정해야 할 때만
`OVERWRITE_CALIBRATION=1`을 wrapper에 전달한다. engine은 새 checkpoint를 끝까지 만든 뒤 atomic
replace하므로 변환 실패 전에 기존 파일을 먼저 삭제하지 않는다. ANN 평가는 변환 전 `best.pt`, SNN
평가는 `best_snn.pt`를 써야 하며 서로 바꾸면 checkpoint 검증에서 거부된다.

## 6. 고정 평가·benchmark 행렬

`scripts/full.sh`는 full data/environment 검사와 모든 선택 sample의 decode 검증, 학습 또는 재개,
전체 보정, 아래 행렬의 evaluate와 benchmark를 순서대로 실행한다.

| dataset | mode | dynamics | T | checkpoint |
|---|---|---|---|---|
| EventHDR eval 19 | ANN | 해당 없음 | 해당 없음 | `best.pt` |
| EventHDR eval 19 | SNN | `literal_eq15` | 4, 8, 16, 32 | `best_snn.pt` |
| EventHDR eval 19 | SNN | `standard_if` | 4, 8, 16, 32 | `best_snn.pt` |
| EventAid-R 14 | ANN | 해당 없음 | 해당 없음 | `best.pt` |
| EventAid-R 14 | SNN | `literal_eq15` | 4, 8, 16, 32 | `best_snn.pt` |
| EventAid-R 14 | SNN | `standard_if` | 4, 8, 16, 32 | `best_snn.pt` |

즉 dataset마다 ANN 1개와 SNN 8개, 전체 18개 quality evaluation과 18개 compute benchmark를 만든다.
benchmark 기본값은 warmup 10 frame, 측정 100 frame이다. recurrent benchmark는 각 측정 window 앞의
같은 H5/ZIP sequence predecessor를 최대 32개까지 timer 밖에서 replay한다. dataset read와
host-to-device 이동도 timer 밖이며, CUDA에서는 CUDA Event로 model forward를 측정한다. graph 생성은
model forward 안에 있으므로 측정에 포함된다.

`literal_eq15`는 공개 식 (15)의 `+h(t-1)`까지 문자 그대로 실행한다. `standard_if`는 그 항을 뺀
rate-conversion 대조군이며 저자 공식 dynamics라고 주장하지 않는다. 두 모드 모두 T회의 IF
recurrence를 실제 실행하지만 analog decoder 때문에 결과를 완전 SNN hardware 성능으로 해석하지
않는다.

## 7. 지표와 산출물

quality는 frame별 PSNR, Gaussian-window SSIM, RMSE와 연속 frame 사이의 `temporal_l1`을 기록한다.
`temporal_l1`은
`L1((pred_t-pred_{t-1}), (target_t-target_{t-1}))`이며 sequence 경계의 첫 frame은 집계에서 제외되고
CSV에는 null이다. 요약은 모든 frame의 `micro`, H5/ZIP group 평균을 다시 평균한 `macro`, 호환 key
`per_scene`으로 나뉜다. LPIPS는 `eval.lpips=true`와 optional dependency를 명시한 별도 run에서만
계산한다.

evaluate는 quality, model-forward latency, RTF/deadline miss, graph topology, dataset coverage와
CUDA peak allocated/reserved memory를 기록한다. benchmark는 mean/p50/p90/p95/p99/max latency, FPS,
raw/retained event rate, graph node rate, event retention, node/edge/isolated-node 수, SNN firing rate,
RTF, state reset과 peak GPU memory를 기록한다.

```text
runs/eventhdr_official_eval_ann/ann/
runs/eventhdr_official_eval_snn/snn_<literal_eq15|standard_if>_T<4|8|16|32>/
runs/eventaid_r_external_ann/ann/
runs/eventaid_r_external_snn/snn_<literal_eq15|standard_if>_T<4|8|16|32>/
```

각 directory의 `metrics.json`, `frames.csv`, `predictions/`는 evaluate가 만들고
`benchmark.json`은 benchmark가 만든다. prediction은 config 기본값에 따라 처음 20 frame의 pred/GT
PNG를 저장한다. 같은 mode/dynamics/T artifact가 있으면 묵시적으로 덮어쓰지 않고 실패하므로 재실행
전 기존 결과를 보존 위치로 옮기거나 별도 `eval.output_dir` config를 사용한다.

## 8. epoch-boundary exact resume

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/full.sh
```

`last.pt`는 완료된 epoch 뒤에만 저장된다. 따라서 중간에 종료된 epoch의 일부 step부터 이어가는 것이
아니라 마지막으로 완료된 epoch 다음부터 동일 trajectory를 재개한다. resume checkpoint는 configured
`run_dir` 안에 있어야 하고, 검증 score가 이미 있으면 같은 run의 historical `best.pt`도 필요하다.

exact resume은 다음 항목의 일치를 강제한다.

- model, optimizer/GC, scheduler, loss, batch/order/worker, AMP·TF32·determinism 설정
- seed, validation protocol, official manifest, transform와 선택 sample identity
- train 51개와 eval 19개 source의 상대 파일 identity·size·full SHA-256
- `src/**/*.py` byte의 SHA-256, Git commit과 source dirty 상태
- device type, PyTorch/CUDA/cuDNN, GPU 이름·compute capability와 CUDA backend flags
- CUDA RNG state 수와 현재 visible CUDA device 수

따라서 다른 GPU 종류, CUDA/PyTorch 조합, source checkout, worker protocol이나 dataset byte로 옮긴
checkpoint는 “exact” 재개로 허용되지 않는다. 절대 data mount path와 mtime/ctime 자체는 checkpoint
protocol에 넣지 않는다. 같은 상대 파일과 byte를 다른 mount에 복사하면 다시 full hash한 뒤 일치할
수 있다. `.data_hash_cache.json`은 같은 절대 경로의 size/mtime/ctime이 모두 같을 때만 기존 full
hash를 재사용하며, 원본을 교체·복원했으면 `train.rehash_data=true`인 별도 config로 cache를 무시한다.

## 9. 과학적 한계와 중단 조건

- 공개된 저자 공식 코드가 확인되지 않아 논문에 없는 graph normalization, spline hyperparameter,
  threshold 세부값은 이 저장소의 명시적 가정이다. “공식 ASGCN 완전 재현”으로 표현하지 않는다.
- 원 ASGCN의 classification pooling/MLP와 dynamic asynchronous K-hop update는 구현하지 않았다.
  현재 graph는 frame interval마다 정적으로 다시 만든다.
- residual U-Net, rasterization, analog ConvGRU, 복원 loss, tone mapping, 정확한 `max_events` cap과
  EventAid-R offset은 과제용 확장이다.
- graph encoder만 IF로 변환된다. PyTorch GPU latency·발화율은 FPGA/ASIC latency, power 또는 energy
  측정이 아니며 event 전송·압축 protocol이나 RTL도 구현하지 않았다.
- EventHDR 공식 train/eval의 물리 scene 독립성을 입증하지 못했으므로 내부 결과에 독립 test-set
  일반화 주장을 붙이지 않는다. EventAid-R도 sensor/domain 차이를 통제한 benchmark는 아니다.
- `literal_eq15`의 self-feedback은 표준 rate conversion과 수학적으로 동등하지 않을 수 있다.
  dynamics와 T를 생략한 ANN↔SNN 비교는 보고하지 않는다.
- file coverage/HDF5·ZIP decode 실패, non-finite loss/metric, 비단조 timestamp, 좌표 범위 오류,
  `max_graph_edges` 초과 또는 OOM이 발생하면 해당 sample을 조용히 제외하지 말고 run을 폐기해 원인과
  설정을 고친 뒤 새 artifact로 실행한다.
~~~~~~~~

# docs/SERVER.md

~~~~~~~~markdown
# MobaXterm·Linux GPU 서버 실행 가이드

MobaXterm은 Windows PC에서 Linux 서버에 접속하는 SSH terminal/SFTP client다. 학습과 평가는
MobaXterm 자체가 아니라 접속한 GPU server 또는 scheduler compute node에서 실행한다. 아래 명령은
저장소 root 기준이며, 전체 EventHDR와 EventAid-R를 사용하는 본실험 경로만 설명한다.

## 1. clone과 환경 설치

MobaXterm에서 SSH session을 열고 서버 terminal에서 실행한다.

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

python3.12 --version
python3.12 -c "import venv, ensurepip; print('venv/ensurepip: OK')"
curl --version | head -n 1
ldd --version | head -n 1

cp .env.example .env
# .env의 TORCH_INDEX_URL을 서버 driver에 맞는 PyTorch 공식 CUDA wheel index로 설정
bash scripts/setup.sh

source .venv/bin/activate
python -m pip check
python scripts/check_env.py --lock constraints/py312.txt
python -m pytest -q
```

공식 wheel index는 서버의 `nvidia-smi`와
[PyTorch 설치 선택기](https://pytorch.org/get-started/locally/)를 기준으로 정한다. 명령행 환경변수가
`.env`보다 우선하므로 GPU allocation 안에서 설치까지 검증하려면 다음처럼 실행할 수 있다.

```bash
REQUIRE_CUDA=1 TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXYZ \
  bash scripts/setup.sh
```

`constraints/py312.txt`는 Python 3.12 profile과 core/dev package version을 고정한다. setup script는
Python, 기존 venv, constraints, Linux glibc와 선택한 torch wheel을 검사하고 마지막에 `pip check`를
실행한다. login node에서 GPU가 보이지 않는 것은 정상일 수 있으므로 `--require-cuda` 검사는 실제 GPU
allocation 안에서 수행한다. cluster가 module을 쓴다면 Python/CUDA module을 먼저 load한다. Slurm
wrapper들은 필요할 때 `CUDA_MODULE=cuda/<version>`도 받는다.

## 2. 전체 데이터 배치

최종 layout은 다음과 같다.

```text
data/
├── EventHDR/
│   ├── train/1.h5 ... 51.h5
│   └── eval/1.h5 ... 19.h5
└── EventAid-R/
    └── R-*.zip                 # manifest의 14개 ZIP
```

### EventHDR: browser download 뒤 import

공식 EventHDR OneDrive folder는 현재 unattended `curl` download를 허용하지 않는다.
`scripts/get_hdr.sh`도 이를 우회한다고 주장하지 않으며, 이미 받은 archive/extracted directory를
검사해 가져오는 도구다.

1. Windows browser에서 공식 OneDrive의 train/eval release를 내려받는다.
2. MobaXterm 왼쪽 SFTP panel로 서버의 예: `$HOME/uploads/`에 ZIP을 전송한다.
3. 배포가 train/eval 별도 archive이면 각각 import한다.

```bash
bash scripts/get_hdr.sh \
  --archive "$HOME/uploads/EventHDR-train.zip" --split train
bash scripts/get_hdr.sh \
  --archive "$HOME/uploads/EventHDR-eval.zip" --split eval
```

train/eval을 함께 담은 한 archive이면 `--split` 없이 한 번 실행한다.

```bash
bash scripts/get_hdr.sh --archive "$HOME/uploads/EventHDR.zip"
```

이미 압축을 푼 directory는 안전하게 copy할 수 있다. source는 그 자체가 `train/`·`eval/`을
포함하거나 상위 `EventHDR/` directory여도 된다.

```bash
bash scripts/get_hdr.sh --source /shared/imports/EventHDR
```

shared storage의 검증된 H5를 복사하지 않고 쓸 때는 split directory symlink를 만든다.

```bash
bash scripts/get_hdr.sh --source /shared/datasets/EventHDR --link
```

`--link`는 source가 정확한 파일 집합인지 먼저 검사하고, destination split이 비어 있지 않으면
교체하지 않는다. 현재 destination만 재검사하려면 다음 명령을 쓴다.

```bash
bash scripts/get_hdr.sh --check
```

import/check는 train `1.h5`–`51.h5`, eval `1.h5`–`19.h5`의 누락·추가 파일, nested H5, HDF5 magic과
합계 100 GB 미만 조건을 검사한다. 기존 이름의 크기가 다른 파일을 덮어쓰지 않으며 archive extraction은
temporary file과 atomic replace를 사용한다. 다른 위치를 쓰려면 모든 명령에
`--destination /absolute/path/EventHDR`를 추가하고 config의 root도 그 위치에 맞춰야 한다.

### EventAid-R 14개 ZIP

```bash
bash scripts/get_aid.sh --all
```

manifest의 14개 scene, 표시 합계 약 24.68 GB를 내려받는다. 각 파일이 ZIP container인지 검사하며
loader가 archive member를 직접 읽으므로 압축을 풀지 않는다. shared storage를 쓰면
`EVENTAID_ROOT=/shared/datasets/EventAid-R bash scripts/get_aid.sh --all`로 destination을 바꾸거나,
config가 기대하는 `data/EventAid-R`를 해당 directory에 연결한다.

### full-data와 decode 검사

GPU allocation 안에서 다음 검사를 한 번 통과시킨다.

```bash
source .venv/bin/activate
python scripts/check_env.py \
  --require-cuda --require-full-data --lock constraints/py312.txt

asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all
```

`hdr_train.json` inspect는 manifest에 따라 EventHDR train 51개와 eval 19개 root를 모두 검사한다.
EventAid 명령은 manifest의 14개 ZIP에 있는 모든 선택 event block과 target을 decode한다.
`--validate-all`은 metadata만 세는 명령이 아니므로 dataset 크기에 따라 오래 걸린다. 실패한 file을
제외해 진행하지 말고 원본을 다시 전송·검증한다.

## 3. 직접 서버에서 전체 실행

실행 순서를 먼저 확인하려면 data/GPU 작업을 실제 수행하지 않는 schedule 출력을 본다.

```bash
DRY_RUN=1 bash scripts/full.sh
```

GPU shell 또는 allocation 안에서 전체 protocol을 시작한다.

```bash
source .venv/bin/activate
mkdir -p logs
bash scripts/full.sh 2>&1 | tee logs/full.log
```

SSH 연결 종료에 대비하려면 tmux를 사용한다.

```bash
mkdir -p logs
tmux new-session -s asgcn -c "$PWD" \
  "bash -lc 'source .venv/bin/activate && bash scripts/full.sh 2>&1 | tee logs/full.log'"
```

`full.sh`는 다음을 순서대로 수행한다.

1. `check_env.py --require-full-data --lock constraints/py312.txt`와 기본 CUDA 검사
2. EventHDR train/eval과 EventAid-R 전체 `inspect --validate-all`
3. EventHDR ANN 40-epoch 학습 또는 `RESUME_CHECKPOINT` 재개
4. EventHDR train 모든 frame을 이용한 `best.pt`→`best_snn.pt` 보정
5. EventHDR/EventAid-R ANN과 `literal_eq15`/`standard_if` × `T=4,8,16,32` evaluate+benchmark

기본 benchmark는 warmup 10, 측정 100 frame이다. 필요하면 실행 전에
`BENCHMARK_WARMUP`, `BENCHMARK_STEPS`, `SIMULATION_STEPS_LIST`를 설정한다. 본실험 기본 행렬을 바꾼
결과는 별도 protocol로 기록한다.

기존 training/evaluation artifact는 묵시적으로 덮어쓰지 않는다. 기존 SNN checkpoint만 의도적으로
다시 만들 때는 `OVERWRITE_CALIBRATION=1`을 사용하며 새 checkpoint가 완성된 뒤 atomic replace된다.
evaluation directory가 이미 있으면 결과를 다른 보존 위치로 옮기거나 config의 `eval.output_dir`을
바꾼 뒤 실행한다.

## 4. 중단 후 epoch-boundary resume

직접 실행은 다음과 같다.

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/full.sh 2>&1 | tee logs/full-resume.log
```

학습만 재개하려면 다음 wrapper를 쓴다.

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/train.sh configs/hdr_train.json
```

`last.pt`는 각 완료 epoch 뒤에 저장되므로 종료된 epoch 내부 step은 되풀이된다. checkpoint는 같은
configured run directory 안에 있어야 하며, source tree/Git 상태, model·optimizer·scheduler·AMP,
validation/data full SHA-256과 PyTorch/CUDA/cuDNN·GPU 이름/compute capability·visible CUDA RNG state가
일치해야 한다. 다른 GPU나 source checkout으로 옮기는 것은 일반 weight load가 아니라 exact resume
요청이므로 거부될 수 있다. 상세 계약은 [EXPERIMENT.md](EXPERIMENT.md)의 resume 절을 따른다.

## 5. Slurm: train → calibrate → evaluation matrix

header의 partition/account/GPU type/walltime은 cluster 정책에 맞춰 수정한다. 기본 요청은 GPU 1개,
CPU 8개, RAM 32 GB이며 train 48시간, calibration 12시간, evaluation 8시간이다. 저장소 root에서
제출하면 `SLURM_SUBMIT_DIR`을 project root로 사용한다. 다른 위치에서 제출할 때는
`--export=ALL,PROJECT_ROOT=/absolute/path/to/repo`를 추가한다.

앞 절의 full-data/decode 검사를 완료한 뒤 다음 dependency chain을 제출한다.

```bash
unset SNN_DYNAMICS

train_id=$(sbatch --parsable \
  --export=ALL,VALIDATE_DATASET=0 \
  server/train.sbatch)

cal_id=$(sbatch --parsable \
  --dependency="afterok:${train_id}" \
  --export=ALL,VALIDATE_DATASET=0,CALIBRATION_SAMPLES=all \
  server/calibrate.sbatch)

for cfg in configs/hdr_ann.json configs/aid_ann.json; do
  sbatch --dependency="afterok:${cal_id}" \
    --export="ALL,VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH=${cfg},CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt,INFERENCE_MODE=ann" \
    server/eval.sbatch
done

for cfg in configs/hdr_snn.json configs/aid_snn.json; do
  for dynamics in literal_eq15 standard_if; do
    for steps in 4 8 16 32; do
      sbatch --dependency="afterok:${cal_id}" \
        --export="ALL,VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH=${cfg},CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS=${dynamics},SIMULATION_STEPS=${steps}" \
        server/eval.sbatch
    done
  done
done
```

모든 evaluation job은 calibration 성공 후 열리며 mode/dynamics/T와 dataset별 output directory가 달라
서로의 artifact를 덮어쓰지 않는다. Slurm log는 기본적으로 `slurm-<job-name>-<job-id>.out/.err`다.
학습 재개 job으로 chain을 시작할 때는 첫 제출에 다음 값을 더한다.

```bash
train_id=$(sbatch --parsable \
  --export=ALL,VALIDATE_DATASET=0,RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  server/train.sbatch)
```

## 6. PBS/Torque: train → calibrate → evaluation matrix

`select`, `ngpus`, queue/project resource 이름은 site마다 다르므로 `server/*.pbs` header를 제출 전에
확인한다. 저장소 root에서 제출하면 `PBS_O_WORKDIR`을 project root로 사용한다. 외부에서 제출할 때는
`-v PROJECT_ROOT=/absolute/path/to/repo`를 추가한다.

```bash
unset SNN_DYNAMICS

train_id=$(qsub \
  -v VALIDATE_DATASET=0 \
  server/train.pbs)

cal_id=$(qsub \
  -W depend="afterok:${train_id}" \
  -v VALIDATE_DATASET=0,CALIBRATION_SAMPLES=all \
  server/calibrate.pbs)

for cfg in configs/hdr_ann.json configs/aid_ann.json; do
  qsub -W depend="afterok:${cal_id}" \
    -v VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH="${cfg}",CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt,INFERENCE_MODE=ann \
    server/eval.pbs
done

for cfg in configs/hdr_snn.json configs/aid_snn.json; do
  for dynamics in literal_eq15 standard_if; do
    for steps in 4 8 16 32; do
      qsub -W depend="afterok:${cal_id}" \
        -v VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH="${cfg}",CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS="${dynamics}",SIMULATION_STEPS="${steps}" \
        server/eval.pbs
    done
  done
done
```

PBS에서 resume chain을 시작할 때는 다음처럼 `RESUME_CHECKPOINT`를 넘긴다.

```bash
train_id=$(qsub \
  -v VALIDATE_DATASET=0,RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  server/train.pbs)
```

PBS는 login environment 전체를 전달하는 `#PBS -V`를 사용하지 않는다. site module이 필수이면
해당 cluster용 `module load`를 job script에 추가한다. 별도 venv나 checkout을 쓸 때는
`PYTHON_BIN`, `PROJECT_ROOT`를 `-v`로 명시한다.

## 7. 산출물 확인과 운영상 오류

학습이 끝나면 다음 파일을 먼저 확인한다.

```bash
ls -lh runs/eventhdr_asgcn/{last.pt,best.pt,best_snn.pt,history.json,config.json}
find runs -name metrics.json -o -name benchmark.json | sort
```

평가 artifact는 다음 위치에 있다.

```text
runs/eventhdr_official_eval_ann/ann/
runs/eventhdr_official_eval_snn/snn_<dynamics>_T<steps>/
runs/eventaid_r_external_ann/ann/
runs/eventaid_r_external_snn/snn_<dynamics>_T<steps>/
```

각 run의 `metrics.json`, `frames.csv`, `predictions/`, `benchmark.json`을 config, Git commit, scheduler
log, `check_env.py` 출력과 함께 보존한다.

자주 중단되는 조건은 다음과 같다.

- `CUDA available: false`: login node가 아닌 GPU allocation인지, CUDA wheel과 driver가 맞는지 확인한다.
- `EventHDR ... exact official file set`: OneDrive 전송이 끝났는지, train 51/eval 19 외 H5가 섞이지
  않았는지 `get_hdr.sh --check`로 확인한다.
- `eventaid_r_zip must contain exactly 14`: `get_aid.sh --all`을 완료하고 ZIP을 압축 해제하지 않는다.
- `Fresh training run_dir is not empty`: 새 run이면 별도 `output.run_dir`, 중단 run이면 `last.pt` resume를
  사용한다.
- resume protocol mismatch: source/GPU/software/data를 원래 run과 일치시킨다. 단순 checkpoint weight
  이식과 exact training resume를 혼동하지 않는다.
- calibrated checkpoint 오류: ANN `best.pt`를 `scripts/calibrate.sh`로 변환한 뒤 SNN 평가에
  `best_snn.pt`를 사용한다.
- evaluation output exists: 기존 artifact를 보존 위치로 옮기거나 별도 output directory를 쓴다.
- `max_graph_edges=2,000,000` 또는 OOM: edge를 조용히 잘라 진행하지 않는다. 별도 config에서
  `max_events`, `graph_radius`, model width를 변경하고 peak memory를 다시 측정해 다른 실험으로 기록한다.
- SSH 종료: foreground shell 대신 tmux, Slurm 또는 PBS job을 사용한다.
~~~~~~~~

# hand_off.md

~~~~~~~~markdown
# ASGCN Event Reconstruction 프로젝트 인계서

이 문서는 다른 ChatGPT나 연구자가 현재 저장소를 교차검증하고 Linux GPU 서버에서 전체 실험을
이어가기 위한 기준 문서다. 코드와 config가 최종 진실이며, 아래 내용은 2026-08-29의 현재
구현과 일치하도록 다시 대조했다.

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

원격 origin은 다음 주소로 설정돼 있다.

```text
https://github.com/costunder/asgcn-event-reconstruction.git
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
추출하지 않고 직접 읽어 중복 저장을 피한다. EventHDR를 browser archive에서 복사할 때는 archive와
배치된 H5가 일시적으로 함께 존재할 수 있으므로 서버의 실제 여유 공간은 별도로 확인한다.

### 3.1 EventHDR 획득과 배치

공식 배포는 [EventHDR 저장소](https://github.com/yunhao-zou/EventHDR)의 OneDrive 링크다. OneDrive가
비대화형 `curl` 요청을 거부하므로 이 저장소는 동작하지 않는 자동 downloader를 제공하지 않는다.
사용자가 browser로 받은 ZIP, 이미 풀어 둔 directory 또는 shared filesystem directory를 아래
도구로 안전하게 배치한다.

```bash
# browser로 받은 train/eval 포함 ZIP을 직접 읽어 data/EventHDR로 복사
bash scripts/get_hdr.sh --archive /absolute/path/EventHDR.zip

# 이미 풀어 둔 EventHDR/{train,eval} 또는 {train,eval} root에서 복사
bash scripts/get_hdr.sh --source /absolute/path/EventHDR

# shared storage를 복사하지 않고 split directory symlink로 연결
bash scripts/get_hdr.sh --source /shared/datasets/EventHDR --link

# train/eval을 따로 받았을 때
bash scripts/get_hdr.sh --source /downloads/train --split train
bash scripts/get_hdr.sh --source /downloads/eval --split eval

# 현재 목적지 재검사
bash scripts/get_hdr.sh --check
```

`scripts/get_hdr.py`는 train의 정확한 51개 이름, eval의 정확한 19개 이름, missing/extra/nested H5,
archive 중복·경로 이탈, HDF5 magic과 선택 데이터 100 GB 미만을 검사한다. 복사는 `.part` 임시 파일
뒤 atomic replace로 완료하며 기존의 다른 크기 파일은 덮어쓰지 않는다. 공식 checksum이 공개되지
않았으므로 이 검사는 배포자 cryptographic checksum 검증을 대신하지 않는다.

### 3.2 EventAid-R 획득과 배치

`manifests/eventaid_r.json`에는 공식 benchmark page가 연결한 14개 Dropbox URL, scene 이름과 표시
용량이 고정돼 있다. Linux에서는 인자 없이 실행하면 전체 14개를 받는다.

```bash
bash scripts/get_aid.sh
```

Windows PowerShell에서는 다음을 쓴다.

```powershell
.\scripts\get_aid.ps1
```

Linux downloader는 재개 가능한 `curl`, retry와 ZIP container 검사를 사용한다. 공식 checksum이
없으므로 최종 내용 검사는 뒤의 `inspect --validate-all` 단계에서 모든 event block과 target을
decode하는 방식이다.

### 3.3 loader 의미론

EventHDR loader는 H5의 `events/{xs,ys,ts,ps}`와 `images/image*`의 `event_idx`, `timestamp`를 검사한다.
timestamp·event boundary가 단조롭고 좌표·polarity가 유효한지 확인한다. `frame_stride=1`에서 모든
target interval을 유지하며 event가 0개인 interval도 삭제하지 않는다. 빈 interval은 zero-node graph와
zero raster를 거쳐 recurrent decoder로 전달된다. `frame_stride>1`이면 건너뛴 interval의 event를
다음 선택 target까지 합치지만 기본값은 1이다.

EventAid-R loader는 ZIP 안의 `event/i.txt`, `gt/j_img.png`, `timestamps.txt`, `shape.txt`를 직접
읽는다. config의 `target_offset=1`은 event interval `i`를 다음 GT `i+1`과 짝짓는 구현 가정이다.
연속 ID, timestamp coverage, shape, 좌표와 polarity를 검증한다. 이 pairing을 저자 공식 코드로
확인한 것은 아니므로 보고서에서 가정으로 표시하고 offset이 다른 실험과 결과를 섞지 않는다.

두 dataset 모두 target을 `[0,1]` luminance로 만든 뒤 기본 config에서
`log1p(5000*x)/log1p(5000)`를 적용한다. EventAid-R의 8-bit 영상에 같은 log mapping을 쓰는 것은
출력 수치 domain을 맞추기 위한 cross-domain 선택이지 두 센서의 radiometric response가 같다는
뜻이 아니다.

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

`configs/hdr_train.json`의 `validate_every=null`은 EventHDR 공식 eval을 매 epoch 보지 않고 마지막
40번째 epoch에서 단 한 번만 실행한다. 그 하나의 candidate를 `best.pt`로 export하므로 epoch 간
selection은 하지 않는다. 그래도 같은 eval에서 산출한 수치는 독립 test나 physical-scene test가
아니며 `EventHDR official eval internal result`로만 보고한다. 이 결과를 보고 hyperparameter를
바꾸면 이후 run에서는 사실상 개발 정보로 사용한 것이므로 독립성 주장을 더 할 수 없다.

EventAid-R은 training과 calibration에 사용하지 않는다. 외부 결과를 본 뒤 radius, cap, threshold,
tone mapping 또는 checkpoint를 바꾸면 기존 EventAid-R 결과를 잠긴 외부 일반화 평가로 부를 수 없다.

## 5. 기본 config와 학습 규칙

`configs/hdr_train.json`의 핵심값은 다음과 같다.

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
| stability | CUDA AMP, L2 grad clip 1.0, non-finite loss/gradient fail-fast |
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

## 6. graph, B-spline, decoder 요약

좌표는 `x/(W-1)`, `y/(H-1)`, interval 내 normalized `t`이고 polarity는 `-1/+1` node feature다.
기본 거리에는 `x,y,t`만 쓴다. radius graph는 cell 폭을 정확히 `D`로 두고 `3^d` 인접 cell에서
후보를 찾은 뒤 Euclidean `distance<D`를 다시 검사한다. 모든 ordered source를 처리해 무방향 쌍의
양 방향 edge를 만들고 self-loop는 제외한다. chunking은 exact 계산 분할이며 approximation이 아니다.

edge pseudo-coordinate `u=distance/D`에 open degree-1 B-spline basis 두 개만 활성화된다. layer는
node를 K=5 control point에 한 번 projection하고 edge마다 두 control point만 gather한 뒤 destination
incoming degree로 평균한다. 고정 graph의 basis/index는 graph layer와 IF timestep 전체에서
재사용한다. mean message에 root transform과 bias를 더하고 ANN에서는 BatchNorm과 ReLU를 적용한다.

node feature는 downsample 4의 raster cell 안에서 평균된다. decoder는 stem, 두 residual encoder
level, 두 residual block bottleneck, analog ConvGRU, bilinear upsampling과 skip connection, sigmoid
head로 구성된다. 자세한 구현 가정과 식 (15) 모호성은 `docs/ASGCN.md`를 본다.

## 7. ANN→SNN calibration과 IF 경로

`best.pt`는 변환되지 않은 ANN inference checkpoint다. calibration은 EventHDR train만 사용하며
기본 wrapper는 모든 train sample을 사용한다.

1. ANN graph layer의 BatchNorm을 kernel/root/bias에 fold한다.
2. 각 layer의 feature별 ReLU maximum `lambda_l`를 측정한다.
3. 식 (6)의 `lambda_(l-1)/lambda_l`와 `1/lambda_l` scaling을 적용한다.
4. dead channel은 unit scale로 두고, 모든 threshold를 정확히 1로 둔다.
5. `best_snn.pt`에 valid sample count, dead-channel summary, persistent conversion flag와 tensor
   SHA-256을 저장한다.

SNN inference는 threshold/normalization/calibration metadata와 layer state가 모두 일치해야 열린다.
초기 membrane은 0.5 threshold, spike amplitude는 threshold, soft reset을 쓴다.

- `literal_eq15`: 논문 식 (15)의 `+previous_spike` self-feedback까지 문자 그대로 실행한다.
- `standard_if`: 그 항을 제거한 비공식 rate-conversion 대조군이다.

두 dynamics는 같은 `best_snn.pt`에서 inference-only override로 비교한다. 마지막 graph layer의 spike
rate에는 `lambda_L`를 곱해 analog decoder 단위로 보낸다. 이는 `literal_eq15`의 ANN parity 증명이
아니다.

## 8. `scripts/full.sh`의 전체 실행 순서

이 script는 설치나 데이터 다운로드를 하지 않는다. 환경과 전체 데이터가 이미 준비된 뒤 저장소
루트에서 실행한다.

```bash
bash scripts/full.sh
```

실행 단계는 정확히 다음 5개다.

1. `check_env.py --require-full-data --lock constraints/py312.txt`와 선택적 CUDA 검사
2. `hdr_train`, `hdr_ann`, `aid_ann` 세 config에 대해 `inspect --validate-all`
3. EventHDR ANN 40-epoch 학습 또는 `RESUME_CHECKPOINT` exact resume
4. EventHDR train 전체를 사용한 ANN→SNN calibration
5. EventHDR와 EventAid-R의 전체 quality evaluation + compute benchmark matrix

2단계에서 `hdr_train` inspect는 manifest의 train 51개와 eval 19개 split을 모두 decode한다.
`hdr_ann`은 standalone EventHDR eval config를, `aid_ann`은 14개 ZIP 전체를 다시 검증한다. 오래 걸려도
파일을 조용히 제외하지 않는다.

5단계 matrix는 다음 18개 run이며 각 run마다 `evaluate`와 `benchmark`를 둘 다 실행한다.

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
DRY_RUN=1 bash scripts/full.sh
```

중요 override는 `RESUME_CHECKPOINT`, `CALIBRATION_SAMPLES`, `SIMULATION_STEPS_LIST`,
`BENCHMARK_WARMUP`, `BENCHMARK_STEPS`, 다섯 config path, ANN/SNN checkpoint path와
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

## 10. provenance, checkpoint integrity와 exact resume

학습 directory의 핵심 artifact는 `config.json`, `history.json`, `last.pt`, `best.pt`와 hidden data hash
cache다. `validate_every=null`이므로 `history.json`의 validation은 마지막 epoch에서만 채워지고
`best.pt`는 그 마지막 candidate다.

validation protocol에는 dataset transform, manifest schema와 모든 file 목록/group mapping,
validation sample identity/context policy, SSIM 정의, selection rule과 train/eval 원본 전체 file의
SHA-256 결합 digest가 저장된다. 절대 root와 mtime은 checkpoint 비교 identity가 아니어서 같은 byte의
복사본을 다른 mount에서 쓸 수 있다. hash cache는 같은 절대 path의 size/mtime/ctime이 모두 같을 때
기존 full hash를 재사용한다. 원본을 교체·복원했거나 강제 전수 hash가 필요하면
`train.rehash_data=true`를 둔다.

training protocol에는 optimizer/GC 축, scheduler, loss weights, gradient clip, data order/workers,
effective AMP, final-only validation rule, recurrent detach, torch/CUDA/cuDNN/GPU/TF32/determinism,
`src/**/*.py` tree hash, Git commit과 source dirty 여부가 들어간다. checkpoint의 model tensor bytes도
이름·dtype·shape를 포함해 SHA-256으로 묶는다.

exact resume은 다음을 모두 요구한다.

- resume checkpoint가 같은 configured run directory 안에 있을 것
- model config, validation protocol, training protocol과 source/data digest가 일치할 것
- optimizer, scheduler, GradScaler, history, epoch와 best score가 있을 것
- Python, NumPy, torch와 visible CUDA device별 RNG state가 유효할 것
- 과거 `best.pt`가 존재하고 `last.pt`의 best digest/protocol과 일치할 것
- ANN/SNN conversion state와 checkpoint type이 일치할 것

학습만 이어갈 때는 다음을 사용한다.

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/train.sh configs/hdr_train.json
```

provenance가 엄격하므로 commit이나 source/runtime 변경 뒤에는 resume이 거부될 수 있다. 또한 runtime
상태를 기록하고 비교하더라도 PyTorch/CUDA의 모든 kernel이 bitwise deterministic하다는 보장은
없다. exact resume은 저장된 state와 protocol의 정확한 복원을 뜻하며 서로 다른 hardware에서의
bitwise 동일성을 과장하지 않는다.

## 11. MobaXterm/Linux GPU 서버 절차

MobaXterm은 SSH/SFTP client이고 실제 연산은 접속한 Linux server에서 수행한다.

```bash
git clone https://github.com/costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

python3.12 --version
ldd --version | head -n 1
nvidia-smi

cp .env.example .env
# .env에서 server driver와 맞는 공식 TORCH_INDEX_URL 등을 설정
bash scripts/setup.sh
source .venv/bin/activate

python scripts/check_env.py --require-cuda --lock constraints/py312.txt
python -m pip check
python -m ruff check .
python -m pytest -q
```

프로젝트는 Python 3.10 이상을 지원한다. 재현용 lock은 Python 3.12.13에서 검증됐고 core/dev package와
torch public version을 `constraints/py312.txt`에 고정한다. 현재 lock의 핵심은 torch 2.13.0,
numpy 2.5.2, h5py 3.16.0, Pillow 12.3.0, pytest 9.1.1, Ruff 0.16.5다. Linux torch 2.13.0 lock
profile은 glibc 2.28 이상을 요구한다. CUDA build는 `nvidia-smi` driver와 PyTorch 공식 selector에
맞는 `TORCH_INDEX_URL`에서 먼저 설치한다.

core runtime dependency는 torch, NumPy, h5py, Pillow와 tqdm이다. development extra는 pytest와
Ruff다. LPIPS만 필요할 때 optional eval extra를 설치한다.

```bash
python -m pip install -e '.[eval]'
```

데이터 배치 뒤 전체 readiness를 검사한다.

```bash
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt

asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all
```

`check_env`는 CUDA, GPU 이름/VRAM, Python/torch/CUDA/cuDNN, lock mismatch, glibc, data와 runs의 남은
공간, runs 쓰기 가능 여부, EventHDR exact 51/19 이름과 EventAid-R exact 14 ZIP을 출력·검사한다.
`--validate-all`은 모든 target/event block을 실제 decode하므로 전체 데이터에서는 오래 걸린다.

## 12. scheduler와 container

SLURM과 PBS/Torque 각각 train, calibration, evaluation entrypoint가 있다.

```text
server/train.sbatch        server/train.pbs
server/calibrate.sbatch    server/calibrate.pbs
server/eval.sbatch         server/eval.pbs
```

기본 요청은 GPU 1개, CPU 8개, RAM 32 GB다. train은 48시간, calibration은 12시간, evaluation은
8시간으로 작성돼 있으나 partition/account/GPU type/resource 이름과 walltime은 cluster 규칙과 실제
측정에 맞춰 바꿔야 한다. wrapper는 `PROJECT_ROOT` 또는 scheduler submit directory를 검증하고 잘못된
checkout에서 실행하지 않는다. `CUDA_MODULE`은 opt-in이다.

SLURM dependency 예시:

```bash
train_id=$(sbatch --parsable server/train.sbatch)
cal_id=$(sbatch --parsable --dependency=afterok:${train_id} server/calibrate.sbatch)
sbatch --dependency=afterok:${cal_id} \
  --export=ALL,CONFIG_PATH=configs/hdr_snn.json,CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,SIMULATION_STEPS=16,SNN_DYNAMICS=literal_eq15 \
  server/eval.sbatch
```

PBS/Torque dependency 예시:

```bash
train_id=$(qsub server/train.pbs)
cal_id=$(qsub -W depend=afterok:${train_id} server/calibrate.pbs)
qsub -W depend=afterok:${cal_id} \
  -v CONFIG_PATH=configs/hdr_snn.json,CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,SIMULATION_STEPS=16,SNN_DYNAMICS=literal_eq15 \
  server/eval.pbs
```

전체 18-run matrix를 scheduler로 돌리려면 dataset/dynamics/T별 eval job을 각각 제출해야 한다.
단일 allocation에서 순차 실행할 때만 `scripts/full.sh`를 직접 사용한다.

Dockerfile은 Python 3.12와 같은 lock을 쓰며 compose는 data를 read-only, runs를 writable로 mount한다.

```bash
docker build \
  --build-arg TORCH_VERSION=2.13.0 \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t asgcn-event-reconstruction .

docker compose run --rm experiment \
  inspect --config configs/hdr_train.json --samples 2
```

## 13. 파일별 책임

| 경로 | 책임 |
|---|---|
| `src/asgcn_recon/graph.py` | node 정규화, exact radius graph, B-spline layer, BN fold, Eq.(6), IF loop |
| `src/asgcn_recon/model.py` | graph build 연결, rasterization, residual U-Net, ConvGRU, diagnostics |
| `src/asgcn_recon/data/eventhdr.py` | H5 index/검증, zero-event 보존, frame interval 구성 |
| `src/asgcn_recon/data/eventaid_r.py` | ZIP 직접 읽기, next-GT pairing, timestamp/shape 검증 |
| `src/asgcn_recon/data/common.py` | luminance/tone map, crop, exact-size event cap, sample schema |
| `src/asgcn_recon/data/factory.py` | manifest schema, exact coverage, split-local H5 group |
| `src/asgcn_recon/losses.py` | Charbonnier, SSIM loss, gradient loss |
| `src/asgcn_recon/metrics.py` | Gaussian SSIM, PSNR, RMSE, temporal metric, micro/macro 집계 |
| `src/asgcn_recon/engine.py` | train/validation/calibration/evaluate/benchmark, checkpoint·resume·provenance |
| `src/asgcn_recon/cli.py` | inspect/train/calibrate/evaluate/benchmark CLI |
| `configs/hdr_train.json` | EventHDR 51 train + 19 final-only internal eval 학습 protocol |
| `configs/hdr_ann.json`, `configs/hdr_snn.json` | EventHDR official eval ANN/SNN 실행 |
| `configs/aid_ann.json`, `configs/aid_snn.json` | EventAid-R 14-scene ANN/SNN 외부 실행 |
| `manifests/eventhdr_split.json` | official separate roots와 H5 sequence-file semantics |
| `manifests/eventaid_r.json` | 14 ZIP 이름, URL, 표시 용량 |
| `scripts/setup.sh`, `scripts/check_env.py` | server 설치와 환경/data inventory |
| `scripts/get_hdr.py`, `scripts/get_hdr.sh` | browser/source/shared EventHDR 안전 import/check |
| `scripts/get_aid.sh`, `scripts/get_aid.ps1` | EventAid-R 다운로드 |
| `scripts/train.sh`, `scripts/calibrate.sh`, `scripts/eval.sh` | 개별 GPU wrapper |
| `scripts/full.sh` | 전체 5단계, 18-run matrix orchestration |
| `server/` | SLURM/PBS train→calibrate→eval entrypoint |
| `docs/ASGCN.md` | 논문 core와 구현 가정의 경계 |
| `docs/EXPERIMENT.md`, `docs/SERVER.md` | 실험 protocol과 server 운용 보조 문서 |
| `tests/` | fixture 기반 CPU unit/integration/end-to-end 회귀검사 |

## 14. 테스트 상태와 검증 범위

현재 test collection은 159개이며 로컬 Windows CPU 환경에서 `158 passed, 1 skipped`가 기준이다.
skip은 Windows symlink privilege가 없는 경우의 shared-storage link test다. Ruff도 통과해야 한다.

```bash
python -m ruff check .
python -m pytest -q
bash -n scripts/*.sh server/*.sbatch server/*.pbs
```

주요 회귀 범위는 다음과 같다.

- strict undirected radius graph와 cell implementation의 pairwise reference parity
- degree-1 open B-spline endpoint, gradient, 초기화, hand calculation과 autograd
- BN fold, 식 (6), dead channel, IF soft reset, dynamics 차이, basis cache
- EventHDR/EventAid 구조·timestamp·좌표·polarity·pairing·multiprocess safety
- exact-size event cap 경계와 zero-event frame
- manifest separate-root/physical-scene claim 차단과 exact file coverage
- final-only validation, balanced/context schedule, loss/gradient non-finite guard
- checkpoint tensor digest, conversion state, provenance와 exact resume 거부 조건
- evaluate/benchmark artifact, metrics, temporal continuity와 전체 orchestration matrix

GitHub Actions는 Ubuntu/Windows의 Python 3.10/3.11/3.12 pytest matrix와 Python 3.12 locked Ruff/shell
syntax job을 정의한다. unit test는 공식 대용량 데이터나 GPU 없이 fixture로 실행된다. 따라서 test
통과는 전체 데이터 GPU 품질·속도 결과가 생성됐다는 뜻이 아니다.

## 15. 현재 한계와 교차검증 체크리스트

현재 저장소에 전체 데이터 GPU run 결과나 A6000/A100 benchmark artifact가 커밋돼 있지 않다. 다음
항목은 실제 server에서 `scripts/full.sh`가 완료된 뒤 결과 파일로 검증해야 한다.

- EventHDR/EventAid-R 전체 decode 성공과 총 frame 수
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
- downloader 검사는 공식 checksum을 대체하지 못한다.
- optional LPIPS는 core lock에 포함되지 않는다.
- 실제 sensor ingest, network transport, compression, RTL, synthesis와 power 측정은 범위 밖이다.

다른 ChatGPT가 교차검증할 때는 최소한 다음 질문에 답해야 한다.

1. 결과 설명이 `paper-core 기반 복원 적응` 범위를 넘어 공식 재현을 주장하는가?
2. EventHDR가 정확히 train 51/eval 19이고 EventAid-R이 정확히 14 ZIP인가?
3. H5 sequence group을 physical scene으로 잘못 해석했는가?
4. `validate_every=null`과 마지막 epoch 단 1회 internal eval이 실제 checkpoint에 기록됐는가?
5. event cap이 정확히 8,192개를 선택하고 zero-event interval을 보존하는가?
6. graph가 strict `<D`, 양방향, self-loop 없음이며 cell optimization이 pairwise reference와 같은가?
7. ANN과 SNN checkpoint type, BN fold, Eq.(6), threshold와 tensor digest가 일치하는가?
8. EventAid-R을 본 뒤 model/config를 바꾸지 않았는가?
9. 보고한 숫자가 실제 `metrics.json`, `benchmark.json`, `history.json`과 server provenance에 있는가?

이 아홉 항목 중 하나라도 확인되지 않으면 해당 수치는 예비 내부 결과로만 취급한다.
~~~~~~~~

# Makefile

~~~~~~~~makefile
PYTHON ?= .venv/bin/python

.PHONY: setup data doctor test lint inspect train full

setup:
	bash scripts/setup.sh

data:
	bash scripts/get_aid.sh --all

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

full:
	bash scripts/full.sh
~~~~~~~~

# manifests/eventaid_r.json

~~~~~~~~json
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
~~~~~~~~

# manifests/eventhdr_split.json

~~~~~~~~json
{
  "status": "final",
  "split_schema": "official_separate_roots_v1",
  "group_semantics": "h5_sequence_file_not_physical_scene",
  "source": "https://github.com/yunhao-zou/EventHDR",
  "note": "Official OneDrive distribution split. Train and eval are distinct roots whose numeric basenames intentionally overlap. H5 files are recurrent/metric sequence groups, not claimed physical-scene identities.",
  "train_files": [
    "1.h5", "2.h5", "3.h5", "4.h5", "5.h5", "6.h5", "7.h5", "8.h5", "9.h5",
    "10.h5", "11.h5", "12.h5", "13.h5", "14.h5", "15.h5", "16.h5", "17.h5",
    "18.h5", "19.h5", "20.h5", "21.h5", "22.h5", "23.h5", "24.h5", "25.h5",
    "26.h5", "27.h5", "28.h5", "29.h5", "30.h5", "31.h5", "32.h5", "33.h5",
    "34.h5", "35.h5", "36.h5", "37.h5", "38.h5", "39.h5", "40.h5", "41.h5",
    "42.h5", "43.h5", "44.h5", "45.h5", "46.h5", "47.h5", "48.h5", "49.h5",
    "50.h5", "51.h5"
  ],
  "val_files": [
    "1.h5", "2.h5", "3.h5", "4.h5", "5.h5", "6.h5", "7.h5", "8.h5", "9.h5",
    "10.h5", "11.h5", "12.h5", "13.h5", "14.h5", "15.h5", "16.h5", "17.h5",
    "18.h5", "19.h5"
  ]
}
~~~~~~~~

# pyproject.toml

~~~~~~~~toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "asgcn-reconstruction"
version = "0.2.0"
description = "ASGCN paper-core event-to-frame reconstruction for EventHDR and EventAid-R"
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
~~~~~~~~

# README.md

~~~~~~~~markdown
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
~~~~~~~~

# requirements.txt

~~~~~~~~text
-c constraints/py312.txt
-e .[dev]
~~~~~~~~

# scripts/calibrate.sh

~~~~~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/calibrate.sh [CONFIG [ANN_CHECKPOINT [SNN_CHECKPOINT]]]" \
    "" \
    "Environment:" \
    "  CALIBRATION_SAMPLES=all|N   Default: all EventHDR calibration samples" \
    "  OVERWRITE_CALIBRATION=0|1  Default: 0; protect an existing output" \
    "  VALIDATE_DATASET=0|1       Default: 1" \
    "  INSPECT_VALIDATE_ALL=0|1   Default: 0" \
    "  INSPECT_SAMPLES=N          Default: 1" \
    "  REQUIRE_CUDA=0|1           Default: 1" \
    "  PYTHON_BIN=PATH            Default: <repo>/.venv/bin/python"
  exit 0
fi

CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_train.json}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}}"
OUTPUT_PATH="${3:-${OUTPUT_PATH:-runs/eventhdr_asgcn/best_snn.pt}}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: calibration config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: ANN checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
for flag_name in REQUIRE_CUDA VALIDATE_DATASET INSPECT_VALIDATE_ALL OVERWRITE_CALIBRATION; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - "${CHECKPOINT_PATH}" "${OUTPUT_PATH}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
if source == output:
    raise SystemExit("ANN input and calibrated SNN output must be different files")
PY

if [[ -e "${OUTPUT_PATH}" || -L "${OUTPUT_PATH}" ]]; then
  if [[ "${OVERWRITE_CALIBRATION}" != "1" ]]; then
    echo "ERROR: calibrated output already exists: ${OUTPUT_PATH}" >&2
    echo "Set OVERWRITE_CALIBRATION=1 only when replacing it is intentional." >&2
    exit 1
  fi
  if [[ -d "${OUTPUT_PATH}" ]]; then
    echo "ERROR: calibrated output path is a directory: ${OUTPUT_PATH}" >&2
    exit 1
  fi
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
  INSPECT_ARGS=(
    --config "${CONFIG_PATH}"
    --samples "${INSPECT_SAMPLES}"
  )
  if [[ "${INSPECT_VALIDATE_ALL}" == "1" ]]; then
    INSPECT_ARGS+=(--validate-all)
  fi
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect "${INSPECT_ARGS[@]}"
fi

if [[ "${CALIBRATION_SAMPLES}" != "all" && ! "${CALIBRATION_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CALIBRATION_SAMPLES must be 'all' or a positive integer" >&2
  exit 2
fi

echo "Calibrating ${CHECKPOINT_PATH} with ${CALIBRATION_SAMPLES} EventHDR samples"
CALIBRATE_ARGS=(
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --output "${OUTPUT_PATH}"
  --samples "${CALIBRATION_SAMPLES}"
)
if [[ "${OVERWRITE_CALIBRATION}" == "1" ]]; then
  CALIBRATE_ARGS+=(--overwrite)
fi
exec "${PYTHON_BIN}" -m asgcn_recon.cli calibrate "${CALIBRATE_ARGS[@]}"
~~~~~~~~

# scripts/check_env.py

~~~~~~~~python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import sys
from pathlib import Path

import torch

from asgcn_recon.data import load_eventhdr_split_manifest

_OFFICIAL_EVENTHDR_TRAIN = {f"{index}.h5" for index in range(1, 52)}
_OFFICIAL_EVENTHDR_EVAL = {f"{index}.h5" for index in range(1, 20)}


def _eventhdr_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([*root.rglob("*.h5"), *root.rglob("*.hdf5")])


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def _exact_coverage_problem(
    label: str,
    present: set[str],
    expected: set[str],
) -> str | None:
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if not missing and not extra:
        return None
    details = []
    if missing:
        details.append("missing=" + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if extra:
        details.append("extra=" + ", ".join(extra[:8]) + (" ..." if len(extra) > 8 else ""))
    return (
        f"{label} must contain exactly {len(expected)} official files "
        f"({'; '.join(details)})"
    )


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        versions[name.strip().lower().replace("_", "-")] = version.strip()
    return versions


def _check_lock(path: Path) -> dict[str, dict[str, str | None]]:
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected in _locked_versions(path).items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        actual_public = actual.split("+", maxsplit=1)[0] if actual else None
        expected_public = expected.split("+", maxsplit=1)[0]
        if actual_public != expected_public:
            mismatches[name] = {"expected": expected, "actual": actual}
    return mismatches


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASGCN server readiness")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-eventhdr-train", action="store_true")
    parser.add_argument("--require-eventhdr-eval", action="store_true")
    parser.add_argument("--require-eventaid-all", action="store_true")
    parser.add_argument("--require-full-data", action="store_true")
    parser.add_argument("--lock", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or project_root / "data").resolve()
    runs_root = (args.runs_root or project_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    gpu_memory_gib = [
        round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2)
        for index in range(torch.cuda.device_count())
    ]
    lock_path = args.lock.resolve() if args.lock else None
    lock_mismatches = _check_lock(lock_path) if lock_path and lock_path.is_file() else None
    lock_python_match = None
    if lock_path:
        match = re.fullmatch(r"py(\d)(\d+)", lock_path.stem)
        if match:
            expected_python = f"{match.group(1)}.{match.group(2)}"
            lock_python_match = platform.python_version().startswith(f"{expected_python}.")
    train_files = _eventhdr_files(data_root / "EventHDR" / "train")
    eval_files = _eventhdr_files(data_root / "EventHDR" / "eval")
    train_root = data_root / "EventHDR" / "train"
    train_present = {path.relative_to(train_root).as_posix() for path in train_files}
    eval_root = data_root / "EventHDR" / "eval"
    eval_present = {path.relative_to(eval_root).as_posix() for path in eval_files}
    data_disk = shutil.disk_usage(data_root if data_root.exists() else project_root)
    runs_disk = shutil.disk_usage(runs_root)
    libc_name, libc_version = platform.libc_ver()
    report = {
        "project_root": str(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "libc": {"name": libc_name or None, "version": libc_version or None},
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_devices": devices,
        "gpu_memory_gib": gpu_memory_gib,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_root": str(data_root),
        "eventhdr_train_h5": len(train_files),
        "eventhdr_eval_h5": len(eval_files),
        "eventaid_r_zip": _count_files(data_root / "EventAid-R", "R-*.zip"),
        "runs_root": str(runs_root),
        "runs_writable": os.access(runs_root, os.W_OK),
        "data_disk_free_gib": round(data_disk.free / (1024**3), 2),
        "runs_disk_free_gib": round(runs_disk.free / (1024**3), 2),
        "lock_file": str(lock_path) if lock_path else None,
        "constraint_versions_match": (not lock_mismatches if lock_mismatches is not None else None),
        "constraint_python_match": lock_python_match,
        "lock_mismatches": lock_mismatches,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    problems: list[str] = []
    if not report["runs_writable"]:
        problems.append(f"Run directory is not writable: {runs_root}")
    if args.require_cuda and not cuda_available:
        problems.append("CUDA was required but torch.cuda.is_available() is false")
    if lock_path and not lock_path.is_file():
        problems.append(f"Dependency lock does not exist: {lock_path}")
    elif lock_mismatches:
        problems.append(f"Installed packages differ from dependency lock: {lock_path}")
    if lock_python_match is False:
        problems.append(f"Python version does not match dependency profile: {lock_path}")
    locked_torch = (
        _locked_versions(lock_path).get("torch") if lock_path and lock_path.is_file() else None
    )
    if (
        platform.system() == "Linux"
        and libc_name.lower() == "glibc"
        and locked_torch == "2.13.0"
        and _version_tuple(libc_version) < (2, 28)
    ):
        problems.append(
            f"torch 2.13.0 wheel profile requires glibc>=2.28; found {libc_version}. "
            "Use a newer cluster container/module instead of building from source blindly"
        )
    require_eventhdr_train = args.require_full_data or args.require_eventhdr_train
    require_eventhdr_eval = args.require_full_data or args.require_eventhdr_eval
    require_eventaid_all = args.require_full_data or args.require_eventaid_all
    if require_eventhdr_train:
        problem = _exact_coverage_problem(
            "eventhdr_train_h5", train_present, _OFFICIAL_EVENTHDR_TRAIN
        )
        if problem:
            problems.append(problem)
    if require_eventhdr_eval:
        problem = _exact_coverage_problem(
            "eventhdr_eval_h5", eval_present, _OFFICIAL_EVENTHDR_EVAL
        )
        if problem:
            problems.append(problem)
    if require_eventhdr_train:
        manifest_path = project_root / "manifests" / "eventhdr_split.json"
        if not manifest_path.is_file():
            problems.append(f"Official EventHDR split manifest is missing: {manifest_path}")
        else:
            manifest = load_eventhdr_split_manifest(manifest_path)
            if manifest.get("status") != "final" or manifest.get("split_schema") != (
                "official_separate_roots_v1"
            ):
                problems.append(
                    "EventHDR training requires a final official_separate_roots_v1 manifest"
                )
            manifest_train = set(manifest.get("train_files", []))
            manifest_eval = set(manifest.get("val_files", []))
            if manifest_train != _OFFICIAL_EVENTHDR_TRAIN:
                problems.append(
                    "EventHDR split manifest train root must declare exactly 1.h5 through 51.h5"
                )
            if manifest_eval != _OFFICIAL_EVENTHDR_EVAL:
                problems.append(
                    "EventHDR split manifest eval root must declare exactly 1.h5 through 19.h5"
                )
    if require_eventaid_all:
        aid_manifest_path = project_root / "manifests" / "eventaid_r.json"
        if aid_manifest_path.is_file():
            aid_manifest = json.loads(aid_manifest_path.read_text(encoding="utf-8"))
            aid_root = data_root / "EventAid-R"
            aid_present = {path.name for path in aid_root.glob("R-*.zip")}
            aid_required = {
                f"{item['scene']}.zip"
                for item in aid_manifest.get("files", [])
                if isinstance(item, dict) and item.get("scene")
            }
            problem = _exact_coverage_problem("eventaid_r_zip", aid_present, aid_required)
            if problem:
                problems.append(problem)
        else:
            problems.append(f"EventAid-R manifest is missing: {aid_manifest_path}")
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
~~~~~~~~

# scripts/eval.sh

~~~~~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_ann.json}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}}"
INFERENCE_MODE="${INFERENCE_MODE:-ann}"
SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
SNN_DYNAMICS="${SNN_DYNAMICS:-}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"

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
if [[ "${VALIDATE_DATASET}" != "0" && "${VALIDATE_DATASET}" != "1" ]]; then
  echo "ERROR: VALIDATE_DATASET must be 0 or 1" >&2
  exit 2
fi
if [[ "${INSPECT_VALIDATE_ALL}" != "0" && "${INSPECT_VALIDATE_ALL}" != "1" ]]; then
  echo "ERROR: INSPECT_VALIDATE_ALL must be 0 or 1" >&2
  exit 2
fi
if [[ "${INFERENCE_MODE}" != "ann" && "${INFERENCE_MODE}" != "snn" ]]; then
  echo "ERROR: INFERENCE_MODE must be ann or snn" >&2
  exit 2
fi
if [[ -n "${SNN_DYNAMICS}" ]]; then
  if [[ "${INFERENCE_MODE}" != "snn" ]]; then
    echo "ERROR: SNN_DYNAMICS is only valid when INFERENCE_MODE=snn" >&2
    exit 2
  fi
  if [[ "${SNN_DYNAMICS}" != "literal_eq15" && "${SNN_DYNAMICS}" != "standard_if" ]]; then
    echo "ERROR: SNN_DYNAMICS must be literal_eq15 or standard_if" >&2
    exit 2
  fi
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

DYNAMICS_ARGS=()
if [[ -n "${SNN_DYNAMICS}" ]]; then
  DYNAMICS_ARGS=(--snn-dynamics "${SNN_DYNAMICS}")
fi

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  INSPECT_ARGS=(
    --config "${CONFIG_PATH}"
    --samples "${INSPECT_SAMPLES}"
  )
  if [[ "${INSPECT_VALIDATE_ALL}" == "1" ]]; then
    INSPECT_ARGS+=(--validate-all)
  fi
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect "${INSPECT_ARGS[@]}"
fi

echo "Evaluating ${CHECKPOINT_PATH} on ${CONFIG_PATH} (${INFERENCE_MODE})"
"${PYTHON_BIN}" -m asgcn_recon.cli evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --inference-mode "${INFERENCE_MODE}" \
  --simulation-steps "${SIMULATION_STEPS}" \
  "${DYNAMICS_ARGS[@]}"

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  echo "Running latency benchmark"
  "${PYTHON_BIN}" -m asgcn_recon.cli benchmark \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --warmup "${BENCHMARK_WARMUP}" \
    --steps "${BENCHMARK_STEPS}" \
    --inference-mode "${INFERENCE_MODE}" \
    --simulation-steps "${SIMULATION_STEPS}" \
    "${DYNAMICS_ARGS[@]}"
fi
~~~~~~~~

# scripts/full.sh

~~~~~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/full.sh" \
    "" \
    "Runs, in order:" \
    "  full environment/data check and complete EventHDR/EventAid-R validation" \
    "  EventHDR ANN train (or RESUME_CHECKPOINT resume)" \
    "  all-sample EventHDR ANN-to-SNN calibration" \
    "  EventHDR and EventAid-R ANN evaluation+benchmark" \
    "  literal_eq15 and standard_if SNN evaluation+benchmark at T=4,8,16,32" \
    "" \
    "Important environment:" \
    "  RESUME_CHECKPOINT=PATH" \
    "  TRAIN_CONFIG / HDR_ANN_CONFIG / HDR_SNN_CONFIG" \
    "  AID_ANN_CONFIG / AID_SNN_CONFIG" \
    "  ANN_CHECKPOINT / SNN_CHECKPOINT" \
    "  PYTHON_BIN=PATH                         Default: <repo>/.venv/bin/python" \
    "  REQUIRE_CUDA=0|1                       Default: 1" \
    "  CALIBRATION_SAMPLES=all|N              Default: all" \
    "  SIMULATION_STEPS_LIST='4 8 16 32'" \
    "  BENCHMARK_WARMUP=N / BENCHMARK_STEPS=N" \
    "  OVERWRITE_CALIBRATION=0|1              Default: 0" \
    "  DRY_RUN=0|1                            Print the complete command schedule"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints/py312.txt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/hdr_train.json}"
HDR_ANN_CONFIG="${HDR_ANN_CONFIG:-configs/hdr_ann.json}"
HDR_SNN_CONFIG="${HDR_SNN_CONFIG:-configs/hdr_snn.json}"
AID_ANN_CONFIG="${AID_ANN_CONFIG:-configs/aid_ann.json}"
AID_SNN_CONFIG="${AID_SNN_CONFIG:-configs/aid_snn.json}"
ANN_CHECKPOINT="${ANN_CHECKPOINT:-runs/eventhdr_asgcn/best.pt}"
SNN_CHECKPOINT="${SNN_CHECKPOINT:-runs/eventhdr_asgcn/best_snn.pt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-2}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"
for flag_name in REQUIRE_CUDA OVERWRITE_CALIBRATION DRY_RUN; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${DRY_RUN}" != "1" && ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
for required_path in \
  "${CONSTRAINTS_FILE}" \
  "${TRAIN_CONFIG}" \
  "${HDR_ANN_CONFIG}" \
  "${HDR_SNN_CONFIG}" \
  "${AID_ANN_CONFIG}" \
  "${AID_SNN_CONFIG}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: required full-run file not found: ${required_path}" >&2
    exit 1
  fi
done
if [[ -n "${RESUME_CHECKPOINT}" && "${DRY_RUN}" != "1" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

read -r -a SIMULATION_STEPS <<< "${SIMULATION_STEPS_LIST}"
if [[ "${#SIMULATION_STEPS[@]}" -eq 0 ]]; then
  echo "ERROR: SIMULATION_STEPS_LIST must contain at least one positive integer" >&2
  exit 2
fi
for step in "${SIMULATION_STEPS[@]}"; do
  if [[ ! "${step}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid simulation step '${step}' in SIMULATION_STEPS_LIST" >&2
    exit 2
  fi
done

run_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

run_evaluation() {
  local config_path="$1"
  local checkpoint_path="$2"
  local mode="$3"
  local simulation_steps="$4"
  local dynamics="$5"
  run_cmd env \
    REQUIRE_CUDA="${REQUIRE_CUDA}" \
    VALIDATE_DATASET=0 \
    RUN_BENCHMARK=1 \
    BENCHMARK_WARMUP="${BENCHMARK_WARMUP}" \
    BENCHMARK_STEPS="${BENCHMARK_STEPS}" \
    INFERENCE_MODE="${mode}" \
    SIMULATION_STEPS="${simulation_steps}" \
    SNN_DYNAMICS="${dynamics}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${PROJECT_ROOT}/scripts/eval.sh" "${config_path}" "${checkpoint_path}"
}

echo "[1/5] Full environment and dataset inventory"
CHECK_ARGS=("${PYTHON_BIN}" scripts/check_env.py --require-full-data --lock "${CONSTRAINTS_FILE}")
if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  CHECK_ARGS+=(--require-cuda)
fi
run_cmd "${CHECK_ARGS[@]}"

echo "[2/5] Decode and validate every EventHDR train/eval and EventAid-R sample"
# TRAIN_CONFIG inspection covers both EventHDR roots. Inspecting HDR_ANN_CONFIG
# here would decode the same 19 eval files a second time without adding coverage.
for config_path in "${TRAIN_CONFIG}" "${AID_ANN_CONFIG}"; do
  run_cmd "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${config_path}" \
    --samples "${INSPECT_SAMPLES}" \
    --validate-all
done

echo "[3/5] EventHDR ANN training"
run_cmd env \
  REQUIRE_CUDA="${REQUIRE_CUDA}" \
  VALIDATE_DATASET=0 \
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/train.sh" "${TRAIN_CONFIG}"
if [[ "${DRY_RUN}" != "1" && ! -f "${ANN_CHECKPOINT}" ]]; then
  echo "ERROR: training completed without ANN checkpoint: ${ANN_CHECKPOINT}" >&2
  exit 1
fi

echo "[4/5] Full EventHDR ANN-to-SNN calibration"
run_cmd env \
  REQUIRE_CUDA="${REQUIRE_CUDA}" \
  VALIDATE_DATASET=0 \
  CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
  OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/calibrate.sh" \
    "${TRAIN_CONFIG}" "${ANN_CHECKPOINT}" "${SNN_CHECKPOINT}"
if [[ "${DRY_RUN}" != "1" && ! -f "${SNN_CHECKPOINT}" ]]; then
  echo "ERROR: calibration completed without SNN checkpoint: ${SNN_CHECKPOINT}" >&2
  exit 1
fi

echo "[5/5] Full EventHDR and EventAid-R evaluation and compute benchmark matrix"
for dataset_spec in \
  "${HDR_ANN_CONFIG}|${HDR_SNN_CONFIG}" \
  "${AID_ANN_CONFIG}|${AID_SNN_CONFIG}"; do
  IFS='|' read -r ann_config snn_config <<< "${dataset_spec}"
  run_evaluation "${ann_config}" "${ANN_CHECKPOINT}" ann 16 ""
  for dynamics in literal_eq15 standard_if; do
    for simulation_steps in "${SIMULATION_STEPS[@]}"; do
      run_evaluation \
        "${snn_config}" \
        "${SNN_CHECKPOINT}" \
        snn \
        "${simulation_steps}" \
        "${dynamics}"
    done
  done
done

echo "Full experiment matrix completed."
echo "ANN checkpoint: ${ANN_CHECKPOINT}"
echo "SNN checkpoint: ${SNN_CHECKPOINT}"
~~~~~~~~

# scripts/get_aid.ps1

~~~~~~~~powershell
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
~~~~~~~~

# scripts/get_aid.sh

~~~~~~~~bash
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

With no SCENE, the complete 14-scene release is downloaded. ZIP files stay
compressed because the loader reads them directly.

Options:
  -d, --destination DIR  Download directory (default: data/EventAid-R)
  --all                  Explicitly download all 14 scenes (~24.68 GB)
  -h, --help             Show this help

Examples:
  ./scripts/get_aid.sh                  # all 14 scenes
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
  DOWNLOAD_ALL=1
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
~~~~~~~~

# scripts/get_hdr.py

~~~~~~~~python
#!/usr/bin/env python3
"""Import the complete official EventHDR release without pretending to download it.

The public OneDrive folder currently rejects unattended curl requests.  This
tool therefore accepts either a browser-downloaded ZIP, an extracted source
directory, or a shared-storage directory and verifies the official file set
before making it visible under ``data/EventHDR``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import BinaryIO

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MAX_DATASET_BYTES = 100_000_000_000
EXPECTED: dict[str, tuple[str, ...]] = {
    "train": tuple(f"{index}.h5" for index in range(1, 52)),
    "eval": tuple(f"{index}.h5" for index in range(1, 20)),
}


class ImportError(RuntimeError):
    """Raised when an EventHDR source cannot be imported safely."""


def _format_names(names: Iterable[str], limit: int = 8) -> str:
    ordered = sorted(names, key=lambda name: (len(name), name))
    preview = ", ".join(ordered[:limit])
    return preview + (" ..." if len(ordered) > limit else "")


def _validate_exact_names(names: Iterable[str], split: str, source: str) -> None:
    present = set(names)
    expected = set(EXPECTED[split])
    missing = expected - present
    extra = present - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + _format_names(missing))
        if extra:
            details.append("extra=" + _format_names(extra))
        raise ImportError(
            f"{source} does not contain the exact official EventHDR {split} file set "
            f"({'; '.join(details)})"
        )


def _validate_magic(stream: BinaryIO, source: str) -> None:
    magic = stream.read(len(HDF5_MAGIC))
    if magic != HDF5_MAGIC:
        raise ImportError(f"Not an HDF5 file: {source}")


def _h5_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
            if path.name in files:
                raise ImportError(f"Duplicate HDF5 filename under {directory}: {path.name}")
            files[path.name] = path
    nested = [
        path
        for path in directory.rglob("*")
        if path.parent != directory
        and path.is_file()
        and path.suffix.lower() in {".h5", ".hdf5"}
    ]
    if nested:
        raise ImportError(
            f"Nested HDF5 files are not allowed under {directory}: "
            + _format_names(path.relative_to(directory).as_posix() for path in nested)
        )
    return files


def validate_split_dir(directory: Path, split: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ImportError(f"EventHDR {split} directory does not exist: {directory}")
    files = _h5_files(directory)
    _validate_exact_names(files, split, str(directory))
    total_bytes = 0
    for name in EXPECTED[split]:
        path = files[name]
        total_bytes += path.stat().st_size
        with path.open("rb") as stream:
            _validate_magic(stream, str(path))
    if total_bytes >= MAX_DATASET_BYTES:
        raise ImportError(
            f"EventHDR {split} source is {total_bytes} bytes; the accepted dataset must be "
            "smaller than 100 GB"
        )
    return files


def _validate_combined_size(files_by_split: dict[str, dict[str, Path]], source: str) -> None:
    total_bytes = sum(
        path.stat().st_size
        for split_files in files_by_split.values()
        for path in split_files.values()
    )
    if total_bytes >= MAX_DATASET_BYTES:
        raise ImportError(
            f"EventHDR source is {total_bytes} bytes; the complete accepted dataset must be "
            "smaller than 100 GB: "
            + source
        )


def _candidate_split_dirs(source: Path, split: str) -> list[Path]:
    candidates = (source, source / split, source / "EventHDR" / split)
    result: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir() and any(
            child.is_file() and child.suffix.lower() in {".h5", ".hdf5"}
            for child in candidate.iterdir()
        ):
            resolved = candidate.resolve()
            if resolved not in result:
                result.append(resolved)
    return result


def locate_source(source: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ImportError(f"Source directory does not exist: {source}")
    located: dict[str, Path] = {}
    for split in splits:
        candidates = _candidate_split_dirs(source, split)
        valid: list[Path] = []
        failures: list[str] = []
        for candidate in candidates:
            try:
                validate_split_dir(candidate, split)
            except ImportError as error:
                failures.append(str(error))
            else:
                valid.append(candidate)
        if len(valid) != 1:
            if len(valid) > 1:
                raise ImportError(
                    f"Ambiguous EventHDR {split} source directories: "
                    + ", ".join(str(path) for path in valid)
                )
            detail = "; ".join(failures) if failures else "no candidate directory found"
            raise ImportError(f"Could not locate exact EventHDR {split} data: {detail}")
        located[split] = valid[0]
    _validate_combined_size(
        {split: validate_split_dir(path, split) for split, path in located.items()},
        str(source),
    )
    return located


def _destination_extras(directory: Path, split: str) -> set[str]:
    if not directory.exists() or directory.is_symlink():
        return set()
    if not directory.is_dir():
        raise ImportError(f"Destination is not a directory: {directory}")
    return set(_h5_files(directory)) - set(EXPECTED[split])


def _prepare_copy_destination(destination: Path, splits: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ImportError(
            f"Copy mode refuses a symlinked EventHDR root; use --check or --link: {destination}"
        )
    for split in splits:
        split_dir = destination / split
        if split_dir.is_symlink():
            raise ImportError(f"Copy mode refuses a symlinked destination: {split_dir}")
        split_dir.mkdir(parents=True, exist_ok=True)
        extras = _destination_extras(split_dir, split)
        if extras:
            raise ImportError(
                f"Destination {split_dir} contains unexpected HDF5 files: "
                + _format_names(extras)
            )


def _copy_one(source: Path, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ImportError(
                f"Refusing to overwrite a different existing file: {target} "
                f"({target.stat().st_size} != {source.stat().st_size} bytes)"
            )
        with target.open("rb") as stream:
            _validate_magic(stream, str(target))
        return "kept"

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            _validate_magic(stream, str(temporary))
        os.replace(temporary, target)
        temporary = None
        return "copied"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_source(source_dirs: dict[str, Path], destination: Path) -> dict[str, int]:
    splits = tuple(source_dirs)
    source_files = {
        split: validate_split_dir(source_dirs[split], split) for split in splits
    }
    _prepare_copy_destination(destination, splits)
    counts = {"copied": 0, "kept": 0}
    for split in splits:
        for name in EXPECTED[split]:
            outcome = _copy_one(source_files[split][name], destination / split / name)
            counts[outcome] += 1
        validate_split_dir(destination / split, split)
    return counts


def _safe_zip_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ImportError(f"Unsafe archive member path: {name!r}")
    return path.parts


def locate_archive_members(
    archive: zipfile.ZipFile, splits: tuple[str, ...]
) -> dict[str, dict[str, zipfile.ZipInfo]]:
    selected: dict[str, dict[str, zipfile.ZipInfo]] = {split: {} for split in splits}
    split_set = set(splits)
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = _safe_zip_parts(info.filename)
        name = parts[-1]
        if Path(name).suffix.lower() not in {".h5", ".hdf5"}:
            continue

        owner: str | None = None
        if len(splits) == 1 and len(parts) == 1:
            owner = splits[0]
        elif len(parts) >= 2 and parts[-2].lower() in split_set:
            owner = parts[-2].lower()
        if owner is None:
            raise ImportError(
                f"Cannot assign archive HDF5 member to train/eval: {info.filename}"
            )
        if name in selected[owner]:
            raise ImportError(
                f"Archive contains duplicate EventHDR {owner} filename {name}: "
                f"{selected[owner][name].filename}, {info.filename}"
            )
        selected[owner][name] = info

    for split in splits:
        _validate_exact_names(selected[split], split, "archive")
        total_bytes = sum(info.file_size for info in selected[split].values())
        if total_bytes >= MAX_DATASET_BYTES:
            raise ImportError(
                f"EventHDR {split} archive content is {total_bytes} bytes; the accepted "
                "dataset must be smaller than 100 GB"
            )
        for name in EXPECTED[split]:
            with archive.open(selected[split][name], "r") as stream:
                _validate_magic(stream, f"archive::{selected[split][name].filename}")
    combined_bytes = sum(
        info.file_size for split_members in selected.values() for info in split_members.values()
    )
    if combined_bytes >= MAX_DATASET_BYTES:
        raise ImportError(
            f"EventHDR archive content is {combined_bytes} bytes; the complete accepted "
            "dataset must be smaller than 100 GB"
        )
    return selected


def _copy_archive_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path
) -> str:
    if target.exists():
        if target.stat().st_size != info.file_size:
            raise ImportError(
                f"Refusing to overwrite a different existing file: {target} "
                f"({target.stat().st_size} != {info.file_size} bytes)"
            )
        with target.open("rb") as stream:
            _validate_magic(stream, str(target))
        return "kept"

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            with archive.open(info, "r") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        with temporary.open("rb") as stream:
            _validate_magic(stream, str(temporary))
        os.replace(temporary, target)
        temporary = None
        return "copied"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_archive(
    archive_path: Path, destination: Path, splits: tuple[str, ...]
) -> dict[str, int]:
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise ImportError(f"Archive does not exist: {archive_path}")
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ImportError(f"Invalid ZIP archive {archive_path}: {error}") from error
    with archive:
        members = locate_archive_members(archive, splits)
        _prepare_copy_destination(destination, splits)
        counts = {"copied": 0, "kept": 0}
        for split in splits:
            for name in EXPECTED[split]:
                outcome = _copy_archive_member(
                    archive, members[split][name], destination / split / name
                )
                counts[outcome] += 1
            validate_split_dir(destination / split, split)
        return counts


def link_source(source_dirs: dict[str, Path], destination: Path) -> dict[str, int]:
    for split, source_dir in source_dirs.items():
        validate_split_dir(source_dir, split)
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    kept = 0
    for split, source_dir in source_dirs.items():
        target = destination / split
        if target.is_symlink():
            if target.resolve() != source_dir.resolve():
                raise ImportError(f"Destination symlink points elsewhere: {target}")
            kept += 1
            continue
        if target.exists():
            if not target.is_dir() or any(target.iterdir()):
                raise ImportError(
                    f"Refusing to replace a non-empty destination; move it first: {target}"
                )
            target.rmdir()
        target.symlink_to(source_dir.resolve(), target_is_directory=True)
        linked += 1
    for split in source_dirs:
        validate_split_dir(destination / split, split)
    return {"linked": linked, "kept": kept}


def check_destination(destination: Path, splits: tuple[str, ...]) -> None:
    files_by_split = {
        split: validate_split_dir(destination / split, split) for split in splits
    }
    _validate_combined_size(files_by_split, str(destination))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import/check the complete official EventHDR train (1-51) and eval (1-19) "
            "HDF5 release. This tool does not claim to bypass OneDrive's browser download."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="extracted EventHDR/train/eval source")
    mode.add_argument("--archive", type=Path, help="browser-downloaded ZIP archive")
    mode.add_argument("--check", action="store_true", help="check files already in destination")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/EventHDR"),
        help="logical EventHDR destination (default: data/EventHDR)",
    )
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED),
        help="import/check only one separately downloaded train or eval folder",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="symlink an extracted/shared source instead of copying it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.destination.expanduser().resolve()
    splits = (args.split,) if args.split else tuple(EXPECTED)
    try:
        if args.link and args.source is None:
            raise ImportError("--link requires --source")
        if args.check:
            check_destination(destination, splits)
            print(
                f"EventHDR check passed: {destination} "
                + ", ".join(f"{split}={len(EXPECTED[split])}" for split in splits)
            )
            return 0
        if args.source is not None:
            source_dirs = locate_source(args.source, splits)
            counts = (
                link_source(source_dirs, destination)
                if args.link
                else copy_source(source_dirs, destination)
            )
        else:
            counts = copy_archive(args.archive, destination, splits)
        check_destination(destination, splits)
        print(
            f"EventHDR import passed: {destination} "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
        return 0
    except (ImportError, OSError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
~~~~~~~~

# scripts/get_hdr.sh

~~~~~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd -- "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/get_hdr.py" "$@"
~~~~~~~~

# scripts/setup.sh

~~~~~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

# Clone 후 한 번 실행하는 Linux 서버 설치 스크립트.
# https://pytorch.org/get-started/locally/ 에서 서버 드라이버에 맞는 wheel을 고른 뒤:
#   TORCH_INDEX_URL=<official-wheel-index> ./scripts/setup.sh
# 재현용으로 버전을 고정할 때:
#   TORCH_VERSION=<version> TORCH_INDEX_URL=<official-wheel-index> \
#     CONSTRAINTS_FILE=constraints/py312.txt PROJECT_EXTRAS=dev ./scripts/setup.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  OVERRIDE_NAMES=(
    PYTHON_BIN VENV_DIR TORCH_VERSION TORCH_INDEX_URL PROJECT_EXTRAS
    REQUIRE_CUDA CONSTRAINTS_FILE EXPECTED_PYTHON_MINOR PIP_EXTRA_ARGS
  )
  declare -A CALLER_OVERRIDES=()
  for variable in "${OVERRIDE_NAMES[@]}"; do
    if [[ -v "${variable}" ]]; then
      CALLER_OVERRIDES["${variable}"]="${!variable}"
    fi
  done
  echo "Loading installer settings: ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  for variable in "${!CALLER_OVERRIDES[@]}"; do
    printf -v "${variable}" '%s' "${CALLER_OVERRIDES[${variable}]}"
    export "${variable}"
  done
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
TORCH_VERSION="${TORCH_VERSION:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PROJECT_EXTRAS="${PROJECT_EXTRAS:-}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-}"
EXPECTED_PYTHON_MINOR="${EXPECTED_PYTHON_MINOR:-}"

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${PROJECT_ROOT}/${VENV_DIR}"
fi

CONSTRAINT_ARGS=()
if [[ -n "${CONSTRAINTS_FILE}" ]]; then
  if [[ "${CONSTRAINTS_FILE}" != /* ]]; then
    CONSTRAINTS_FILE="${PROJECT_ROOT}/${CONSTRAINTS_FILE}"
  fi
  if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
    echo "ERROR: constraints file not found: ${CONSTRAINTS_FILE}" >&2
    exit 1
  fi
  CONSTRAINT_ARGS=(-c "${CONSTRAINTS_FILE}")
  echo "Using dependency constraints: ${CONSTRAINTS_FILE}"
  if [[ -z "${EXPECTED_PYTHON_MINOR}" ]] \
    && [[ "$(basename -- "${CONSTRAINTS_FILE}")" =~ ^py([0-9])([0-9]+)\.txt$ ]]; then
    EXPECTED_PYTHON_MINOR="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  fi
fi

# PIP_EXTRA_ARGS is intentionally optional. It is split on spaces, so paths with
# spaces should instead be configured through pip.conf. Parse it before the first
# network operation so private mirrors/proxies apply to bootstrap packages too.
PIP_ARGS=()
if [[ -n "${PIP_EXTRA_ARGS:-}" ]]; then
  read -r -a PIP_ARGS <<<"${PIP_EXTRA_ARGS}"
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

"${PYTHON_BIN}" - "${TORCH_VERSION}" "${CONSTRAINTS_FILE}" <<'PY'
import platform
import re
import sys
from pathlib import Path

requested_torch = sys.argv[1]
constraint_path = Path(sys.argv[2]) if sys.argv[2] else None
if not requested_torch and constraint_path is not None:
    for raw_line in constraint_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*torch==([^\s#]+)\s*", raw_line)
        if match:
            requested_torch = match.group(1).split("+", maxsplit=1)[0]
            break

if platform.system() == "Linux" and requested_torch == "2.13.0":
    libc_name, libc_version = platform.libc_ver()
    numbers = tuple(int(value) for value in re.findall(r"\d+", libc_version))
    if libc_name.lower() != "glibc" or numbers < (2, 28):
        found = f"{libc_name or 'unknown'} {libc_version or 'unknown'}"
        raise SystemExit(
            "The locked torch 2.13.0 wheel profile requires Linux glibc>=2.28; "
            f"found {found}. Use a newer cluster container/module."
        )
    print(f"glibc preflight: {libc_version} (minimum 2.28)")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" - "${EXPECTED_PYTHON_MINOR}" <<'PY'
import sys

expected = sys.argv[1]
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if sys.version_info < (3, 10):
    raise SystemExit(f"Virtual environment requires Python 3.10+; found {actual}")
if expected and actual != expected:
    raise SystemExit(
        f"Dependency profile requires Python {expected}, but the virtual environment "
        f"uses Python {actual}. Remove VENV_DIR and recreate it with the matching PYTHON_BIN."
    )
print(f"Virtual environment Python: {actual}")
PY
"${VENV_PYTHON}" -m pip install \
  "${PIP_ARGS[@]}" "${CONSTRAINT_ARGS[@]}" --upgrade pip setuptools wheel

TORCH_SPEC="torch"
if [[ -n "${TORCH_VERSION}" ]]; then
  TORCH_SPEC="torch==${TORCH_VERSION}"
fi

# Installing torch first preserves an explicitly chosen CUDA wheel when the
# editable project (which declares torch>=2.3) is installed below.
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  "${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" "${CONSTRAINT_ARGS[@]}" \
    --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
else
  "${VENV_PYTHON}" -m pip install \
    "${PIP_ARGS[@]}" "${CONSTRAINT_ARGS[@]}" "${TORCH_SPEC}"
fi

INSTALL_TARGET="${PROJECT_ROOT}"
if [[ -n "${PROJECT_EXTRAS}" ]]; then
  INSTALL_TARGET="${PROJECT_ROOT}[${PROJECT_EXTRAS}]"
fi
"${VENV_PYTHON}" -m pip install \
  "${PIP_ARGS[@]}" "${CONSTRAINT_ARGS[@]}" -e "${INSTALL_TARGET}"
"${VENV_PYTHON}" -m pip check

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
echo "Next: ./scripts/get_aid.sh --all"
echo "Then: ./scripts/get_hdr.sh --archive /path/to/EventHDR.zip"
echo "Finally: ./scripts/full.sh"
~~~~~~~~

# scripts/train.sh

~~~~~~~~bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_train.json}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
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
if [[ "${VALIDATE_DATASET}" != "0" && "${VALIDATE_DATASET}" != "1" ]]; then
  echo "ERROR: VALIDATE_DATASET must be 0 or 1" >&2
  exit 2
fi
if [[ "${INSPECT_VALIDATE_ALL}" != "0" && "${INSPECT_VALIDATE_ALL}" != "1" ]]; then
  echo "ERROR: INSPECT_VALIDATE_ALL must be 0 or 1" >&2
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
  INSPECT_ARGS=(
    --config "${CONFIG_PATH}"
    --samples "${INSPECT_SAMPLES}"
  )
  if [[ "${INSPECT_VALIDATE_ALL}" == "1" ]]; then
    INSPECT_ARGS+=(--validate-all)
  fi
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect "${INSPECT_ARGS[@]}"
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
~~~~~~~~

# server/calibrate.pbs

~~~~~~~~bash
#!/usr/bin/env bash
#PBS -N asgcn-cal
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe

# Dependency example:
#   train_id=$(qsub server/train.pbs)
#   cal_id=$(qsub -W depend=afterok:${train_id} server/calibrate.pbs)
#   qsub -W depend=afterok:${cal_id} -v CONFIG_PATH=configs/hdr_snn.json,\
# CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn server/eval.pbs

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    PROJECT_ROOT="${PBS_O_WORKDIR}"
  elif [[ -z "${PBS_JOBID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/calibrate.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run qsub from the repository root or use:" >&2
  echo "  qsub -v PROJECT_ROOT=/absolute/path/to/repo server/calibrate.pbs" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_train.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}"
export OUTPUT_PATH="${OUTPUT_PATH:-runs/eventhdr_asgcn/best_snn.pt}"
export CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
export OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NCPUS:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PBS job: ${PBS_JOBID:-interactive}"
echo "Config: ${CONFIG_PATH}"
echo "ANN checkpoint: ${CHECKPOINT_PATH}"
echo "SNN checkpoint: ${OUTPUT_PATH}"
echo "Calibration samples: ${CALIBRATION_SAMPLES}"
nvidia-smi || true

bash "${PROJECT_ROOT}/scripts/calibrate.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${OUTPUT_PATH}"
~~~~~~~~

# server/calibrate.sbatch

~~~~~~~~bash
#!/usr/bin/env bash
#SBATCH --job-name=asgcn-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Dependency example:
#   train_id=$(sbatch --parsable server/train.sbatch)
#   cal_id=$(sbatch --parsable --dependency=afterok:${train_id} server/calibrate.sbatch)
#   sbatch --dependency=afterok:${cal_id} --export=ALL,CONFIG_PATH=configs/hdr_snn.json,\
# CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn server/eval.sbatch

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/calibrate.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run sbatch from the repository root or use:" >&2
  echo "  sbatch --export=ALL,PROJECT_ROOT=/absolute/path/to/repo server/calibrate.sbatch" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_train.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}"
export OUTPUT_PATH="${OUTPUT_PATH:-runs/eventhdr_asgcn/best_snn.pt}"
export CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
export OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "ANN checkpoint: ${CHECKPOINT_PATH}"
echo "SNN checkpoint: ${OUTPUT_PATH}"
echo "Calibration samples: ${CALIBRATION_SAMPLES}"
nvidia-smi || true

srun bash "${PROJECT_ROOT}/scripts/calibrate.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${OUTPUT_PATH}"
~~~~~~~~

# server/eval.pbs

~~~~~~~~bash
#!/usr/bin/env bash
#PBS -N asgcn-eval
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=08:00:00
#PBS -j oe

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    PROJECT_ROOT="${PBS_O_WORKDIR}"
  elif [[ -z "${PBS_JOBID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/eval.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run qsub from the repository root or use:" >&2
  echo "  qsub -v PROJECT_ROOT=/absolute/path/to/repo server/eval.pbs" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_ann.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}"
export INFERENCE_MODE="${INFERENCE_MODE:-ann}"
export SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
export SNN_DYNAMICS="${SNN_DYNAMICS:-}"
export RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
export BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
export BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NCPUS:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PBS job: ${PBS_JOBID:-interactive}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
if [[ -n "${SNN_DYNAMICS}" ]]; then
  echo "SNN dynamics: ${SNN_DYNAMICS}"
fi
nvidia-smi || true

bash "${PROJECT_ROOT}/scripts/eval.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}"
~~~~~~~~

# server/eval.sbatch

~~~~~~~~bash
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
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/eval.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run sbatch from the repository root or use:" >&2
  echo "  sbatch --export=ALL,PROJECT_ROOT=/absolute/path/to/repo server/eval.sbatch" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
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
export SNN_DYNAMICS="${SNN_DYNAMICS:-}"
export RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
export BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
export BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
if [[ -n "${SNN_DYNAMICS}" ]]; then
  echo "SNN dynamics: ${SNN_DYNAMICS}"
fi
nvidia-smi || true

srun bash "${PROJECT_ROOT}/scripts/eval.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}"
~~~~~~~~

# server/train.pbs

~~~~~~~~bash
#!/usr/bin/env bash
#PBS -N asgcn-train
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    PROJECT_ROOT="${PBS_O_WORKDIR}"
  elif [[ -z "${PBS_JOBID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/train.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run qsub from the repository root or use:" >&2
  echo "  qsub -v PROJECT_ROOT=/absolute/path/to/repo server/train.pbs" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_train.json}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NCPUS:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PBS job: ${PBS_JOBID:-interactive}"
echo "Config: ${CONFIG_PATH}"
nvidia-smi || true

bash "${PROJECT_ROOT}/scripts/train.sh" "${CONFIG_PATH}"
~~~~~~~~

# server/train.sbatch

~~~~~~~~bash
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
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/train.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run sbatch from the repository root or use:" >&2
  echo "  sbatch --export=ALL,PROJECT_ROOT=/absolute/path/to/repo server/train.sbatch" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
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
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
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
~~~~~~~~

# src/asgcn_recon/__init__.py

~~~~~~~~python
"""ASGCN-style event-to-frame reconstruction."""

__version__ = "0.1.0"
~~~~~~~~

# src/asgcn_recon/cli.py

~~~~~~~~python
from __future__ import annotations

import argparse
import json
from typing import Any

from tqdm import tqdm

from .data import build_dataset
from .engine import benchmark, calibrate, evaluate, train
from .utils import experiment_base_dir, load_json, resolve_experiment_paths, resolve_path


def _inspect_one_split(dataset: Any, samples: int, validate_all: bool = False) -> dict[str, Any]:
    details = []
    preview_count = min(samples, len(dataset))
    count = len(dataset) if validate_all else preview_count
    indices = tqdm(range(count), desc="validate-data", disable=not validate_all)
    for index in indices:
        item = dataset[index]
        if index < preview_count:
            details.append(
                {
                    "sample_id": item["sample_id"],
                    "events": int(item["events"].shape[0]),
                    "target_shape": list(item["target"].shape),
                    "sensor_size": list(item["sensor_size"]),
                    "metadata": item["metadata"],
                }
            )
    result: dict[str, Any] = {
        "samples": len(dataset),
        "validated_samples": count,
        "validation_complete": count == len(dataset),
        "preview": details,
    }
    if hasattr(dataset, "scene_info"):
        result["scenes"] = dataset.scene_info
    if hasattr(dataset, "files"):
        result["files"] = len(dataset.files)
    if hasattr(dataset, "zero_event_intervals"):
        result["zero_event_intervals"] = int(dataset.zero_event_intervals)
    return result


def _calibration_sample_limit(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("samples must be a positive integer or 'all'") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("samples must be a positive integer or 'all'")
    return parsed


def inspect_dataset(
    config: dict[str, Any], samples: int = 3, validate_all: bool = False
) -> dict[str, Any]:
    if samples < 0:
        raise ValueError("inspect samples must be non-negative")
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
                split_details[split] = _inspect_one_split(dataset, samples, validate_all)
            finally:
                if hasattr(dataset, "close"):
                    dataset.close()
        result["splits"] = split_details
        result["samples"] = sum(detail["samples"] for detail in split_details.values())
        return result

    dataset = build_dataset(data_config, split="eval")
    try:
        result.update(_inspect_one_split(dataset, samples, validate_all))
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
    inspect_cmd.add_argument(
        "--validate-all",
        action="store_true",
        help="decode and validate every selected sample while keeping only the preview",
    )

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
    eval_cmd.add_argument(
        "--snn-dynamics",
        choices=["literal_eq15", "standard_if"],
        default=None,
        help="inference-only override; the checkpoint architecture remains unchanged",
    )

    bench_cmd = subparsers.add_parser("benchmark", help="benchmark compute-only latency")
    bench_cmd.add_argument("--config", required=True)
    bench_cmd.add_argument("--checkpoint", required=True)
    bench_cmd.add_argument("--warmup", type=int, default=10)
    bench_cmd.add_argument("--steps", type=int, default=100)
    bench_cmd.add_argument("--inference-mode", choices=["ann", "snn"], default="ann")
    bench_cmd.add_argument("--simulation-steps", type=int, default=16)
    bench_cmd.add_argument(
        "--snn-dynamics",
        choices=["literal_eq15", "standard_if"],
        default=None,
    )

    calibrate_cmd = subparsers.add_parser("calibrate", help="calibrate ANN-to-SNN thresholds")
    calibrate_cmd.add_argument("--config", required=True)
    calibrate_cmd.add_argument("--checkpoint", required=True)
    calibrate_cmd.add_argument("--output", required=True)
    calibrate_cmd.add_argument(
        "--samples",
        type=_calibration_sample_limit,
        default=None,
        metavar="N|all",
        help="balanced calibration sample count; default 'all' uses every training frame",
    )
    calibrate_cmd.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing calibrated output checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = resolve_path(args.config, ".")
    config = resolve_experiment_paths(load_json(config_path), config_path)
    base_dir = experiment_base_dir(config_path)
    if args.command == "inspect":
        result = inspect_dataset(config, args.samples, args.validate_all)
    elif args.command == "train":
        resume = resolve_path(args.resume, base_dir) if args.resume else None
        result = {"best_checkpoint": str(train(config, resume_from=resume))}
    elif args.command == "evaluate":
        result = evaluate(
            config,
            resolve_path(args.checkpoint, base_dir),
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
        )
    elif args.command == "benchmark":
        result = benchmark(
            config,
            resolve_path(args.checkpoint, base_dir),
            warmup=args.warmup,
            steps=args.steps,
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
        )
    elif args.command == "calibrate":
        result = {
            "calibrated_checkpoint": str(
                calibrate(
                    config,
                    resolve_path(args.checkpoint, base_dir),
                    resolve_path(args.output, base_dir),
                    samples=args.samples,
                    overwrite=args.overwrite,
                )
            )
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
~~~~~~~~

# src/asgcn_recon/data/__init__.py

~~~~~~~~python
from .eventaid_r import EventAidRZipDataset
from .eventhdr import EventHDRDataset
from .factory import build_dataset, collate_samples, load_eventhdr_split_manifest

__all__ = [
    "EventAidRZipDataset",
    "EventHDRDataset",
    "build_dataset",
    "collate_samples",
    "load_eventhdr_split_manifest",
]
~~~~~~~~

# src/asgcn_recon/data/common.py

~~~~~~~~python
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


def uniform_cap_ratio(event_count: int, max_events: int | None) -> float:
    """Return the source-to-retained ratio of the exact-size uniform cap."""
    if max_events is None or max_events <= 0 or event_count <= max_events:
        return 1.0
    return float(event_count) / float(max_events)


def stratified_subsample(events: np.ndarray, max_events: int | None) -> np.ndarray:
    """Select exactly ``max_events`` time-spread events when a cap is required.

    A ceil-stride cap has a severe boundary discontinuity: 8,193 inputs retain only
    4,097 values for an 8,192 cap.  Linspace selection keeps the requested count,
    includes both temporal endpoints, and remains deterministic.
    """
    if max_events is None or max_events <= 0 or len(events) <= max_events:
        return events.astype(np.float32, copy=False)
    indices = np.linspace(0, len(events) - 1, num=int(max_events), dtype=np.int64)
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
        image = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
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
~~~~~~~~

# src/asgcn_recon/data/eventaid_r.py

~~~~~~~~python
from __future__ import annotations

import io
import os
import re
import zipfile
import zlib
from itertools import pairwise
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
    uniform_cap_ratio,
)

_EVENT_RE = re.compile(r"(?:^|/)event/(\d+)\.txt$", re.IGNORECASE)
_GT_RE = re.compile(r"(?:^|/)gt/(\d+)_img\.png$", re.IGNORECASE)


def _validate_event_coordinates(
    events: np.ndarray, *, height: int, width: int, source: str
) -> None:
    if not len(events):
        return
    xs, ys = events[:, 0], events[:, 1]
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError(f"Invalid EventAid-R event block {source}: coordinates must be finite")
    if np.any((xs < 0) | (xs >= width)) or np.any((ys < 0) | (ys >= height)):
        raise ValueError(
            f"Invalid EventAid-R event block {source}: coordinates must lie within "
            f"x=[0,{width}), y=[0,{height})"
        )


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
                if timestamps_name is None:
                    raise ValueError(f"Invalid EventAid-R scene {path}: timestamps.txt is missing")
                try:
                    timestamps = [
                        int(value) for value in zf.read(timestamps_name).decode("utf-8").split()
                    ]
                except (UnicodeDecodeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid EventAid-R timestamps in {path}::{timestamps_name}"
                    ) from error
                if not timestamps:
                    raise ValueError(
                        f"Invalid EventAid-R timestamps in {path}::{timestamps_name}: empty file"
                    )
                if any(current <= previous for previous, current in pairwise(timestamps)):
                    raise ValueError(
                        f"Invalid EventAid-R timestamps in {path}::{timestamps_name}: "
                        "values must be strictly increasing"
                    )
                event_ids = sorted(events)
                target_ids = sorted(targets)
                if not event_ids or not target_ids:
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: event and GT files are required"
                    )
                if event_ids != list(range(event_ids[0], event_ids[-1] + 1)):
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: event IDs are not contiguous"
                    )
                if target_ids != list(range(target_ids[0], target_ids[-1] + 1)):
                    raise ValueError(f"Invalid EventAid-R scene {path}: GT IDs are not contiguous")
                paired_ids = [
                    event_id for event_id in event_ids if event_id + self.target_offset in targets
                ]
                boundary = abs(self.target_offset)
                if self.target_offset >= 0:
                    allowed_event_gaps = set(event_ids[-boundary:]) if boundary else set()
                    allowed_target_gaps = set(target_ids[:boundary]) if boundary else set()
                else:
                    allowed_event_gaps = set(event_ids[:boundary])
                    allowed_target_gaps = set(target_ids[-boundary:])
                unpaired_events = set(event_ids) - set(paired_ids)
                paired_targets = {event_id + self.target_offset for event_id in paired_ids}
                unpaired_targets = set(target_ids) - paired_targets
                if (
                    not paired_ids
                    or unpaired_events - allowed_event_gaps
                    or unpaired_targets - allowed_target_gaps
                ):
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: event/GT pairing has internal gaps"
                    )
                if paired_ids[0] < 1 or len(timestamps) <= paired_ids[-1]:
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: timestamps.txt does not cover "
                        "every paired event interval"
                    )
                scene = path.stem
                scene_info[scene] = {"shape": shape, "frames": len(targets), "events": len(events)}
                for event_id in paired_ids:
                    target_id = event_id + self.target_offset
                    samples.append(
                        {
                            "path": path,
                            "scene": scene,
                            "frame_id": event_id,
                            "event_name": events[event_id],
                            "target_name": targets[target_id],
                            "shape": shape,
                            "sequence_index": event_id,
                            "t0_us": timestamps[event_id - 1],
                            "t1_us": timestamps[event_id],
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
    def _read_events(raw: bytes, source: str = "event file") -> np.ndarray:
        if not raw.strip():
            return np.empty((0, 4), dtype=np.float32)
        try:
            rows = np.loadtxt(io.BytesIO(raw), dtype=np.float64, comments=None, ndmin=2)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid EventAid-R event block {source}: every token must be numeric"
            ) from error
        if rows.ndim != 2 or rows.shape[1] != 4:
            raise ValueError(
                f"Invalid EventAid-R event block {source}: expected four values per event"
            )
        # Official text columns: timestamp, x, y, polarity.
        timestamps = rows[:, 0]
        if not np.all(np.isfinite(timestamps)):
            raise ValueError(f"Invalid EventAid-R event block {source}: timestamps must be finite")
        if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError(
                f"Invalid EventAid-R event block {source}: timestamps must be monotonically "
                "non-decreasing"
            )
        polarity = rows[:, 3]
        if not np.all(np.isfinite(polarity)):
            raise ValueError(f"Invalid EventAid-R event block {source}: polarity must be finite")
        valid_polarity = (polarity == -1) | (polarity == 0) | (polarity == 1)
        if not np.all(valid_polarity):
            raise ValueError(
                f"Invalid EventAid-R event block {source}: polarity values must be -1/1 or 0/1"
            )
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
        source = f"{item['path']}::{item['event_name']}"
        events = self._read_events(zf.read(item["event_name"]), source=source)
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
        _validate_event_coordinates(events, height=height, width=width, source=source)
        raw_event_count = len(events)
        # Keep the sensor ROI aligned for recurrent/temporal evaluation.
        crop_identity = f"{item['scene']}\0{item['path'].name}"
        crop_seed = (self.seed + zlib.crc32(crop_identity.encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(crop_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        cropped_event_count = len(events)
        dataset_sampling_ratio = uniform_cap_ratio(cropped_event_count, self.max_events)
        events = stratified_subsample(events, self.max_events)
        retained_event_count = len(events)
        sample_id = f"{item['scene']}/{item['frame_id']:06d}"
        return make_sample(
            events,
            target,
            sample_id,
            (crop.height, crop.width),
            {
                "dataset": "EventAid-R",
                "scene": item["scene"],
                "sequence_index": item["sequence_index"],
                "source": str(item["path"]),
                "t0_us": item["t0_us"],
                "t1_us": item["t1_us"],
                "dt_us": (item["t1_us"] - item["t0_us"])
                if item["t0_us"] is not None and item["t1_us"] is not None
                else None,
                "raw_event_count": raw_event_count,
                "cropped_event_count": cropped_event_count,
                "retained_event_count": retained_event_count,
                "dataset_sampling_ratio": dataset_sampling_ratio,
                "crop": {
                    "left": crop.left,
                    "top": crop.top,
                    "width": crop.width,
                    "height": crop.height,
                },
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
~~~~~~~~

# src/asgcn_recon/data/eventhdr.py

~~~~~~~~python
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
    uniform_cap_ratio,
)

_EVENT_ARRAY_NAMES = ("xs", "ys", "ts", "ps")


def _invalid_file(path: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid EventHDR file {path}: {detail}")


def _numeric_scalar_attr(node: h5py.Dataset, name: str, path: Path) -> float:
    if name not in node.attrs:
        raise _invalid_file(path, f"images/{node.name.rsplit('/', 1)[-1]} is missing '{name}'")
    raw = np.asarray(node.attrs[name])
    if raw.size != 1 or raw.dtype.kind not in "iuf":
        raise _invalid_file(
            path,
            f"images/{node.name.rsplit('/', 1)[-1]} attribute '{name}' must be one number",
        )
    value = float(raw.reshape(-1)[0])
    if not np.isfinite(value):
        raise _invalid_file(
            path,
            f"images/{node.name.rsplit('/', 1)[-1]} attribute '{name}' must be finite",
        )
    return value


def _validate_event_values(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    *,
    expected: int,
    height: int,
    width: int,
    source: str,
) -> None:
    arrays = {"xs": xs, "ys": ys, "ts": ts, "ps": ps}
    lengths = {name: int(values.size) for name, values in arrays.items()}
    if any(values.ndim != 1 for values in arrays.values()) or any(
        length != expected for length in lengths.values()
    ):
        raise ValueError(
            f"Invalid EventHDR event block {source}: expected {expected} values per array, "
            f"got {lengths}"
        )

    if not np.all(np.isfinite(ts)):
        raise ValueError(f"Invalid EventHDR event block {source}: timestamps must be finite")
    if ts.size > 1 and np.any(ts[1:] < ts[:-1]):
        raise ValueError(
            f"Invalid EventHDR event block {source}: timestamps must be monotonically "
            "non-decreasing"
        )

    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError(f"Invalid EventHDR event block {source}: coordinates must be finite")
    if np.any((xs < 0) | (xs >= width)) or np.any((ys < 0) | (ys >= height)):
        raise ValueError(
            f"Invalid EventHDR event block {source}: coordinates must lie within "
            f"x=[0,{width}), y=[0,{height})"
        )

    if not np.all(np.isfinite(ps)):
        raise ValueError(f"Invalid EventHDR event block {source}: polarity must be finite")
    valid_polarity = (ps == -1) | (ps == 0) | (ps == 1)
    if not np.all(valid_polarity):
        raise ValueError(
            f"Invalid EventHDR event block {source}: polarity values must be -1/1 or 0/1"
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
        file_to_scene: dict[str, str] | None = None,
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
        self.zero_event_intervals = 0
        discovered = sorted([*self.root.rglob("*.h5"), *self.root.rglob("*.hdf5")])
        if not discovered:
            raise FileNotFoundError(
                f"No EventHDR .h5/.hdf5 files found under {self.root}. "
                "Place the official files in this directory or update dataset.root."
            )
        self.file_keys = {path: path.relative_to(self.root).as_posix() for path in discovered}
        key_to_path = {key: path for path, key in self.file_keys.items()}
        self.files = discovered
        if allowed_files is not None:
            allowed = [str(value).replace("\\", "/") for value in allowed_files]
            if len(allowed) != len(set(allowed)):
                raise ValueError("EventHDR allowed_files contains duplicate paths")
            missing = sorted(set(allowed) - set(key_to_path))
            if missing:
                preview = ", ".join(missing[:8])
                suffix = " ..." if len(missing) > 8 else ""
                raise FileNotFoundError(
                    f"EventHDR split requires {len(allowed)} files but {len(missing)} are "
                    f"missing under {self.root}: {preview}{suffix}"
                )
            self.files = [key_to_path[key] for key in allowed]
        selected_keys = [self.file_keys[path] for path in self.files]
        if file_to_scene is None:
            self.file_to_scene = {key: key for key in selected_keys}
        else:
            if not isinstance(file_to_scene, dict):
                raise TypeError("EventHDR file_to_scene must be a dictionary")
            normalized_mapping: dict[str, str] = {}
            for raw_key, scene_id in file_to_scene.items():
                if not isinstance(raw_key, str) or not raw_key.strip():
                    raise ValueError("EventHDR file_to_scene contains an invalid file key")
                key = raw_key.replace("\\", "/")
                if key in normalized_mapping:
                    raise ValueError(
                        f"EventHDR file_to_scene contains duplicate normalized key: {key}"
                    )
                if (
                    not isinstance(scene_id, str)
                    or not scene_id.strip()
                    or scene_id != scene_id.strip()
                ):
                    raise ValueError(f"EventHDR file_to_scene has an invalid scene ID for {key}")
                normalized_mapping[key] = scene_id
            missing_scenes = sorted(set(selected_keys) - set(normalized_mapping))
            if missing_scenes:
                raise ValueError(
                    "EventHDR file_to_scene is missing selected files: "
                    + ", ".join(missing_scenes[:8])
                    + (" ..." if len(missing_scenes) > 8 else "")
                )
            self.file_to_scene = {key: normalized_mapping[key] for key in selected_keys}
        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No valid EventHDR frames found under {self.root}")

    def _build_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.files:
            source_file = self.file_keys[path]
            scene = self.file_to_scene[source_file]
            with h5py.File(path, "r") as h5:
                events_group = h5.get("events")
                images_group = h5.get("images")
                if not isinstance(events_group, h5py.Group):
                    raise _invalid_file(path, "required group 'events' is missing")
                if not isinstance(images_group, h5py.Group):
                    raise _invalid_file(path, "required group 'images' is missing")

                lengths: dict[str, int] = {}
                for name in _EVENT_ARRAY_NAMES:
                    node = events_group.get(name)
                    if not isinstance(node, h5py.Dataset):
                        raise _invalid_file(path, f"required array 'events/{name}' is missing")
                    allowed_kinds = "biuf" if name == "ps" else "iuf"
                    if node.ndim != 1 or node.dtype.kind not in allowed_kinds:
                        raise _invalid_file(
                            path, f"events/{name} must be a one-dimensional numeric array"
                        )
                    lengths[name] = len(node)
                if len(set(lengths.values())) != 1:
                    raise _invalid_file(
                        path, f"event arrays must have equal lengths, got {lengths}"
                    )
                event_count = lengths["ts"]

                image_keys = sorted(k for k in images_group if k.startswith("image"))
                if not image_keys:
                    raise _invalid_file(path, "group 'images' contains no image arrays")
                selected_start_idx = 0
                selected_start_timestamp: float | None = None
                selected_sequence_index = 0
                previous_end_idx: int | None = None
                previous_timestamp: float | None = None
                for frame_index, key in enumerate(image_keys):
                    node = images_group[key]
                    if not isinstance(node, h5py.Dataset):
                        raise _invalid_file(path, f"images/{key} must be an image array")
                    raw_end_idx = _numeric_scalar_attr(node, "event_idx", path)
                    if not raw_end_idx.is_integer():
                        raise _invalid_file(path, f"images/{key} event_idx must be an integer")
                    end_idx = int(raw_end_idx)
                    timestamp = _numeric_scalar_attr(node, "timestamp", path)
                    if not 0 <= end_idx <= event_count:
                        raise _invalid_file(
                            path,
                            f"images/{key} event_idx={end_idx} is outside [0,{event_count}]",
                        )
                    if previous_end_idx is not None and end_idx < previous_end_idx:
                        raise _invalid_file(
                            path, "image event_idx values must be monotonically non-decreasing"
                        )
                    if previous_timestamp is not None and timestamp < previous_timestamp:
                        raise _invalid_file(
                            path, "image timestamps must be monotonically non-decreasing"
                        )
                    previous_end_idx = end_idx
                    previous_timestamp = timestamp
                    if frame_index % self.frame_stride == 0:
                        is_zero_event_interval = end_idx == selected_start_idx
                        if is_zero_event_interval:
                            self.zero_event_intervals += 1
                        samples.append(
                            {
                                "path": path,
                                "scene": scene,
                                "source_file": source_file,
                                "image_key": key,
                                "start_idx": selected_start_idx,
                                "end_idx": end_idx,
                                "t0": selected_start_timestamp,
                                "timestamp": timestamp,
                                "sequence_index": selected_sequence_index,
                                "zero_event_interval": is_zero_event_interval,
                            }
                        )
                        # With frame_stride > 1, aggregate every skipped event interval
                        # into the next selected output instead of silently discarding it.
                        selected_start_idx = end_idx
                        selected_start_timestamp = timestamp
                        selected_sequence_index += 1
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
        raw_ps = np.asarray(h5["events/ps"][start:end])
        image = np.asarray(h5["images"][item["image_key"]])
        target = image_array_to_tensor(
            image,
            self.target_channels,
            tone_map=self.tone_map,
            tone_map_mu=self.tone_map_mu,
        )
        height, width = target.shape[-2:]
        _validate_event_values(
            xs,
            ys,
            ts,
            raw_ps,
            expected=end - start,
            height=height,
            width=width,
            source=f"{item['path']}::{item['image_key']}",
        )
        ps = normalize_polarity(raw_ps)
        events = np.column_stack((xs, ys, ts, ps))
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1e-9)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        raw_event_count = len(events)
        # Recurrent pixels and temporal losses must refer to the same sensor ROI
        # throughout one source sequence. The crop is deterministic per file, not
        # per frame; epoch-varying sequence crops are intentionally not implemented.
        crop_identity = f"{item['scene']}\0{item['source_file']}"
        crop_seed = (self.seed + zlib.crc32(crop_identity.encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(crop_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        cropped_event_count = len(events)
        dataset_sampling_ratio = uniform_cap_ratio(cropped_event_count, self.max_events)
        events = stratified_subsample(events, self.max_events)
        retained_event_count = len(events)
        sample_id = (
            f"{item['scene']}/{item['image_key']}"
            if item["scene"] == item["source_file"]
            else f"{item['scene']}/{item['source_file']}/{item['image_key']}"
        )
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
                "source_file": item["source_file"],
                "scene": item["scene"],
                "sequence_index": item["sequence_index"],
                "raw_event_count": raw_event_count,
                "cropped_event_count": cropped_event_count,
                "retained_event_count": retained_event_count,
                "dataset_sampling_ratio": dataset_sampling_ratio,
                "zero_event_interval": bool(item["zero_event_interval"]),
                "crop": {
                    "left": crop.left,
                    "top": crop.top,
                    "width": crop.width,
                    "height": crop.height,
                },
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
~~~~~~~~

# src/asgcn_recon/data/factory.py

~~~~~~~~python
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .eventaid_r import EventAidRZipDataset
from .eventhdr import EventHDRDataset


def _normalize_eventhdr_file_key(value: str, manifest_path: Path) -> str:
    normalized = value.replace("\\", "/")
    key = PurePosixPath(normalized)
    if (
        key.is_absolute()
        or ".." in key.parts
        or not key.name
        or ":" in key.parts[0]
        or key.suffix.lower() not in {".h5", ".hdf5"}
    ):
        raise ValueError(
            f"EventHDR split manifest {manifest_path} has invalid relative HDF5 path: {value!r}"
        )
    return key.as_posix()


def _normalize_file_list(
    values: Any,
    *,
    field: str,
    manifest_path: Path,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' must be "
            "a non-empty list of HDF5 filenames"
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' contains "
            "a non-string or empty filename"
        )
    normalized = [_normalize_eventhdr_file_key(value, manifest_path) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"EventHDR split manifest {manifest_path} field '{field}' has duplicates")
    return normalized


def _normalize_scene_list(
    values: Any,
    *,
    field: str,
    manifest_path: Path,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' must be "
            "a non-empty list of physical scene IDs"
        )
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in values
    ):
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' contains "
            "an invalid physical scene ID"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"EventHDR split manifest {manifest_path} field '{field}' has duplicates")
    return list(values)


def load_eventhdr_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"EventHDR split manifest does not exist: {manifest_path}. "
            "Paths in checked-in configs are resolved from the repository root."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"EventHDR split manifest {manifest_path} must contain an object")
    raw_status = manifest.get("status", "provisional")
    if not isinstance(raw_status, str):
        raise TypeError(f"EventHDR split manifest {manifest_path} field 'status' must be a string")
    status = raw_status.strip().lower()
    if status not in {"provisional", "final"}:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} status must be 'provisional' or 'final'"
        )
    manifest["status"] = status

    scene_fields = ("scene_groups", "train_scenes", "val_scenes")
    present_scene_fields = [field for field in scene_fields if field in manifest]
    declared_schema = manifest.get("split_schema")
    if declared_schema is not None and not isinstance(declared_schema, str):
        raise TypeError(
            f"EventHDR split manifest {manifest_path} field 'split_schema' must be a string"
        )
    if declared_schema == "official_separate_roots_v1":
        if status != "final":
            raise ValueError(
                f"Official EventHDR separate-root manifest {manifest_path} must have "
                "status='final'"
            )
        if present_scene_fields:
            raise ValueError(
                f"Official EventHDR separate-root manifest {manifest_path} must not "
                "declare physical-scene fields"
            )
        deprecated_group_fields = [
            field for field in ("train_scene_groups", "val_scene_groups") if field in manifest
        ]
        if deprecated_group_fields:
            raise ValueError(
                f"Official EventHDR separate-root manifest {manifest_path} generates "
                "split-local H5 sequence groups automatically; remove: "
                + ", ".join(deprecated_group_fields)
            )
        required_semantics = "h5_sequence_file_not_physical_scene"
        if manifest.get("group_semantics") != required_semantics:
            raise ValueError(
                f"Official EventHDR separate-root manifest {manifest_path} must set "
                f"group_semantics='{required_semantics}'"
            )
        manifest["train_files"] = _normalize_file_list(
            manifest.get("train_files"), field="train_files", manifest_path=manifest_path
        )
        manifest["val_files"] = _normalize_file_list(
            manifest.get("val_files"), field="val_files", manifest_path=manifest_path
        )
        manifest["train_file_to_group"] = {
            file_key: f"official-train-h5::{file_key}"
            for file_key in manifest["train_files"]
        }
        manifest["val_file_to_group"] = {
            file_key: f"official-eval-h5::{file_key}" for file_key in manifest["val_files"]
        }
        # ``EventHDRDataset`` keeps the historical file_to_scene API, but values in
        # this schema are explicitly H5 sequence groups, not physical-scene labels.
        manifest["train_file_to_scene"] = manifest["train_file_to_group"]
        manifest["val_file_to_scene"] = manifest["val_file_to_group"]
        manifest["file_to_group"] = {
            "train": manifest["train_file_to_group"],
            "val": manifest["val_file_to_group"],
        }
        manifest["file_to_scene"] = {
            "train": manifest["train_file_to_scene"],
            "val": manifest["val_file_to_scene"],
        }
        return manifest

    if present_scene_fields and len(present_scene_fields) != len(scene_fields):
        missing = ", ".join(field for field in scene_fields if field not in manifest)
        raise ValueError(
            f"EventHDR split manifest {manifest_path} has an incomplete physical-scene "
            f"schema; missing: {missing}"
        )

    if not present_scene_fields:
        if status == "final":
            raise ValueError(
                f"Final EventHDR split manifest {manifest_path} requires scene_groups, "
                "train_scenes, and val_scenes; legacy train_files/val_files cannot "
                "prove physical-scene separation"
            )
        for field in ("train_files", "val_files"):
            manifest[field] = sorted(
                _normalize_file_list(manifest.get(field), field=field, manifest_path=manifest_path)
            )
        overlap = sorted(set(manifest["train_files"]) & set(manifest["val_files"]))
        if overlap:
            raise ValueError(
                f"EventHDR split manifest {manifest_path} leaks files across train/val: "
                + ", ".join(overlap[:8])
                + (" ..." if len(overlap) > 8 else "")
            )
        manifest["split_schema"] = "legacy_files_v1"
        manifest["train_file_to_scene"] = {
            key: key for key in manifest["train_files"]
        }
        manifest["val_file_to_scene"] = {key: key for key in manifest["val_files"]}
        manifest["file_to_scene"] = {
            **manifest["train_file_to_scene"],
            **manifest["val_file_to_scene"],
        }
        return manifest

    raw_groups = manifest["scene_groups"]
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field 'scene_groups' must be "
            "a non-empty object"
        )
    scene_groups: dict[str, list[str]] = {}
    file_to_scene: dict[str, str] = {}
    for scene_id, values in raw_groups.items():
        if not isinstance(scene_id, str) or not scene_id.strip() or scene_id != scene_id.strip():
            raise ValueError(
                f"EventHDR split manifest {manifest_path} has an invalid physical scene ID"
            )
        files = _normalize_file_list(
            values,
            field=f"scene_groups.{scene_id}",
            manifest_path=manifest_path,
        )
        for file_key in files:
            owner = file_to_scene.get(file_key)
            if owner is not None:
                raise ValueError(
                    f"EventHDR split manifest {manifest_path} assigns file {file_key!r} "
                    f"to multiple physical scenes: {owner!r}, {scene_id!r}"
                )
            file_to_scene[file_key] = scene_id
        scene_groups[scene_id] = files

    train_scenes = _normalize_scene_list(
        manifest["train_scenes"], field="train_scenes", manifest_path=manifest_path
    )
    val_scenes = _normalize_scene_list(
        manifest["val_scenes"], field="val_scenes", manifest_path=manifest_path
    )
    unknown_scenes = sorted((set(train_scenes) | set(val_scenes)) - set(scene_groups))
    if unknown_scenes:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} references undefined physical "
            "scenes: " + ", ".join(unknown_scenes)
        )
    scene_overlap = sorted(set(train_scenes) & set(val_scenes))
    if scene_overlap:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} leaks physical scenes across "
            "train/val: " + ", ".join(scene_overlap)
        )
    unassigned_scenes = sorted(set(scene_groups) - set(train_scenes) - set(val_scenes))
    if unassigned_scenes:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} leaves physical scenes unassigned: "
            + ", ".join(unassigned_scenes)
        )

    train_files = [file for scene in train_scenes for file in scene_groups[scene]]
    val_files = [file for scene in val_scenes for file in scene_groups[scene]]
    for field, normalized in (("train_files", train_files), ("val_files", val_files)):
        if field in manifest:
            declared = _normalize_file_list(
                manifest[field], field=field, manifest_path=manifest_path
            )
            if set(declared) != set(normalized):
                raise ValueError(
                    f"EventHDR split manifest {manifest_path} field '{field}' does not "
                    "match the physical-scene assignment"
                )
        manifest[field] = normalized
    manifest["scene_groups"] = dict(sorted(scene_groups.items()))
    manifest["train_scenes"] = train_scenes
    manifest["val_scenes"] = val_scenes
    manifest["file_to_scene"] = dict(sorted(file_to_scene.items()))
    manifest["train_file_to_scene"] = {
        file_key: file_to_scene[file_key] for file_key in train_files
    }
    manifest["val_file_to_scene"] = {
        file_key: file_to_scene[file_key] for file_key in val_files
    }
    manifest["split_schema"] = "physical_scenes_v1"
    return manifest


def _discover_eventhdr_files(
    root: Path,
    *,
    exclude_roots: tuple[Path, ...] = (),
) -> set[str]:
    """List H5 keys under one logical root without double-counting nested roots."""
    resolved_root = root.resolve()
    resolved_excludes = tuple(
        candidate.resolve()
        for candidate in exclude_roots
        if candidate.resolve() != resolved_root
        and candidate.resolve().is_relative_to(resolved_root)
    )
    discovered: set[str] = set()
    for pattern in ("*.h5", "*.hdf5"):
        for path in root.rglob(pattern):
            if any(path.resolve().is_relative_to(excluded) for excluded in resolved_excludes):
                continue
            discovered.add(path.relative_to(root).as_posix())
    return discovered


def build_dataset(config: dict[str, Any], split: str = "train"):
    cfg = dict(config)
    dataset_type = cfg.pop("type")
    expected_file_count = cfg.pop("expected_file_count", None)
    file_manifest = cfg.pop("file_manifest", None)
    training_root = Path(cfg["root"])
    validation_root = Path(cfg["val_root"]) if cfg.get("val_root") else training_root
    if split == "val" and cfg.get("val_root"):
        cfg["root"] = cfg["val_root"]
    root = cfg.pop("root")
    cfg.pop("val_root", None)
    split_manifest = cfg.pop("split_manifest", None)
    cfg["random_crop"] = split == "train" and cfg.get("crop_size") is not None
    if dataset_type == "eventhdr":
        eventhdr_group_semantics: str | None = None
        if split_manifest and split in {"train", "val", "calibration"}:
            manifest_path = Path(split_manifest)
            manifest = load_eventhdr_split_manifest(manifest_path)
            eventhdr_group_semantics = manifest.get("group_semantics")
            if eventhdr_group_semantics is None:
                eventhdr_group_semantics = (
                    "physical_scene"
                    if manifest["split_schema"] == "physical_scenes_v1"
                    else "h5_sequence_file_not_physical_scene"
                )
            if manifest["status"] == "final":
                roots_match = training_root.resolve() == validation_root.resolve()
                if manifest["split_schema"] == "official_separate_roots_v1" and roots_match:
                    raise ValueError(
                        "Official EventHDR train/eval split requires distinct dataset.root "
                        "and dataset.val_root directories"
                    )
                if roots_match:
                    coverage_specs = (
                        (
                            "dataset.root",
                            training_root,
                            set(manifest["file_to_scene"]),
                            (),
                        ),
                    )
                else:
                    coverage_specs = (
                        (
                            "dataset.root train_files",
                            training_root,
                            set(manifest["train_files"]),
                            (validation_root,),
                        ),
                        (
                            "dataset.val_root val_files",
                            validation_root,
                            set(manifest["val_files"]),
                            (training_root,),
                        ),
                    )
                for label, coverage_root, declared, excluded_roots in coverage_specs:
                    discovered = _discover_eventhdr_files(
                        coverage_root,
                        exclude_roots=excluded_roots,
                    )
                    undeclared = sorted(discovered - declared)
                    missing = sorted(declared - discovered)
                    if undeclared or missing:
                        details = []
                        if undeclared:
                            details.append("undeclared: " + ", ".join(undeclared[:8]))
                        if missing:
                            details.append("missing: " + ", ".join(missing[:8]))
                        raise ValueError(
                            f"Final EventHDR manifest must cover every H5 under {label} ("
                            + "; ".join(details)
                            + ")"
                        )
            key = "val_files" if split == "val" else "train_files"
            if manifest["split_schema"] == "official_separate_roots_v1":
                mapping_key = "val_file_to_group" if split == "val" else "train_file_to_group"
            else:
                mapping_key = (
                    "val_file_to_scene" if split == "val" else "train_file_to_scene"
                )
            cfg["allowed_files"] = manifest[key]
            cfg["file_to_scene"] = manifest[mapping_key]
        dataset = EventHDRDataset(root=root, **cfg)
        if eventhdr_group_semantics is not None:
            dataset.group_semantics = eventhdr_group_semantics
        if expected_file_count is not None and len(dataset.files) != int(expected_file_count):
            dataset.close()
            raise ValueError(
                f"EventHDR coverage requires exactly {int(expected_file_count)} H5 files; "
                f"found {len(dataset.files)}"
            )
        return dataset
    if dataset_type == "eventaid_r_zip":
        dataset = EventAidRZipDataset(root=root, **cfg)
        if file_manifest is not None:
            manifest_path = Path(file_manifest)
            if not manifest_path.is_file():
                dataset.close()
                raise FileNotFoundError(f"EventAid-R file manifest does not exist: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                f"{item['scene']}.zip"
                for item in manifest.get("files", [])
                if isinstance(item, dict) and item.get("scene")
            }
            present = {path.name for path in dataset.zip_paths}
            if not expected or present != expected:
                dataset.close()
                missing = sorted(expected - present)
                extra = sorted(present - expected)
                raise ValueError(
                    "EventAid-R coverage does not match the fixed file manifest "
                    f"(missing={missing}, extra={extra})"
                )
        if expected_file_count is not None and len(dataset.zip_paths) != int(expected_file_count):
            dataset.close()
            raise ValueError(
                f"EventAid-R coverage requires exactly {int(expected_file_count)} ZIP files; "
                f"found {len(dataset.zip_paths)}"
            )
        return dataset
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Graphs and sensor resolutions are variable-sized; the model loops over this small list.
    return batch
~~~~~~~~

# src/asgcn_recon/engine.py

~~~~~~~~python
from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import build_dataset, collate_samples, load_eventhdr_split_manifest
from .graph import PAPER_CORE_VERSION, PaperSplineConv
from .losses import ReconstructionLoss
from .metrics import (
    MetricAccumulator,
    frame_metrics,
    percentile,
    temporal_consistency_error,
)
from .model import ASGCNReconstructor
from .utils import (
    atomic_torch_save,
    load_json,
    move_sample,
    resolve_device,
    save_image,
    save_json,
    set_seed,
    write_frame_csv,
)


def build_model(config: dict[str, Any]) -> ASGCNReconstructor:
    return ASGCNReconstructor(**config)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Return a deterministic digest that binds checkpoint metadata to tensor bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_model_state_digest(
    state: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
) -> str:
    """Require every paper-core checkpoint to bind metadata to exact tensor bytes."""
    expected = checkpoint.get("model_state_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing a valid model_state_sha256")
    computed = _model_state_sha256(state)
    if computed != expected:
        raise ValueError(
            f"Checkpoint {checkpoint_path} model_state_sha256 does not match tensor bytes"
        )
    return computed


def _validate_loaded_conversion_state(
    model: ASGCNReconstructor,
    metadata: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    """Cross-check user-editable checkpoint metadata against persistent layer flags."""
    for name, tensor in model.state_dict().items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Checkpoint {checkpoint_path} contains non-finite state: {name}")
    bn_flags = [bool(layer.bn_bypassed.item()) for layer in model.encoder.layers]
    normalized_flags = [bool(layer.snn_normalized.item()) for layer in model.encoder.layers]
    if len(set(bn_flags)) > 1 or len(set(normalized_flags)) > 1:
        raise ValueError(f"Checkpoint {checkpoint_path} contains partially converted graph layers")
    state_bn_folded = all(bn_flags)
    state_normalized = all(normalized_flags)
    metadata_bn_folded = bool(metadata.get("batch_norm_folded"))
    metadata_normalized = bool(metadata.get("parameter_normalized"))
    if metadata_bn_folded != state_bn_folded:
        raise ValueError(
            f"Checkpoint {checkpoint_path} batch_norm_folded metadata disagrees "
            "with layer bn_bypassed state"
        )
    if metadata_normalized != state_normalized:
        raise ValueError(
            f"Checkpoint {checkpoint_path} parameter_normalized metadata disagrees "
            "with layer snn_normalized state"
        )
    checkpoint_type = metadata.get("checkpoint_type")
    if checkpoint_type == "snn_inference" and not (state_bn_folded and state_normalized):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is labeled snn_inference but its graph "
            "layers are not fully BN-folded and Eq. (6)-normalized"
        )
    if checkpoint_type == "snn_inference":
        sample_counts = [int(value) for value in model.encoder.calibration_samples_seen.tolist()]
        if not sample_counts or min(sample_counts) < 1:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has graph layers without valid calibration"
            )
        for index, layer in enumerate(model.encoder.layers):
            if not bool((layer.activation_max > 0).all()):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} has non-positive lambda"
                )
            if not bool((layer.threshold > 0).all()):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} has non-positive threshold"
                )
            if not torch.equal(layer.threshold, torch.ones_like(layer.threshold)):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} threshold is not the "
                    "unit threshold produced by Eq. (6) conversion"
                )
        summary = metadata.get("snn_calibration_summary")
        if not isinstance(summary, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} is missing calibration summary")
        if summary.get("valid_samples_per_layer") != sample_counts:
            raise ValueError(
                f"Checkpoint {checkpoint_path} calibration metadata disagrees with layer state"
            )
        minimum = min(sample_counts)
        if int(summary.get("minimum_valid_samples", 0) or 0) != minimum:
            raise ValueError(f"Checkpoint {checkpoint_path} calibration minimum is inconsistent")
        if int(metadata.get("snn_calibration_valid_samples", 0) or 0) != minimum:
            raise ValueError(
                f"Checkpoint {checkpoint_path} valid calibration count is inconsistent"
            )
        selected = int(metadata.get("snn_calibration_samples", 0) or 0)
        if selected < minimum:
            raise ValueError(
                f"Checkpoint {checkpoint_path} selected calibration count is inconsistent"
            )
    if checkpoint_type in {"ann_inference", "training"} and (state_bn_folded or state_normalized):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is labeled {checkpoint_type} but contains "
            "converted SNN graph-layer state"
        )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[ASGCNReconstructor, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dictionary")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has no embedded model_config. Legacy/raw "
            "state dictionaries are incompatible with the paper-core architecture."
        )
    architecture_version = model_config.get("architecture_version")
    if architecture_version != PAPER_CORE_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has architecture_version="
            f"{architecture_version!r}; paper-core version {PAPER_CORE_VERSION} is "
            "required. Legacy edge-MLP checkpoints cannot be loaded as ASGCN."
        )
    if model_config != fallback_model_config:
        raise ValueError(
            f"Checkpoint {checkpoint_path} model_config differs from config.model. "
            "Use the exact training architecture; inference-only SNN dynamics must "
            "be selected with the explicit --snn-dynamics override."
        )
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has no model state dictionary; raw state "
            "dictionaries are incompatible with the paper-core checkpoint protocol."
        )
    state = checkpoint.pop("model")
    _validate_model_state_digest(state, checkpoint, checkpoint_path)
    metadata = checkpoint
    model = build_model(model_config).to(device)
    model.load_state_dict(state, strict=True)
    _validate_loaded_conversion_state(model, metadata, checkpoint_path)
    del state
    return model, metadata


def _dataset_group_key(dataset, index: int) -> str:
    """Return a stable file/scene key without decoding the sample payload."""
    records = getattr(dataset, "samples", None)
    if records is not None:
        record = records[index]
        if isinstance(record, dict):
            if record.get("scene") is not None:
                return str(record["scene"])
            if record.get("path") is not None:
                path = Path(record["path"])
                root = getattr(dataset, "root", None)
                if root is not None:
                    try:
                        return path.relative_to(Path(root)).as_posix()
                    except ValueError:
                        pass
                return path.name
    # Supported datasets expose ``samples``. This fallback keeps custom datasets
    # usable in validation without making balanced sampling silently incorrect.
    sample = dataset[index]
    metadata = sample.get("metadata", {})
    return str(metadata.get("scene") or metadata.get("source") or "unknown")


def _balanced_sample_indices(dataset, limit: int | None, seed: int = 2026) -> list[int]:
    """Select near-equal, time-spread samples from every file/scene.

    Round-robin allocation prevents long files from consuming the complete
    validation/calibration budget. Within each group, linspace covers the whole
    sequence instead of only its prefix. Sorting the result keeps each scene's
    original temporal order for the recurrent decoder.
    """
    size = len(dataset)
    if limit is None or int(limit) >= size:
        return list(range(size))
    limit = int(limit)
    if limit < 1:
        raise ValueError("sample limit must be at least 1")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(size):
        grouped[_dataset_group_key(dataset, index)].append(index)
    group_keys = sorted(grouped)
    random.Random(seed).shuffle(group_keys)

    quotas = {key: 0 for key in group_keys}
    allocated = 0
    depth = 0
    while allocated < limit:
        added = False
        for key in group_keys:
            indices = grouped[key]
            if depth < len(indices):
                quotas[key] += 1
                allocated += 1
                added = True
                if allocated == limit:
                    break
        if not added:
            break
        depth += 1

    selected: list[int] = []
    for key, quota in quotas.items():
        indices = grouped[key]
        if quota == 1:
            offsets = [len(indices) // 2]
        elif quota > 1:
            offsets = np.linspace(0, len(indices) - 1, num=quota, dtype=int).tolist()
        else:
            offsets = []
        selected.extend(indices[offset] for offset in offsets)
    return sorted(selected)


def _balanced_contiguous_indices(
    dataset,
    limit: int | None,
    seed: int = 2026,
    *,
    require_all_groups: bool = False,
) -> list[int]:
    """Select one deterministic contiguous window from every allocated group."""
    size = len(dataset)
    if limit is None or int(limit) >= size:
        return list(range(size))
    limit = int(limit)
    if limit < 1:
        raise ValueError("sample limit must be at least 1")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(size):
        grouped[_dataset_group_key(dataset, index)].append(index)
    if require_all_groups and limit < len(grouped):
        raise ValueError(
            f"validation sample limit {limit} is smaller than the {len(grouped)} "
            "available groups; every validation group must be represented"
        )

    group_keys = sorted(grouped)
    allocation_order = list(group_keys)
    random.Random(seed).shuffle(allocation_order)
    quotas = {key: 0 for key in group_keys}
    allocated = 0
    depth = 0
    while allocated < limit:
        added = False
        for key in allocation_order:
            if depth < len(grouped[key]):
                quotas[key] += 1
                allocated += 1
                added = True
                if allocated == limit:
                    break
        if not added:
            break
        depth += 1

    window_rng = random.Random(f"contiguous:{seed}")
    selected: list[int] = []
    for key in group_keys:
        indices = grouped[key]
        quota = quotas[key]
        if quota:
            start = window_rng.randrange(len(indices) - quota + 1)
            selected.extend(indices[start : start + quota])
    return sorted(selected)


def _representative_schedule(
    dataset, count: int, seed: int, *, contiguous: bool = False
) -> list[int]:
    """Build an exact-length balanced schedule, cycling only if count exceeds data."""
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count == 0:
        return []
    sampler = _balanced_contiguous_indices if contiguous else _balanced_sample_indices
    base = sampler(dataset, min(count, len(dataset)), seed=seed)
    repeats = (count + len(base) - 1) // len(base)
    return (base * repeats)[:count]


def _prefix_context_schedule(
    dataset,
    scored_indices: list[int],
    max_context_frames: int | None = None,
) -> tuple[list[int], set[int]]:
    """Prepend contiguous predecessor frames as unscored recurrent context."""
    if not scored_indices:
        return [], set()
    if max_context_frames is not None and int(max_context_frames) < 0:
        raise ValueError("max_context_frames must be non-negative or null")
    scored = set(scored_indices)
    if len(scored) != len(scored_indices):
        raise ValueError("prefix context schedule requires unique scored indices")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        grouped[_dataset_group_key(dataset, index)].append(index)

    schedule: list[int] = []
    score_positions: set[int] = set()
    selected_groups = sorted(
        {_dataset_group_key(dataset, index) for index in scored_indices},
        key=lambda key: grouped[key][0],
    )
    for group in selected_groups:
        group_indices = grouped[group]
        selected = [index for index in group_indices if index in scored]
        first_position = group_indices.index(selected[0])
        last_position = group_indices.index(selected[-1])
        start_position = (
            0 if max_context_frames is None else max(0, first_position - int(max_context_frames))
        )
        for index in group_indices[start_position : last_position + 1]:
            position = len(schedule)
            schedule.append(index)
            if index in scored:
                score_positions.add(position)
    return schedule, score_positions


def _dataset_sample_identity(dataset, index: int) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "dataset_index": index,
        "group": _dataset_group_key(dataset, index),
    }
    records = getattr(dataset, "samples", None)
    if records is None or not isinstance(records[index], dict):
        return identity
    record = records[index]
    for key in (
        "source_file",
        "sequence_index",
        "frame_id",
        "image_key",
        "event_name",
        "target_name",
        "start_idx",
        "end_idx",
        "timestamp",
    ):
        value = record.get(key)
        if value is not None:
            identity[key] = value.item() if isinstance(value, np.generic) else value
    return identity


def _dataset_source_fingerprint(dataset) -> dict[str, Any]:
    root = Path(getattr(dataset, "root", ".")).resolve()
    sources = getattr(dataset, "files", None)
    if sources is None:
        sources = getattr(dataset, "zip_paths", [])
    files = []
    for raw_path in sources:
        path = Path(raw_path).resolve()
        stat = path.stat()
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            key = path.as_posix()
        files.append(
            {
                "path": key,
                "size": stat.st_size,
            }
        )
    return {"files": files}


def _dataset_content_fingerprint(
    dataset, cache: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Hash every selected source file for path- and mtime-independent exact resume."""
    cache = {} if cache is None else cache
    root = Path(getattr(dataset, "root", ".")).resolve()
    sources = getattr(dataset, "files", None)
    if sources is None:
        sources = getattr(dataset, "zip_paths", [])
    combined = hashlib.sha256()
    total_bytes = 0
    entries: list[tuple[str, Path]] = []
    for value in sources:
        raw_path = Path(value).resolve()
        try:
            relative = raw_path.relative_to(root).as_posix()
        except ValueError:
            relative = raw_path.name
        entries.append((relative, raw_path))
    for relative, raw_path in sorted(entries):
        stat = raw_path.stat()
        cache_key = str(raw_path)
        signature = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        cached = cache.get(cache_key)
        if not isinstance(cached, dict) or any(
            cached.get(key) != value for key, value in signature.items()
        ):
            file_hash = hashlib.sha256()
            size = 0
            with raw_path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    file_hash.update(chunk)
                    size += len(chunk)
            cached = {**signature, "sha256": file_hash.hexdigest()}
            cache[cache_key] = cached
        size = int(cached["size"])
        digest = str(cached["sha256"])
        total_bytes += size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "algorithm": "sha256-full-files-v1",
        "files": len(sources),
        "bytes": total_bytes,
        "sha256": combined.hexdigest(),
    }


def _load_data_hash_cache(path: Path, rehash: bool) -> dict[str, dict[str, Any]]:
    if rehash or not path.is_file():
        return {}
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return {}
    if payload.get("version") != 1 or not isinstance(payload.get("files"), dict):
        return {}
    return payload["files"]


def _sampling_summary(dataset, indices: list[int]) -> dict[str, Any]:
    counts = Counter(_dataset_group_key(dataset, index) for index in indices)
    available_counts = Counter(_dataset_group_key(dataset, index) for index in range(len(dataset)))
    return {
        "selected_samples": len(indices),
        "selected_groups": len(counts),
        "available_groups": len(available_counts),
        "per_group": dict(sorted(counts.items())),
        "available_per_group": dict(sorted(available_counts.items())),
        "selected": [_dataset_sample_identity(dataset, index) for index in indices],
        "source_fingerprint": _dataset_source_fingerprint(dataset),
    }


def _sampling_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep routine epoch logs compact; exact identities live in the protocol."""
    keys = (
        "selected_samples",
        "selected_groups",
        "available_groups",
        "per_group",
        "available_per_group",
        "context_policy",
        "max_context_frames_per_group",
        "context_samples",
        "forward_samples",
    )
    return {key: summary[key] for key in keys if key in summary}


def _sample_sequence_info(
    sample: dict[str, Any],
) -> tuple[str, int | None, tuple[int, int]]:
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    scene = str(metadata.get("scene", "unknown"))
    raw_index = metadata.get("sequence_index")
    try:
        sequence_index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        sequence_index = None
    sensor_size = tuple(int(value) for value in sample["sensor_size"])
    return scene, sequence_index, sensor_size


def _continues_sequence(
    scene: str,
    sequence_index: int | None,
    sensor_size: tuple[int, int],
    previous_scene: str | None,
    previous_sequence_index: int | None,
    previous_sensor_size: tuple[int, int] | None,
) -> bool:
    if scene != previous_scene or sensor_size != previous_sensor_size:
        return False
    if sequence_index is None or previous_sequence_index is None:
        return True
    return sequence_index == previous_sequence_index + 1


def _macro_ssim(validation: dict[str, Any]) -> float:
    """Read the scene-balanced selection score, with legacy checkpoint support."""
    macro = validation.get("macro", {})
    if "ssim" in macro:
        return float(macro["ssim"])
    if "ssim" in validation:  # Checkpoints written before structured validation.
        return float(validation["ssim"])
    return float("-inf")


def _resume_best_macro_ssim(checkpoint: dict[str, Any]) -> float:
    """Reject legacy micro-SSIM best scores that cannot be compared to macro SSIM."""
    metric = checkpoint.get("best_metric")
    if metric == "macro_ssim":
        return float(checkpoint.get("best_ssim", _macro_ssim(checkpoint.get("val", {}))))
    validation = checkpoint.get("val", {})
    macro = validation.get("macro", {}) if isinstance(validation, dict) else {}
    if "ssim" in macro:
        return float(macro["ssim"])
    raise ValueError(
        "Resume checkpoint predates macro-SSIM model selection, so its best_ssim is "
        "not comparable. Start a new run or migrate the checkpoint with a verified "
        "macro validation score."
    )


def _validate_resume_best_pair(
    resume_checkpoint: dict[str, Any], best_checkpoint: dict[str, Any]
) -> None:
    """Ensure last.pt and best.pt belong to the same exact training run."""
    if best_checkpoint.get("validation_protocol") != resume_checkpoint.get("validation_protocol"):
        raise ValueError("Historical best.pt has a different validation protocol")
    if best_checkpoint.get("model_config") != resume_checkpoint.get("model_config"):
        raise ValueError("Historical best.pt has a different model configuration")
    if best_checkpoint.get("training_protocol") != resume_checkpoint.get("training_protocol"):
        raise ValueError("Historical best.pt has a different training protocol")
    for name, checkpoint in (
        ("resume checkpoint", resume_checkpoint),
        ("historical best.pt", best_checkpoint),
    ):
        if checkpoint.get("paper_core_version") != PAPER_CORE_VERSION:
            raise ValueError(f"{name} does not declare paper_core_version={PAPER_CORE_VERSION}")
    if best_checkpoint.get("best_metric") != "macro_ssim":
        raise ValueError("Historical best.pt does not use macro_ssim model selection")
    resume_best = _resume_best_macro_ssim(resume_checkpoint)
    historical_best = _resume_best_macro_ssim(best_checkpoint)
    if historical_best != resume_best:
        raise ValueError("Historical best.pt score does not match the resume checkpoint")
    if _macro_ssim(best_checkpoint.get("val", {})) != historical_best:
        raise ValueError("Historical best.pt validation score is internally inconsistent")
    if int(best_checkpoint.get("epoch", -1)) > int(resume_checkpoint.get("epoch", -1)):
        raise ValueError("Historical best.pt is newer than the resume checkpoint")
    best_digest = best_checkpoint.get("model_state_sha256")
    resume_digest = resume_checkpoint.get("best_model_state_sha256")
    if not isinstance(best_digest, str) or re.fullmatch(r"[0-9a-f]{64}", best_digest) is None:
        raise ValueError("Historical best.pt is missing a valid model state digest")
    if best_digest != resume_digest:
        raise ValueError("Historical best.pt model digest does not match the resume checkpoint")


def _validate_snn_request(
    inference_mode: str,
    simulation_steps: int,
    checkpoint: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> None:
    if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
        raise ValueError("simulation_steps must be an integer")
    simulation_steps = int(simulation_steps)
    if inference_mode == "ann":
        if checkpoint is not None and (
            checkpoint.get("checkpoint_type") == "snn_inference"
            or bool(checkpoint.get("parameter_normalized"))
        ):
            location = f" {checkpoint_path}" if checkpoint_path is not None else ""
            raise ValueError(
                f"ANN inference requires an ANN checkpoint;{location} contains "
                "Eq. (6)-normalized SNN weights. Use best.pt for ANN inference or "
                "select inference_mode='snn'."
            )
        return
    if inference_mode != "snn":
        return
    if int(simulation_steps) < 1:
        raise ValueError("simulation_steps must be at least 1 for SNN inference")
    if checkpoint is None:
        return
    calibration_samples = int(checkpoint.get("snn_calibration_samples", 0) or 0)
    valid_calibration_samples = int(checkpoint.get("snn_calibration_valid_samples", 0) or 0)
    requirements_met = (
        checkpoint.get("checkpoint_type") == "snn_inference"
        and bool(checkpoint.get("batch_norm_folded"))
        and calibration_samples >= 1
        and valid_calibration_samples >= 1
        and checkpoint.get("paper_core_version") == PAPER_CORE_VERSION
        and bool(checkpoint.get("parameter_normalized"))
    )
    if not requirements_met:
        location = f" {checkpoint_path}" if checkpoint_path is not None else ""
        raise ValueError(
            f"SNN inference requires a calibrated checkpoint;{location} is missing "
            "checkpoint_type=snn_inference, batch_norm_folded=true, "
            "snn_calibration_samples>=1, "
            "snn_calibration_valid_samples>=1, "
            f"paper_core_version={PAPER_CORE_VERSION}, or parameter_normalized=true. "
            "Run calibrate first."
        )


def _set_inference_snn_dynamics(
    model: ASGCNReconstructor,
    inference_mode: str,
    override: str | None,
) -> None:
    if override is None:
        return
    if inference_mode != "snn":
        raise ValueError("snn_dynamics override is only valid for SNN inference")
    if override not in {"literal_eq15", "standard_if"}:
        raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
    model.snn_dynamics = override


def _inference_run_label(
    inference_mode: str,
    simulation_steps: int,
    snn_dynamics: str,
) -> str:
    return "ann" if inference_mode == "ann" else f"snn_{snn_dynamics}_T{int(simulation_steps)}"


def _prediction_artifact_stem(sample_id: Any, index: int) -> str:
    """Create a bounded, collision-resistant filename valid on Linux and Windows."""
    raw = str(sample_id)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")[:64] or "sample"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{int(index):08d}_{slug}_{digest}"


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


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


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        raise TypeError("Exact resume requires a dictionary rng_state")
    missing = sorted({"python", "numpy", "torch"} - set(state))
    if missing:
        raise ValueError("Exact resume rng_state is missing: " + ", ".join(missing))
    if not torch.is_tensor(state["torch"]):
        raise ValueError("Exact resume rng_state['torch'] must be a tensor")
    if torch.cuda.is_available():
        cuda_state = state.get("cuda")
        if not isinstance(cuda_state, list) or len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "Exact CUDA resume requires one rng_state['cuda'] tensor per visible device"
            )
        if any(not torch.is_tensor(value) for value in cuda_state):
            raise ValueError("Exact resume CUDA RNG entries must be tensors")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Exact resume contains an invalid RNG state schema") from error


def _optimizer_mode(train_config: dict[str, Any]) -> str:
    mode = str(train_config.get("optimizer", "adamw")).strip().lower()
    if mode not in {"adamw", "adam_gc"}:
        raise ValueError("train.optimizer must be 'adamw' or 'adam_gc'")
    return mode


def _scheduler_spec(train_config: dict[str, Any]) -> dict[str, Any] | None:
    raw_milestones = train_config.get("lr_milestones")
    if raw_milestones is None or raw_milestones == []:
        return None
    if not isinstance(raw_milestones, (list, tuple)):
        raise TypeError("train.lr_milestones must be a list of positive epochs")
    milestones = sorted(int(value) for value in raw_milestones)
    if not milestones or any(value < 1 for value in milestones):
        raise ValueError("train.lr_milestones must contain positive epochs")
    gamma = float(train_config.get("lr_gamma", 0.1))
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("train.lr_gamma must be finite and greater than zero")
    return {
        "name": "MultiStepLR",
        "milestones": milestones,
        "gamma": gamma,
        "step_unit": "epoch",
        "step_timing": "after_epoch",
    }


def _build_optimizer(model: torch.nn.Module, train_config: dict[str, Any]) -> torch.optim.Optimizer:
    mode = _optimizer_mode(train_config)
    optimizer_class = torch.optim.AdamW if mode == "adamw" else torch.optim.Adam
    return optimizer_class(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-6)),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_config: dict[str, Any]
) -> torch.optim.lr_scheduler.MultiStepLR | None:
    spec = _scheduler_spec(train_config)
    if spec is None:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=spec["milestones"],
        gamma=spec["gamma"],
    )


def _centralize_gradients(model: torch.nn.Module) -> None:
    """Apply paper-style gradient centralization to matrix/kernel gradients."""
    spline_parameters: set[int] = set()
    for module in model.modules():
        if not isinstance(module, PaperSplineConv):
            continue
        for parameter in (module.weight, module.root):
            if parameter is None:
                continue
            spline_parameters.add(id(parameter))
            gradient = parameter.grad
            if gradient is None or gradient.ndim <= 1:
                continue
            dimensions = tuple(range(gradient.ndim - 1))
            gradient.subtract_(gradient.mean(dim=dimensions, keepdim=True))

    for parameter in model.parameters():
        if id(parameter) in spline_parameters:
            continue
        gradient = parameter.grad
        if gradient is None or gradient.ndim <= 1:
            continue
        dimensions = tuple(range(1, gradient.ndim))
        gradient.subtract_(gradient.mean(dim=dimensions, keepdim=True))


def _source_tree_sha256(project_root: Path) -> str:
    """Hash executable project source so resume cannot cross silent code edits."""
    digest = hashlib.sha256()
    source_root = project_root / "src"
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Python source files found under {source_root}")
    for path in files:
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_provenance(project_root: Path) -> dict[str, Any]:
    """Return best-effort Git identity without making Git a runtime dependency."""

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal", "--", "src")
    return {
        "git_commit": commit,
        "git_source_dirty": None if status is None else bool(status),
    }


def _training_protocol(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Return every configured choice that can change the optimization trajectory.

    ``epochs``, logging cadence, the resume path, and output paths are deliberately
    absent: changing those does not alter an already completed optimizer step. The
    normalized values below make omitted defaults compare equal to explicit defaults.
    """
    train_config = config["train"]
    requested_amp = bool(train_config.get("amp", True))
    effective_amp = requested_amp and device.type == "cuda"
    configured_weights = train_config.get("loss_weights") or {}
    loss_weights = {
        "charbonnier": float(configured_weights.get("charbonnier", 1.0)),
        "ssim": float(configured_weights.get("ssim", 0.2)),
        "gradient": float(configured_weights.get("gradient", 0.1)),
        "temporal": float(configured_weights.get("temporal", 0.0)),
    }
    max_train_samples = train_config.get("max_train_samples")
    grad_clip = float(train_config.get("grad_clip", 1.0))
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("train.grad_clip must be finite and greater than zero")
    num_workers = int(train_config.get("num_workers", 0))
    prefetch_factor = train_config.get("prefetch_factor")
    persistent_workers = train_config.get("persistent_workers")
    effective_persistent_workers = (
        None
        if num_workers == 0
        else True
        if persistent_workers is None
        else bool(persistent_workers)
    )
    effective_prefetch_factor = (
        None if num_workers == 0 else 2 if prefetch_factor is None else int(prefetch_factor)
    )
    optimizer_mode = _optimizer_mode(train_config)
    optimizer_name = "AdamW" if optimizer_mode == "adamw" else "Adam"
    raw_validate_every = train_config.get("validate_every", 1)
    validate_every = (
        None if raw_validate_every is None else max(1, int(raw_validate_every))
    )
    project_root = Path(__file__).resolve().parents[2]
    git_provenance = _git_provenance(project_root)
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        compute_capability = list(torch.cuda.get_device_capability(device_index))
    else:
        gpu_name = None
        compute_capability = None
    return {
        "version": 3,
        "seed": int(config.get("seed", 2026)),
        "optimizer": {
            "mode": optimizer_mode,
            "name": optimizer_name,
            "learning_rate": float(train_config.get("learning_rate", 2e-4)),
            "weight_decay": float(train_config.get("weight_decay", 1e-6)),
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "gradient_centralization": optimizer_mode == "adam_gc",
            "gradient_centralization_dimensions": "all_except_output",
        },
        "scheduler": _scheduler_spec(train_config),
        "loss_weights": loss_weights,
        "gradient_clipping": {
            "max_norm": grad_clip,
            "norm_type": 2.0,
        },
        "data_order": {
            "batch_size": int(train_config.get("batch_size", 1)),
            "max_train_samples": (None if max_train_samples is None else int(max_train_samples)),
            "shuffle": False,
            "num_workers": num_workers,
            "persistent_workers": effective_persistent_workers,
            "prefetch_factor": effective_prefetch_factor,
        },
        "mixed_precision": {
            "requested": requested_amp,
            "effective": effective_amp,
            "autocast_dtype": "float16" if effective_amp else None,
            "gradient_scaler": effective_amp,
        },
        "validate_every": validate_every,
        "checkpoint_selection": (
            "single_final_epoch" if validate_every is None else "best_validation_macro_ssim"
        ),
        "recurrent_state_detached_each_sample": True,
        "runtime": {
            "device_type": device.type,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "cudnn": (torch.backends.cudnn.version() if device.type == "cuda" else None),
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
            "cuda_matmul_allow_tf32": (
                bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None
            ),
            "cudnn_allow_tf32": (
                bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None
            ),
            "cudnn_benchmark": (
                bool(torch.backends.cudnn.benchmark) if device.type == "cuda" else None
            ),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
        "source": {
            "source_tree_sha256": _source_tree_sha256(project_root),
            **git_provenance,
        },
    }


def _validate_training_protocol(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = checkpoint.get("training_protocol")
    if actual is None:
        raise ValueError(
            "Resume checkpoint is missing training_protocol and cannot provide an "
            "exact training resume. Start a new run with the current checkpoint schema."
        )
    if not isinstance(actual, dict):
        raise TypeError("Resume checkpoint training_protocol must be a dictionary")
    if actual != expected:
        keys = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        changed = ", ".join(keys) if keys else "unknown fields"
        raise ValueError("Resume training protocol differs from the checkpoint in: " + changed)


def _ensure_finite_loss(
    loss: torch.Tensor,
    loss_parts: dict[str, float],
    *,
    epoch: int,
    step: int,
    sample_id: Any,
) -> None:
    context = f"epoch={epoch}, step={step}, sample={sample_id}"
    invalid_parts = sorted(
        name for name, value in loss_parts.items() if not math.isfinite(float(value))
    )
    invalid_values = [] if bool(torch.isfinite(loss.detach()).all().item()) else ["total loss"]
    invalid_values.extend(f"{name} component" for name in invalid_parts)
    if invalid_values:
        raise FloatingPointError(f"Non-finite {', '.join(invalid_values)} at {context}")


def _clip_and_validate_gradients(
    model: torch.nn.Module,
    max_norm: float,
    *,
    epoch: int,
    step: int,
    sample_id: Any,
) -> float:
    """Clip gradients with one device synchronization for non-finite detection."""
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("train.grad_clip must be finite and greater than zero")
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=True
        )
    except RuntimeError as error:
        raise FloatingPointError(
            "Non-finite gradients after clipping validation at "
            f"epoch={epoch}, step={step}, sample={sample_id}"
        ) from error
    finite_norm = float(total_norm.detach().cpu())
    if not math.isfinite(finite_norm):
        raise FloatingPointError(
            "Non-finite gradient norm after clipping at "
            f"epoch={epoch}, step={step}, sample={sample_id}"
        )
    return finite_norm


def _validation_dataset(config: dict[str, Any]):
    return build_dataset(config["dataset"], split="val")


def _validation_protocol(
    config: dict[str, Any],
    val_sampling: dict[str, Any],
    train_dataset,
    val_dataset,
    digest_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    data = copy.deepcopy(config["dataset"])
    data.pop("root", None)
    data.pop("val_root", None)
    manifest_path = data.pop("split_manifest", None)
    manifest = load_eventhdr_split_manifest(manifest_path) if manifest_path else None
    if manifest is None:
        manifest_identity = None
    else:
        manifest_identity = {
            "status": str(manifest.get("status", "missing")).strip().lower(),
            "split_schema": manifest["split_schema"],
            "train_files": manifest["train_files"],
            "val_files": manifest["val_files"],
            "file_to_scene": manifest["file_to_scene"],
        }
        if manifest["split_schema"] == "physical_scenes_v1":
            manifest_identity.update(
                {
                    "scene_groups": manifest["scene_groups"],
                    "train_scenes": manifest["train_scenes"],
                    "val_scenes": manifest["val_scenes"],
                }
            )
    print("Verifying cached hashes or hashing train/validation files for exact resume...")
    return {
        "version": 5,
        "seed": int(config.get("seed", 2026)),
        "recurrent": bool(config["model"].get("recurrent", True)),
        "dataset_transform": data,
        "split_manifest": manifest_identity,
        "dataset_content": {
            "train": _dataset_content_fingerprint(train_dataset, digest_cache),
            "validation": _dataset_content_fingerprint(val_dataset, digest_cache),
        },
        "max_val_samples": config["train"].get("max_val_samples"),
        "sampling": val_sampling,
        "selection_metric": (
            "single_final_epoch_macro_ssim"
            if config["train"].get("validate_every", 1) is None
            else "macro_ssim"
        ),
        "ssim": "gaussian_valid_11_sigma1.5",
    }


def _enforce_training_split_status(config: dict[str, Any]) -> None:
    manifest_path = config.get("dataset", {}).get("split_manifest")
    if not manifest_path:
        return
    manifest = load_eventhdr_split_manifest(manifest_path)
    status = str(manifest.get("status", "missing")).strip().lower()
    if status != "final":
        raise ValueError(
            f"Training split manifest {manifest_path} has status='{status}', not 'final'. "
            "A provisional split cannot be used for training."
        )


@torch.no_grad()
def validate(
    model: ASGCNReconstructor,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
    score_positions: set[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator()
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    recurrent_state = None
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        if len(batch) != 1:
            raise ValueError("Stateful validation currently requires batch_size=1")
        sample = move_sample(batch[0], device)
        scene, sequence_index, sensor_size = _sample_sequence_info(sample)
        if not _continues_sequence(
            scene,
            sequence_index,
            sensor_size,
            current_scene,
            previous_sequence_index,
            previous_sensor_size,
        ):
            recurrent_state = None
        current_scene = scene
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        prediction, diagnostics = model.forward_sample(sample, recurrent_state=recurrent_state)
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if score_positions is None or index in score_positions:
            target = sample["target"].unsqueeze(0)
            accumulator.update(scene, sample["sample_id"], frame_metrics(prediction, target))
    return accumulator.summary()


def train(config: dict[str, Any], resume_from: str | Path | None = None) -> Path:
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    train_config = config["train"]
    _enforce_training_split_status(config)
    run_dir = Path(config["output"]["run_dir"])
    resume_path = resume_from or train_config.get("resume")
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        if resume_path.resolve().parent != run_dir.resolve():
            raise ValueError(
                "Exact resume must use a checkpoint inside the configured run_dir so "
                "the historical best.pt remains available."
            )
    else:
        existing_artifacts = [
            path
            for name in ("last.pt", "best.pt", "history.json", "config.json")
            if (path := run_dir / name).exists()
        ]
        if existing_artifacts:
            raise ValueError(
                f"Fresh training run_dir is not empty: {run_dir}. Use --resume with "
                "last.pt or choose a new output.run_dir; existing results are not overwritten."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
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
    val_indices = _balanced_contiguous_indices(
        val_dataset,
        train_config.get("max_val_samples"),
        seed=seed,
        require_all_groups=True,
    )
    val_sampling = _sampling_summary(val_dataset, val_indices)
    recurrent_validation = bool(config["model"].get("recurrent", True))
    validation_context_frames = train_config.get("validation_context_frames", 64)
    if validation_context_frames is not None:
        validation_context_frames = int(validation_context_frames)
        if validation_context_frames < 0:
            raise ValueError("train.validation_context_frames must be non-negative or null")
    if recurrent_validation:
        val_schedule, val_score_positions = _prefix_context_schedule(
            val_dataset,
            val_indices,
            max_context_frames=validation_context_frames,
        )
        context_policy = (
            "full_group_prefix" if validation_context_frames is None else "bounded_predecessor"
        )
    else:
        val_schedule = val_indices
        val_score_positions = set(range(len(val_indices)))
        context_policy = "none_non_recurrent"
    val_sampling.update(
        {
            "context_policy": context_policy,
            "max_context_frames_per_group": validation_context_frames
            if recurrent_validation
            else 0,
            "context_samples": len(val_schedule) - len(val_indices),
            "forward_samples": len(val_schedule),
        }
    )
    val_sampling_counts = _sampling_counts(val_sampling)
    hash_cache_path = run_dir / ".data_hash_cache.json"
    digest_cache = _load_data_hash_cache(
        hash_cache_path, bool(train_config.get("rehash_data", False))
    )
    validation_protocol = _validation_protocol(
        config, val_sampling, train_dataset, val_dataset, digest_cache
    )
    save_json(hash_cache_path, {"version": 1, "files": digest_cache})
    val_loader = _data_loader(
        Subset(val_dataset, val_schedule),
        1,
        int(train_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(train_config),
    )

    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        model, resume_checkpoint = load_model_checkpoint(resume_path, device, config["model"])
    else:
        model = build_model(config["model"]).to(device)
    optimizer_mode = _optimizer_mode(train_config)
    optimizer = _build_optimizer(model, train_config)
    scheduler = _build_scheduler(optimizer, train_config)
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    training_protocol = _training_protocol(config, device)
    scaler = _make_grad_scaler(amp_enabled)
    criterion = ReconstructionLoss(train_config.get("loss_weights"))
    temporal_weight = float(train_config.get("loss_weights", {}).get("temporal", 0.0))

    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_config") != config["model"]:
            raise ValueError(
                "Exact resume requires config.model to match the checkpoint model_config"
            )
        if resume_checkpoint.get("validation_protocol") != validation_protocol:
            raise ValueError(
                "Resume validation protocol differs from the checkpoint. Keep the seed, "
                "dataset transforms, split manifest, validation sampling, and SSIM protocol fixed."
            )
        _validate_training_protocol(resume_checkpoint, training_protocol)
        historical_score = _resume_best_macro_ssim(resume_checkpoint)
        historical_path = run_dir / "best.pt"
        if math.isfinite(historical_score):
            if not historical_path.is_file():
                raise ValueError(
                    f"Exact resume requires the historical best checkpoint: {historical_path}"
                )
            historical_model, historical_best = load_model_checkpoint(
                historical_path,
                torch.device("cpu"),
                config["model"],
            )
            computed_best_digest = _model_state_sha256(historical_model.state_dict())
            if historical_best.get("model_state_sha256") != computed_best_digest:
                raise ValueError("Historical best.pt tensor bytes do not match its digest")
            _validate_resume_best_pair(resume_checkpoint, historical_best)
            del historical_model
            del historical_best
        elif historical_path.exists():
            raise ValueError(
                "Resume checkpoint has no validated best score, but run_dir contains a "
                "best.pt from another or inconsistent run"
            )
    save_json(run_dir / "config.json", config)

    best_ssim = float("-inf")
    best_model_state_sha256: str | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_checkpoint is not None:
        if "optimizer" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has model weights but no optimizer state; "
                "it cannot be used for exact training resume"
            )
        optimizer.load_state_dict(resume_checkpoint.pop("optimizer"))
        _optimizer_to(optimizer, device)
        if "scheduler" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has no scheduler state/schema and cannot "
                "provide an exact training resume"
            )
        scheduler_state = resume_checkpoint.pop("scheduler")
        if scheduler is None:
            if scheduler_state is not None:
                raise ValueError("Resume checkpoint unexpectedly contains scheduler state")
        elif not isinstance(scheduler_state, dict):
            raise ValueError("Resume checkpoint is missing MultiStepLR scheduler state")
        else:
            scheduler.load_state_dict(scheduler_state)
        if "scaler" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has no GradScaler state and cannot provide "
                "an exact training resume"
            )
        scaler_state = resume_checkpoint.pop("scaler")
        if not isinstance(scaler_state, dict):
            raise ValueError("Resume checkpoint GradScaler state must be a dictionary")
        scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_ssim = _resume_best_macro_ssim(resume_checkpoint)
        best_model_state_sha256 = resume_checkpoint.get("best_model_state_sha256")
        history = list(resume_checkpoint.get("history", []))
        if "rng_state" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has no RNG state and cannot provide an exact resume"
            )
        _restore_rng_state(resume_checkpoint.pop("rng_state"))

    epochs = int(train_config.get("epochs", 40))
    raw_validate_every = train_config.get("validate_every", 1)
    validate_every = (
        None if raw_validate_every is None else max(1, int(raw_validate_every))
    )
    max_train_samples = train_config.get("max_train_samples")
    for epoch in range(start_epoch, epochs + 1):
        epoch_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        model.train()
        _reset_cuda_peak_memory(device)
        current_scene = None
        previous_sequence_index = None
        previous_sensor_size = None
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
            scene, sequence_index, sensor_size = _sample_sequence_info(sample)
            if not _continues_sequence(
                scene,
                sequence_index,
                sensor_size,
                current_scene,
                previous_sequence_index,
                previous_sensor_size,
            ):
                recurrent_state = None
                previous_prediction = None
                previous_target = None
            current_scene = scene
            previous_sequence_index = sequence_index
            previous_sensor_size = sensor_size
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
            _ensure_finite_loss(
                loss,
                loss_parts,
                epoch=epoch,
                step=step,
                sample_id=sample.get("sample_id", "unknown"),
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if optimizer_mode == "adam_gc":
                _centralize_gradients(model)
            _clip_and_validate_gradients(
                model,
                float(train_config.get("grad_clip", 1.0)),
                epoch=epoch,
                step=step,
                sample_id=sample.get("sample_id", "unknown"),
            )
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

        should_validate = epoch == epochs or (
            validate_every is not None and epoch % validate_every == 0
        )
        val_metrics = (
            validate(
                model,
                val_loader,
                device,
                max_samples=None,
                score_positions=val_score_positions,
            )
            if should_validate
            else {}
        )
        train_mean_loss = running_loss / max(seen, 1)
        if not math.isfinite(train_mean_loss):
            raise FloatingPointError(
                f"Non-finite mean training loss at epoch={epoch}: {train_mean_loss}"
            )
        validation_ssim = _macro_ssim(val_metrics)
        if should_validate and not math.isfinite(validation_ssim):
            raise FloatingPointError(
                f"Non-finite validation macro SSIM at epoch={epoch}: {validation_ssim}"
            )
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val": val_metrics,
            "val_sampling": val_sampling_counts,
            "learning_rate": (
                epoch_learning_rates[0] if len(epoch_learning_rates) == 1 else epoch_learning_rates
            ),
            "gpu_memory": _cuda_peak_memory(device),
        }
        history.append(record)
        save_json(run_dir / "history.json", history)
        model_state = model.state_dict()
        model_state_sha256 = _model_state_sha256(model_state)
        checkpoint = {
            "checkpoint_type": "training",
            "epoch": epoch,
            "model": model_state,
            "model_state_sha256": model_state_sha256,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "model_config": (
                resume_checkpoint.get("model_config", config["model"])
                if resume_checkpoint is not None
                else config["model"]
            ),
            "config": config,
            "val": val_metrics,
            "val_sampling": val_sampling_counts,
            "best_ssim": best_ssim,
            "best_model_state_sha256": best_model_state_sha256,
            "best_metric": "macro_ssim",
            "checkpoint_selection": (
                "single_final_epoch"
                if validate_every is None
                else "best_validation_macro_ssim"
            ),
            "paper_core_version": PAPER_CORE_VERSION,
            "validation_protocol": validation_protocol,
            "training_protocol": training_protocol,
            "history": history,
            "rng_state": _capture_rng_state(),
        }
        if validation_ssim > best_ssim:
            best_ssim = validation_ssim
            best_model_state_sha256 = model_state_sha256
            checkpoint["best_ssim"] = best_ssim
            checkpoint["best_model_state_sha256"] = best_model_state_sha256
            best_checkpoint = {
                "checkpoint_type": "ann_inference",
                "epoch": checkpoint["epoch"],
                "model": checkpoint["model"],
                "model_config": checkpoint["model_config"],
                "val": checkpoint["val"],
                "val_sampling": checkpoint["val_sampling"],
                "best_ssim": checkpoint["best_ssim"],
                "best_metric": checkpoint["best_metric"],
                "checkpoint_selection": checkpoint["checkpoint_selection"],
                "model_state_sha256": best_model_state_sha256,
                "paper_core_version": checkpoint["paper_core_version"],
                "validation_protocol": checkpoint["validation_protocol"],
                "training_protocol": checkpoint["training_protocol"],
            }
            atomic_torch_save(best_checkpoint, run_dir / "best.pt")
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        print(record)
    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(
            "Training completed without a best.pt. Check that macro SSIM is finite and "
            "validation ran successfully."
        )
    return best_path


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
    snn_dynamics: str | None = None,
) -> dict[str, Any]:
    _validate_snn_request(inference_mode, simulation_steps)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    eval_config = config.get("eval", {})
    eval_batch_size = int(eval_config.get("batch_size", 1))
    if eval_batch_size != 1:
        raise ValueError("Stateful evaluation requires eval.batch_size=1")
    loader = _data_loader(
        dataset,
        eval_batch_size,
        int(eval_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(eval_config),
    )
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(inference_mode, simulation_steps, checkpoint, checkpoint_path)
    _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
    model.eval()
    lpips_model = _maybe_lpips(bool(eval_config.get("lpips", False)), device)
    _reset_cuda_peak_memory(device)
    accumulator = MetricAccumulator()
    frame_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    realtime_factors: list[float] = []
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    recurrent_state = None
    previous_prediction = None
    previous_target = None
    output_base = Path(eval_config.get("output_dir", "runs/evaluation"))
    run_label = _inference_run_label(
        inference_mode,
        simulation_steps,
        model.snn_dynamics,
    )
    output_dir = output_base / run_label
    protected_outputs = (
        output_dir / "metrics.json",
        output_dir / "frames.csv",
        output_dir / "predictions",
    )
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(
            f"Evaluation output already exists for {run_label}: {output_dir}. "
            "Move/remove that run or choose a new eval.output_dir; results are never "
            "silently overwritten."
        )
    save_limit = int(eval_config.get("save_predictions", 0))
    max_samples = eval_config.get("max_samples")
    saved = 0
    prediction_stems: set[str] = set()
    for index, batch in enumerate(tqdm(loader, desc=f"evaluate-{inference_mode}")):
        if max_samples is not None and index >= int(max_samples):
            break
        sample = move_sample(batch[0], device)
        scene, sequence_index, sensor_size = _sample_sequence_info(sample)
        if not _continues_sequence(
            scene,
            sequence_index,
            sensor_size,
            current_scene,
            previous_sequence_index,
            previous_sensor_size,
        ):
            recurrent_state = None
            previous_prediction = None
            previous_target = None
        current_scene = scene
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
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
        temporal_l1 = None
        metrics = frame_metrics(prediction, target, lpips_model)
        if previous_prediction is not None and previous_target is not None:
            temporal_l1 = float(
                temporal_consistency_error(
                    prediction,
                    previous_prediction,
                    target,
                    previous_target,
                ).cpu()
            )
            metrics["temporal_l1"] = temporal_l1
        previous_prediction = prediction.detach()
        previous_target = target.detach()
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
            "temporal_l1": temporal_l1,
            "raw_events": int(sample["metadata"].get("raw_event_count", sample["events"].shape[0])),
            "cropped_events": int(
                sample["metadata"].get("cropped_event_count", sample["events"].shape[0])
            ),
            "retained_events": int(sample["events"].shape[0]),
            "events": int(sample["events"].shape[0]),
            "dataset_sampling_ratio": diagnostics["dataset_sampling_ratio"],
            "model_sampling_factor": diagnostics["event_sampling_factor"],
            "effective_sampling_ratio": diagnostics["effective_sampling_ratio"],
            "nodes": diagnostics["nodes"],
            "edges": diagnostics["edges"],
            "isolated_nodes": int(diagnostics["isolated_nodes"]),
            "isolate_ratio": float(diagnostics["isolate_ratio"]),
            "max_degree": int(diagnostics["max_degree"]),
        }
        frame_rows.append(row)
        latencies.append(latency_ms)
        if saved < save_limit:
            safe_name = _prediction_artifact_stem(sample["sample_id"], index)
            if safe_name in prediction_stems:
                raise RuntimeError(f"Duplicate prediction artifact stem: {safe_name}")
            prediction_stems.add(safe_name)
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
        "dataset_coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "output_dir": str(output_dir),
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "snn_dynamics": model.snn_dynamics if inference_mode == "snn" else None,
        "graph_topology": {
            "isolate_ratio": (
                sum(row["isolated_nodes"] for row in frame_rows)
                / sum(row["nodes"] for row in frame_rows)
                if sum(row["nodes"] for row in frame_rows) > 0
                else None
            ),
            "max_degree": max((row["max_degree"] for row in frame_rows), default=0),
        },
        "quality": quality,
        "latency": latency,
        "gpu_memory": _cuda_peak_memory(device),
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


def _sample_event_counts(sample: dict[str, Any]) -> tuple[int, int]:
    """Return raw/source and retained counts, tolerating custom dataset metadata."""
    retained = int(sample["events"].shape[0])
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        return retained, retained
    value = metadata.get("raw_event_count")
    try:
        raw = int(value)
    except (TypeError, ValueError, OverflowError):
        raw = retained
    if isinstance(value, bool) or raw < retained:
        raw = retained
    return raw, retained


def _dataset_coverage_summary(dataset, data_config: dict[str, Any]) -> dict[str, Any]:
    dataset_type = data_config["type"]
    if dataset_type == "eventhdr":
        root = Path(dataset.root)
        files = sorted(path.relative_to(root).as_posix() for path in dataset.files)
        mapping = getattr(dataset, "file_to_scene", {})
        declared_semantics = getattr(dataset, "group_semantics", None)
        if declared_semantics == "h5_sequence_file_not_physical_scene":
            grouping = "source_h5_sequence_file"
        elif declared_semantics == "physical_scene":
            grouping = "physical_scene"
        else:
            grouping = (
                "physical_scene"
                if any(mapping.get(file_key) != file_key for file_key in files)
                else "source_h5_file"
            )
    elif dataset_type == "eventaid_r_zip":
        files = sorted(path.name for path in dataset.zip_paths)
        grouping = "eventaid_scene_zip"
    else:
        files = []
        grouping = "unknown"
    expected = data_config.get("expected_file_count")
    return {
        "file_count": len(files),
        "expected_file_count": int(expected) if expected is not None else None,
        "complete": expected is None or len(files) == int(expected),
        "files": files,
        "quality_grouping": grouping,
        "target_offset": (
            int(data_config.get("target_offset", 1)) if dataset_type == "eventaid_r_zip" else None
        ),
    }


@torch.no_grad()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warmup: int = 10,
    steps: int = 100,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
    snn_dynamics: str | None = None,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _validate_snn_request(inference_mode, simulation_steps)
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(inference_mode, simulation_steps, checkpoint, checkpoint_path)
    _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
    model.eval()
    cuda_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    cuda_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    latencies: list[float] = []
    raw_event_counts: list[int] = []
    retained_event_counts: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    isolated_node_counts: list[int] = []
    max_degrees: list[int] = []
    layer_spike_totals: list[float] = []
    layer_neuron_step_totals: list[int] = []
    realtime_factors: list[float] = []
    recurrent_state = None
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    measured_state_resets = 0
    seed = int(config.get("seed", 2026))
    recurrent = model.decoder.recurrent is not None
    warmup_indices = _representative_schedule(dataset, warmup, seed, contiguous=False)
    measured_indices = _representative_schedule(dataset, steps, seed + 1, contiguous=recurrent)
    measured_schedule: list[tuple[bool, int]] = []
    context_frames = 0
    benchmark_context_frames = config.get("eval", {}).get("recurrent_context_frames", 32)
    if benchmark_context_frames is not None:
        benchmark_context_frames = int(benchmark_context_frames)
        if benchmark_context_frames < 0:
            raise ValueError("eval.recurrent_context_frames must be non-negative or null")
    if recurrent:
        # ``_representative_schedule`` cycles only when steps exceed the dataset.
        # Build each cycle separately so the prefix helper always receives unique indices.
        for offset in range(0, len(measured_indices), len(dataset)):
            chunk = measured_indices[offset : offset + len(dataset)]
            context_indices, score_positions = _prefix_context_schedule(
                dataset,
                chunk,
                max_context_frames=benchmark_context_frames,
            )
            context_frames += len(context_indices) - len(chunk)
            measured_schedule.extend(
                (position in score_positions, index)
                for position, index in enumerate(context_indices)
            )
    else:
        measured_schedule = [(True, index) for index in measured_indices]
    schedule = [(False, index) for index in warmup_indices] + measured_schedule
    for iteration, (measured, sample_index) in enumerate(schedule):
        if iteration == len(warmup_indices):
            recurrent_state = None
            current_scene = None
            previous_sequence_index = None
            previous_sensor_size = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
        raw = dataset[sample_index]  # I/O intentionally outside the timer.
        sample = move_sample(raw, device)
        scene, sequence_index, sensor_size = _sample_sequence_info(sample)
        continuation = _continues_sequence(
            scene,
            sequence_index,
            sensor_size,
            current_scene,
            previous_sequence_index,
            previous_sensor_size,
        )
        if not continuation:
            recurrent_state = None
            if measured:
                measured_state_resets += 1
        current_scene = scene
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        if measured:
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
        elapsed_ms = None
        if measured:
            if cuda_end is not None:
                cuda_end.record()
                cuda_end.synchronize()
                elapsed_ms = float(cuda_start.elapsed_time(cuda_end))
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if measured:
            assert elapsed_ms is not None
            latencies.append(elapsed_ms)
            raw_event_count, retained_event_count = _sample_event_counts(sample)
            raw_event_counts.append(raw_event_count)
            retained_event_counts.append(retained_event_count)
            node_counts.append(int(diagnostics["nodes"]))
            edge_counts.append(int(diagnostics["edges"]))
            isolated_node_counts.append(int(diagnostics["isolated_nodes"]))
            max_degrees.append(int(diagnostics["max_degree"]))
            spike_counts = diagnostics["spike_counts"]
            neuron_steps = diagnostics["firing_rate_denominators"]
            if spike_counts:
                if not layer_spike_totals:
                    layer_spike_totals = [0.0] * len(spike_counts)
                    layer_neuron_step_totals = [0] * len(neuron_steps)
                if len(spike_counts) != len(layer_spike_totals):
                    raise RuntimeError("SNN firing-stat layer count changed during benchmark")
                for layer_index, (spikes, denominator) in enumerate(
                    zip(spike_counts, neuron_steps, strict=True)
                ):
                    layer_spike_totals[layer_index] += (
                        float(spikes.detach().cpu()) if torch.is_tensor(spikes) else float(spikes)
                    )
                    layer_neuron_step_totals[layer_index] += int(denominator)
            metadata = sample.get("metadata", {})
            dt_us = metadata.get("dt_us") if isinstance(metadata, dict) else None
            if dt_us:
                realtime_factors.append(elapsed_ms / (float(dt_us) / 1000.0))
    elapsed_seconds = sum(latencies) / 1000.0
    raw_events_per_second = sum(raw_event_counts) / elapsed_seconds
    retained_events_per_second = sum(retained_event_counts) / elapsed_seconds
    graph_nodes_per_second = sum(node_counts) / elapsed_seconds
    total_raw_events = sum(raw_event_counts)
    total_neuron_steps = sum(layer_neuron_step_totals)
    layer_firing_rates = [
        spikes / neuron_steps if neuron_steps > 0 else None
        for spikes, neuron_steps in zip(
            layer_spike_totals,
            layer_neuron_step_totals,
            strict=True,
        )
    ]
    result: dict[str, Any] = {
        **_latency_summary(latencies),
        "raw_events_per_second": raw_events_per_second,
        "retained_events_per_second": retained_events_per_second,
        "graph_nodes_per_second": graph_nodes_per_second,
        # Deprecated compatibility alias; new consumers should use the retained rate.
        "events_per_second": retained_events_per_second,
        "mean_raw_events": statistics.fmean(raw_event_counts),
        "mean_retained_events": statistics.fmean(retained_event_counts),
        "retention_ratio": (
            sum(retained_event_counts) / total_raw_events if total_raw_events > 0 else None
        ),
        "mean_nodes": statistics.fmean(node_counts),
        "mean_edges": statistics.fmean(edge_counts),
        "mean_isolated_nodes": statistics.fmean(isolated_node_counts),
        "isolate_ratio": (
            sum(isolated_node_counts) / sum(node_counts) if sum(node_counts) > 0 else None
        ),
        "max_degree": max(max_degrees, default=0),
        "layer_firing_rates": layer_firing_rates or None,
        "mean_firing_rate": (
            sum(layer_spike_totals) / total_neuron_steps if total_neuron_steps > 0 else None
        ),
        "deadline_miss_ratio": (
            sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
            if realtime_factors
            else None
        ),
        "rtf_p95": percentile(realtime_factors, 0.95) if realtime_factors else None,
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "snn_dynamics": model.snn_dynamics if inference_mode == "snn" else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
        "gpu_memory": _cuda_peak_memory(device),
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter",
        "io_excluded": True,
        "dataset_coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        "sampling": _sampling_summary(dataset, measured_indices),
        "warmup_frames": len(warmup_indices),
        "recurrent_context_policy": (
            "full_group_prefix"
            if recurrent and benchmark_context_frames is None
            else "bounded_predecessor"
            if recurrent
            else None
        ),
        "max_recurrent_context_frames_per_group": benchmark_context_frames if recurrent else 0,
        "recurrent_context_frames": context_frames,
        "state_resets": measured_state_resets,
        "state_reset_ratio": measured_state_resets / len(measured_indices),
    }
    benchmark_base = Path(config.get("eval", {}).get("output_dir", "runs/evaluation"))
    benchmark_dir = benchmark_base / _inference_run_label(
        inference_mode,
        simulation_steps,
        model.snn_dynamics,
    )
    benchmark_path = benchmark_dir / "benchmark.json"
    if benchmark_path.exists():
        raise FileExistsError(
            f"Benchmark output already exists: {benchmark_path}. Move/remove the prior "
            "artifact or choose a new eval.output_dir."
        )
    result["output_path"] = str(benchmark_path)
    save_json(benchmark_path, result)
    return result


@torch.no_grad()
def calibrate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples: int | None = None,
    overwrite: bool = False,
) -> Path:
    if samples is not None and int(samples) < 1:
        raise ValueError("calibration samples must be at least 1")
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("ANN input and calibrated SNN output must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Calibrated checkpoint already exists: {output_path}. Move it or choose a new "
            "output path, or explicitly request overwrite."
        )
    _enforce_training_split_status(config)
    device = resolve_device(config.get("device", "auto"))
    data_config = copy.deepcopy(config["dataset"])
    # Calibration is restricted to EventHDR train, never EventAid-R.
    if data_config["type"] != "eventhdr":
        raise ValueError("SNN calibration must use EventHDR training data")
    dataset = build_dataset(data_config, split="calibration")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    model.fold_batch_norm()
    model.reset_activation_maxima()
    calibration_limit = len(dataset) if samples is None else min(int(samples), len(dataset))
    calibration_indices = _balanced_sample_indices(
        dataset, calibration_limit, seed=int(config.get("seed", 2026))
    )
    try:
        calibration_sampling = _sampling_summary(dataset, calibration_indices)
        for index in tqdm(calibration_indices, desc="calibrate-SNN"):
            sample = move_sample(dataset[index], device)
            model.calibrate_sample(sample, momentum=-1.0)
        calibration_summary = model.calibration_summary()
        model.apply_parameter_normalization()
        model_state = model.state_dict()
        inference_checkpoint = {
            "checkpoint_type": "snn_inference",
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": checkpoint.get("model_config", config["model"]),
            "epoch": checkpoint.get("epoch"),
            "source_checkpoint": str(checkpoint_path),
            "batch_norm_folded": True,
            "snn_calibrated": True,
            "paper_core_version": PAPER_CORE_VERSION,
            "parameter_normalized": True,
            "snn_calibration_samples": len(calibration_indices),
            "snn_calibration_valid_samples": calibration_summary["minimum_valid_samples"],
            "snn_calibration_summary": calibration_summary,
            "snn_calibration_sampling": calibration_sampling,
        }
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    atomic_torch_save(inference_checkpoint, output_path)
    return output_path
~~~~~~~~

# src/asgcn_recon/graph.py

~~~~~~~~python
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

PAPER_CORE_VERSION = 2


@dataclass
class EventGraph:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


def _safe_batch_norm(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
    """Use running statistics when a graph has fewer than two events."""
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


def uniformly_sample_events(events: torch.Tensor, factor: int = 1) -> torch.Tensor:
    """Apply the paper's deterministic event sampling factor R."""
    if isinstance(factor, bool) or int(factor) != factor:
        raise ValueError("event_sampling_factor must be an integer")
    factor = int(factor)
    if factor < 1:
        raise ValueError("event_sampling_factor must be at least 1")
    return events[::factor]


def prepare_event_nodes(
    events: torch.Tensor, sensor_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Eq. (9)-(10) event nodes while retaining temporal order.

    The paper does not specify coordinate normalization. This implementation records
    the explicit assumption x,y,t in [0,1] and polarity in {-1,+1}. Graph distances
    use x,y,t by default; polarity remains a node feature.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Events must have shape [N,4] with x,y,t,p columns")
    height, width = (int(value) for value in sensor_size)
    if height < 1 or width < 1:
        raise ValueError("sensor_size must contain positive height and width")
    if events.numel() == 0:
        return (
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
        )
    events = events.float()
    if not bool(torch.isfinite(events).all()):
        raise ValueError("Event coordinates, timestamps, and polarities must be finite")
    if bool((events[1:, 2] < events[:-1, 2]).any()):
        raise ValueError("Event timestamps must be monotonically non-decreasing")
    x = events[:, 0] / max(width - 1, 1)
    y = events[:, 1] / max(height - 1, 1)
    t = events[:, 2]
    t = (t - t[0]) / (t[-1] - t[0]).abs().clamp_min(1e-6)
    polarity = torch.where(events[:, 3] > 0, 1.0, -1.0)
    polarity_position = (polarity + 1.0) * 0.5
    positions = torch.stack((x, y, t, polarity_position), dim=-1)
    node_features = torch.stack((x, y, t, polarity), dim=-1)
    return node_features, positions


def build_radius_graph(
    positions: torch.Tensor,
    radius: float,
    *,
    position_dims: int = 3,
    chunk_size: int = 512,
    max_edges: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the paper's exact radius graph with a uniform-cell candidate search.

    Every ordered edge direction is materialized so source-to-target aggregation is
    equivalent to an undirected graph.  Cells have width ``radius``; therefore only
    the 3^d adjacent cells can contain a valid neighbor.  Exact Euclidean filtering
    after candidate generation preserves the brute-force graph while avoiding the
    O(N^2) distance matrix on sparse event volumes.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("graph_radius must be positive")
    position_dims = int(position_dims)
    if position_dims < 1 or position_dims > positions.shape[1]:
        raise ValueError("graph_position_dims must select available position columns")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("graph_chunk_size must be at least 1")
    if max_edges is not None:
        if isinstance(max_edges, bool) or int(max_edges) != max_edges:
            raise ValueError("max_graph_edges must be an integer or null")
        max_edges = int(max_edges)
        if max_edges < 1:
            raise ValueError("max_graph_edges must be at least 1 or null")

    count = int(positions.shape[0])
    device = positions.device
    if count == 0:
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=positions.dtype),
        )

    coordinates = positions[:, :position_dims]
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Graph coordinates must be finite")
    if bool(((coordinates < 0) | (coordinates > 1)).any()):
        raise ValueError("Normalized graph coordinates must lie in [0,1]")

    cells_per_axis = max(2, math.ceil(1.0 / radius) + 1)
    if cells_per_axis**position_dims >= torch.iinfo(torch.long).max:
        raise ValueError("graph_radius is too small for integer spatial hashing")
    strides = torch.tensor(
        [cells_per_axis**dimension for dimension in range(position_dims)],
        device=device,
        dtype=torch.long,
    )
    cells = torch.floor(coordinates / radius).to(torch.long)
    cells = cells.clamp_(0, cells_per_axis - 1)
    cell_hash = (cells * strides).sum(dim=1)
    sorted_hash, sorted_nodes = torch.sort(cell_hash)
    offset_axis = torch.tensor((-1, 0, 1), device=device, dtype=torch.long)
    offsets = torch.cartesian_prod(*([offset_axis] * position_dims)).reshape(
        -1, position_dims
    )

    sources: list[torch.Tensor] = []
    destination_chunks: list[torch.Tensor] = []
    distances_kept: list[torch.Tensor] = []
    retained_edge_count = 0
    # Bound worst-case candidate materialization even if every event occupies one cell.
    effective_chunk_size = min(chunk_size, max(1, 4_000_000 // count))
    for start in range(0, count, effective_chunk_size):
        stop = min(start + effective_chunk_size, count)
        local_sources = torch.arange(start, stop, device=device, dtype=torch.long)
        neighbor_cells = cells[start:stop, None, :] + offsets[None, :, :]
        valid_cells = ((neighbor_cells >= 0) & (neighbor_cells < cells_per_axis)).all(dim=2)
        neighbor_hashes = (neighbor_cells * strides).sum(dim=2)
        candidate_sources = local_sources[:, None].expand_as(neighbor_hashes)[valid_cells]
        candidate_hashes = neighbor_hashes[valid_cells]
        left = torch.searchsorted(sorted_hash, candidate_hashes, right=False)
        right = torch.searchsorted(sorted_hash, candidate_hashes, right=True)
        counts = right - left
        nonempty = counts > 0
        if not bool(nonempty.any()):
            continue
        candidate_sources = candidate_sources[nonempty]
        left = left[nonempty]
        counts = counts[nonempty]
        candidate_count = int(counts.sum().item())
        expanded_sources = torch.repeat_interleave(
            candidate_sources, counts, output_size=candidate_count
        )
        expanded_left = torch.repeat_interleave(left, counts, output_size=candidate_count)
        starts = counts.cumsum(0) - counts
        within_group = torch.arange(candidate_count, device=device) - torch.repeat_interleave(
            starts, counts, output_size=candidate_count
        )
        candidate_destinations = sorted_nodes[expanded_left + within_group]
        candidate_distances = torch.linalg.vector_norm(
            coordinates[expanded_sources] - coordinates[candidate_destinations], dim=1
        )
        within_radius = (expanded_sources != candidate_destinations) & (
            candidate_distances < radius
        )
        chunk_edge_count = int(within_radius.sum().item())
        if (
            max_edges is not None
            and retained_edge_count + chunk_edge_count > max_edges
        ):
            raise RuntimeError(
                "Radius graph exceeded max_graph_edges="
                f"{max_edges:,} while processing {count:,} nodes. Reduce "
                "graph_radius/max_events or raise the explicit memory guard after "
                "measuring accelerator memory."
            )
        if chunk_edge_count == 0:
            continue
        retained_edge_count += chunk_edge_count
        sources.append(expanded_sources[within_radius])
        destination_chunks.append(candidate_destinations[within_radius])
        distances_kept.append(candidate_distances[within_radius])

    if not sources:
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=positions.dtype),
        )
    source = torch.cat(sources)
    destination = torch.cat(destination_chunks)
    distance = torch.cat(distances_kept)
    order = torch.argsort(source * count + destination)
    source = source[order]
    destination = destination[order]
    distance = distance[order]
    edge_index = torch.stack((source, destination), dim=0)
    edge_attr = (distance / radius).clamp(0.0, 1.0).unsqueeze(-1)
    return edge_index, edge_attr


def build_event_graph(
    events: torch.Tensor,
    sensor_size: tuple[int, int],
    *,
    event_sampling_factor: int,
    graph_radius: float,
    graph_position_dims: int,
    graph_chunk_size: int,
    max_graph_edges: int | None = None,
) -> EventGraph:
    sampled = uniformly_sample_events(events, event_sampling_factor)
    node_features, positions = prepare_event_nodes(sampled, sensor_size)
    edge_index, edge_attr = build_radius_graph(
        positions,
        graph_radius,
        position_dims=graph_position_dims,
        chunk_size=graph_chunk_size,
        max_edges=max_graph_edges,
    )
    return EventGraph(node_features, positions, edge_index, edge_attr)


def linear_open_bspline_basis(
    pseudo: torch.Tensor, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return indices and weights for an open degree-1 B-spline basis.

    For each scalar pseudo-coordinate this yields the two non-zero basis terms.
    The weights form a partition of unity, including exact endpoints.
    """
    kernel_size = int(kernel_size)
    if kernel_size < 2:
        raise ValueError("spline_kernel_size must be at least 2")
    if pseudo.ndim != 2 or pseudo.shape[1] != 1:
        raise ValueError("ASGCN edge pseudo-coordinates must have shape [E,1]")
    if not torch.isfinite(pseudo).all():
        raise ValueError("ASGCN edge pseudo-coordinates must be finite")
    if bool(((pseudo < 0) | (pseudo > 1)).any()):
        raise ValueError("ASGCN edge pseudo-coordinates must lie in [0,1]")

    scaled = pseudo[:, 0] * float(kernel_size - 1)
    cell = torch.floor(scaled)
    left = cell.to(torch.long).remainder(kernel_size)
    right = (left + 1).remainder(kernel_size)
    right_weight = scaled - cell
    left_weight = 1.0 - right_weight
    indices = torch.stack((left, right), dim=-1)
    weights = torch.stack((left_weight, right_weight), dim=-1)
    return indices, weights


class PaperSplineConv(nn.Module):
    """Eq. (11) weighted open B-spline convolution followed by BN.

    ASGCN names a PyG weighted B-spline tensor-product kernel but omits its
    hyperparameters. The reconstruction adaptation therefore exposes them in every
    config. Degree 1 is implemented directly in PyTorch and matches the public
    SplineCNN operator definition for scalar pseudo-coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        degree: int = 1,
        root_weight: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if int(degree) != 1:
            raise ValueError("Only the configured open degree-1 B-spline is supported")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.degree = int(degree)
        if self.kernel_size < 2:
            raise ValueError("spline_kernel_size must be at least 2")
        self.weight = nn.Parameter(
            torch.empty(self.kernel_size, self.in_channels, self.out_channels)
        )
        self.root = (
            nn.Parameter(torch.empty(self.in_channels, self.out_channels)) if root_weight else None
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.norm = nn.BatchNorm1d(self.out_channels)
        self.register_buffer("bn_bypassed", torch.tensor(False), persistent=True)
        self.register_buffer("snn_normalized", torch.tensor(False), persistent=True)
        self.register_buffer("activation_max", torch.ones(self.out_channels), persistent=True)
        self.register_buffer("threshold", torch.ones(self.out_channels), persistent=True)
        self._bn_is_folded = False
        self._snn_is_normalized = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_bound = 1.0 / math.sqrt(self.kernel_size * self.in_channels)
        nn.init.uniform_(self.weight, -weight_bound, weight_bound)
        if self.root is not None:
            root_bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.root, -root_bound, root_bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.norm.reset_parameters()
        self.bn_bypassed.fill_(False)
        self.snn_normalized.fill_(False)
        self.activation_max.fill_(1.0)
        self.threshold.fill_(1.0)
        self._bn_is_folded = False
        self._snn_is_normalized = False

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
        self._snn_is_normalized = bool(self.snn_normalized.item())

    def spline_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        source, destination = edge_index
        output = torch.zeros((x.shape[0], self.out_channels), device=x.device, dtype=x.dtype)
        if source.numel() > 0:
            indices, basis = (
                basis_cache
                if basis_cache is not None
                else linear_open_bspline_basis(edge_attr, self.kernel_size)
            )
            # Project nodes once for every control point, then gather only the two
            # degree-1 basis terms that are active on each edge.
            projected = torch.einsum("ni,kio->nko", x, self.weight)
            for active_basis in range(2):
                messages = projected[source, indices[:, active_basis]]
                messages = messages * basis[:, active_basis, None].to(messages.dtype)
                # CPU autocast can produce bfloat16 projections while ``x``
                # (and therefore ``output``) remains float32. ``index_add_``
                # requires matching dtypes, so accumulate in the output dtype.
                output.index_add_(0, destination, messages.to(output.dtype))
            degree = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            degree.index_add_(
                0,
                destination,
                torch.ones((destination.numel(), 1), device=x.device, dtype=x.dtype),
            )
            output = output / degree.clamp_min(1.0)
        return output

    def affine(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output = self.spline_aggregate(x, edge_index, edge_attr, basis_cache)
        if self.root is not None:
            output = output + x @ self.root
        if self.bias is not None:
            output = output + self.bias
        return output

    def preactivation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output = self.affine(x, edge_index, edge_attr, basis_cache)
        return output if self._bn_is_folded else _safe_batch_norm(self.norm, output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = self.preactivation(x, edge_index, edge_attr, basis_cache)
        return torch.relu(preactivation), preactivation

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        """Fold Eq. (13) into every output term of Eq. (11), yielding Eq. (14)."""
        if self._bn_is_folded:
            return
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm")
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        self.weight.mul_(scale.view(1, 1, -1))
        if self.root is not None:
            self.root.mul_(scale.view(1, -1))
        if self.bias is None:
            raise RuntimeError("BN folding requires an affine convolution bias")
        self.bias.copy_((self.bias - self.norm.running_mean) * scale + self.norm.bias)
        self.bn_bypassed.fill_(True)
        self._bn_is_folded = True

    @torch.no_grad()
    def apply_parameter_normalization(
        self, input_scale: torch.Tensor, output_scale: torch.Tensor
    ) -> None:
        """Apply Eq. (6): W_l <- W_l lambda_(l-1)/lambda_l, b_l <- b_l/lambda_l."""
        if not self._bn_is_folded:
            raise RuntimeError("Fold BatchNorm before ANN-to-SNN parameter normalization")
        if self._snn_is_normalized:
            raise RuntimeError("ANN-to-SNN parameter normalization was already applied")
        input_scale = input_scale.to(device=self.weight.device, dtype=self.weight.dtype)
        output_scale = output_scale.to(device=self.weight.device, dtype=self.weight.dtype)
        if input_scale.shape != (self.in_channels,):
            raise ValueError("Input activation scale does not match spline input channels")
        if output_scale.shape != (self.out_channels,):
            raise ValueError("Output activation scale does not match spline output channels")
        input_scale = input_scale.clamp_min(1e-6)
        output_scale = output_scale.clamp_min(1e-6)
        self.weight.mul_(input_scale.view(1, -1, 1))
        self.weight.div_(output_scale.view(1, 1, -1))
        if self.root is not None:
            self.root.mul_(input_scale.view(-1, 1))
            self.root.div_(output_scale.view(1, -1))
        if self.bias is not None:
            self.bias.div_(output_scale)
        self.activation_max.copy_(output_scale)
        self.threshold.fill_(1.0)
        self.snn_normalized.fill_(True)
        self._snn_is_normalized = True


class ASGCNEncoder(nn.Module):
    """Equation-faithful ASGCN graph core adapted to a reconstruction decoder."""

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_layers: int = 6,
        *,
        spline_kernel_size: int = 5,
        spline_degree: int = 1,
        spline_root_weight: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        graph_layers = int(graph_layers)
        if graph_layers < 1:
            raise ValueError("graph_layers must be at least 1 for ASGCN")
        self.hidden_dim = hidden_dim
        channels = [4] + [hidden_dim] * graph_layers
        self.layers = nn.ModuleList(
            [
                PaperSplineConv(
                    channels[index],
                    channels[index + 1],
                    kernel_size=spline_kernel_size,
                    degree=spline_degree,
                    root_weight=spline_root_weight,
                    bias=True,
                )
                for index in range(graph_layers)
            ]
        )
        self.register_buffer(
            "calibration_samples_seen",
            torch.zeros(graph_layers, dtype=torch.long),
            persistent=True,
        )

    def _basis_cache(self, graph: EventGraph) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute the fixed edge basis once for all layers and IF timesteps."""
        if graph.edge_attr.shape[0] == 0:
            return None
        return linear_open_bspline_basis(
            graph.edge_attr,
            self.layers[0].kernel_size,
        )

    def forward_ann(
        self, graph: EventGraph, return_activations: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = graph.node_features
        activations: list[torch.Tensor] = []
        basis_cache = self._basis_cache(graph)
        for layer in self.layers:
            hidden, preactivation = layer(
                hidden,
                graph.edge_index,
                graph.edge_attr,
                basis_cache,
            )
            if return_activations:
                activations.append(torch.relu(preactivation))
        return hidden, activations

    def forward_snn(
        self,
        graph: EventGraph,
        simulation_steps: int = 16,
        dynamics: str = "literal_eq15",
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run explicit IF timesteps using literal Eq. (15) or a standard-IF control."""
        if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
            raise ValueError("simulation_steps must be an integer")
        simulation_steps = int(simulation_steps)
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be at least 1")
        if dynamics not in {"literal_eq15", "standard_if"}:
            raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
        if any(not layer._snn_is_normalized for layer in self.layers):
            raise RuntimeError("SNN inference requires Eq. (6) parameter normalization")
        node_count = int(graph.node_features.shape[0])
        if node_count == 0:
            empty = graph.node_features.new_empty((0, self.hidden_dim))
            zeros = [graph.node_features.new_zeros(()) for _ in self.layers]
            return empty, zeros

        membranes = [
            layer.threshold.to(graph.node_features).expand(node_count, -1).clone() * 0.5
            for layer in self.layers
        ]
        previous_spikes = [
            graph.node_features.new_zeros((node_count, layer.out_channels)) for layer in self.layers
        ]
        spike_sums = [torch.zeros_like(spikes) for spikes in previous_spikes]
        active_counts = [graph.node_features.new_zeros(()) for _ in self.layers]
        basis_cache = self._basis_cache(graph)

        for _ in range(simulation_steps):
            hidden = graph.node_features
            for index, layer in enumerate(self.layers):
                current = layer.affine(
                    hidden,
                    graph.edge_index,
                    graph.edge_attr,
                    basis_cache,
                )
                integrated = membranes[index] + current
                if dynamics == "literal_eq15":
                    # This is the paper's written +h_i^l(t-1) recurrence. It is
                    # intentionally separate from the standard rate-conversion IF
                    # control because the paper does not resolve their mismatch.
                    integrated = integrated + previous_spikes[index]
                threshold = layer.threshold.to(integrated).expand_as(integrated)
                spikes = torch.where(
                    integrated >= threshold, threshold, torch.zeros_like(integrated)
                )
                membranes[index] = integrated - spikes
                previous_spikes[index] = spikes
                spike_sums[index] = spike_sums[index] + spikes
                active_counts[index] = active_counts[index] + (spikes != 0).sum()
                hidden = spikes

        firing_rates = [
            active.to(graph.node_features.dtype) / float(simulation_steps * max(1, spikes.numel()))
            for active, spikes in zip(active_counts, spike_sums, strict=True)
        ]
        return spike_sums[-1] / float(simulation_steps), firing_rates

    @torch.no_grad()
    def update_activation_maxima(self, activations: list[torch.Tensor]) -> None:
        if len(activations) != len(self.layers):
            raise ValueError("Activation count does not match graph layer count")
        for index, (layer, activation) in enumerate(zip(self.layers, activations, strict=True)):
            if activation.numel() == 0:
                continue
            if activation.ndim != 2 or activation.shape[1] != layer.out_channels:
                raise ValueError("Calibration activation shape does not match graph layer")
            if not bool(torch.isfinite(activation).all()):
                raise FloatingPointError("Non-finite activation encountered during calibration")
            maxima = activation.amax(dim=0)
            layer.activation_max.copy_(torch.maximum(layer.activation_max, maxima))
            self.calibration_samples_seen[index].add_(1)

    @torch.no_grad()
    def reset_activation_maxima(self) -> None:
        for layer in self.layers:
            layer.activation_max.zero_()
        self.calibration_samples_seen.zero_()

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.eval()
        for layer in self.layers:
            layer.fold_batch_norm()

    @torch.no_grad()
    def apply_parameter_normalization(self) -> None:
        missing = torch.nonzero(self.calibration_samples_seen == 0).flatten().tolist()
        if missing:
            raise RuntimeError(
                "Cannot apply Eq. (6): no non-empty calibration activation for layer(s) "
                + ", ".join(str(index) for index in missing)
            )
        previous_scale = self.layers[0].weight.new_ones(self.layers[0].in_channels)
        for layer in self.layers:
            measured = layer.activation_max.detach().clone()
            # A ReLU channel that stayed identically zero has no usable lambda.
            # Keep it at unit scale instead of dividing its parameters by epsilon.
            output_scale = torch.where(measured > 0, measured, torch.ones_like(measured))
            layer.apply_parameter_normalization(previous_scale, output_scale)
            previous_scale = output_scale

    def calibration_summary(self) -> dict[str, list[int] | int]:
        dead_channels = [int((layer.activation_max <= 0).sum().item()) for layer in self.layers]
        valid_samples = [int(value) for value in self.calibration_samples_seen.tolist()]
        return {
            "valid_samples_per_layer": valid_samples,
            "minimum_valid_samples": min(valid_samples, default=0),
            "dead_channels_per_layer": dead_channels,
        }

    def output_activation_scale(self, reference: torch.Tensor) -> torch.Tensor:
        """Return lambda_L used to express spikes in the analog decoder's units."""
        if not self.layers[-1]._snn_is_normalized:
            raise RuntimeError("Output activation scale is available after Eq. (6) conversion")
        return self.layers[-1].activation_max.to(reference)
~~~~~~~~

# src/asgcn_recon/losses.py

~~~~~~~~python
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
        values: dict[str, Any] = {
            name: float(value.detach().cpu()) for name, value in terms.items()
        }
        values["total"] = float(total.detach().cpu())
        return total, values
~~~~~~~~

# src/asgcn_recon/metrics.py

~~~~~~~~python
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
    """Compute mean SSIM with the standard Gaussian local-statistics window.

    The default 11x11 window and sigma of 1.5 follow the original SSIM
    formulation.  For images smaller than the requested window, the largest
    fitting odd window is used so that validation crops of any positive size
    remain supported.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4:
        raise ValueError("prediction and target must be BCHW tensors")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    min_side = min(prediction.shape[-2:])
    if min_side < 1:
        raise ValueError("prediction and target must have non-empty spatial dimensions")
    size = min(window_size, min_side)
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)

    # CUDA autocast commonly produces float16 predictions against float32 targets.
    # Promote local statistics to at least float32 while preserving gradients.
    computation_dtype = torch.promote_types(prediction.dtype, target.dtype)
    if computation_dtype in {torch.float16, torch.bfloat16}:
        computation_dtype = torch.float32
    # An outer training autocast context would otherwise cast conv2d back to
    # float16 after the explicit promotion above, defeating the stabilization.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = prediction.to(dtype=computation_dtype)
        target = target.to(dtype=computation_dtype)

        # Build the Gaussian in the same stable computation dtype.
        kernel_dtype = torch.float64 if computation_dtype == torch.float64 else torch.float32
        coordinates = torch.arange(size, dtype=kernel_dtype, device=prediction.device)
        coordinates = coordinates - (size - 1) / 2
        gaussian_1d = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d).to(dtype=computation_dtype)
        channels = prediction.shape[1]
        window = gaussian_2d.expand(channels, 1, size, size).contiguous()

        def local_mean(value: torch.Tensor) -> torch.Tensor:
            return F.conv2d(value, window, groups=channels)

        mu_x = local_mean(prediction)
        mu_y = local_mean(target)
        sigma_x = local_mean(prediction * prediction) - mu_x.square()
        sigma_y = local_mean(target * target) - mu_y.square()
        sigma_xy = local_mean(prediction * target) - mu_x * mu_y
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
        minimum = torch.finfo(denominator.dtype).tiny
        return (numerator / denominator.clamp_min(minimum)).mean().clamp(-1.0, 1.0)


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=mse.device) / mse.clamp_min(1e-12))


def temporal_consistency_error(
    prediction: torch.Tensor,
    previous_prediction: torch.Tensor,
    target: torch.Tensor,
    previous_target: torch.Tensor,
) -> torch.Tensor:
    """Measure L1 error between predicted and target frame-to-frame changes.

    This is a no-flow temporal consistency diagnostic. It is intentionally named
    ``temporal_l1`` in reports so it cannot be confused with a flow-warped metric.
    """

    shapes = {
        prediction.shape,
        previous_prediction.shape,
        target.shape,
        previous_target.shape,
    }
    if len(shapes) != 1 or prediction.ndim != 4:
        raise ValueError("all temporal metric inputs must have the same BCHW shape")
    predicted_change = prediction - previous_prediction
    target_change = target - previous_target
    return F.l1_loss(predicted_change, target_change)


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
        names = sorted(
            {key for frame in self.frames for key in frame if key not in {"scene", "sample_id"}}
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for frame in self.frames:
            grouped[str(frame["scene"])].append(frame)
        per_scene: dict[str, dict[str, float | int]] = {}
        for scene, items in grouped.items():
            scene_summary: dict[str, float | int] = {"frames": len(items)}
            for name in names:
                values = [float(item[name]) for item in items if name in item]
                if values:
                    scene_summary[name] = sum(values) / len(values)
                    scene_summary[f"{name}_frames"] = len(values)
            per_scene[scene] = scene_summary
        micro = {}
        for name in names:
            values = [float(item[name]) for item in self.frames if name in item]
            if values:
                micro[name] = sum(values) / len(values)
        macro = {
            name: sum(float(scene[name]) for scene in per_scene.values() if name in scene)
            / sum(1 for scene in per_scene.values() if name in scene)
            for name in names
            if any(name in scene for scene in per_scene.values())
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
~~~~~~~~

# src/asgcn_recon/model.py

~~~~~~~~python
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .graph import PAPER_CORE_VERSION, ASGCNEncoder, EventGraph, build_event_graph


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
        torch.ones((linear.numel(), 1), device=features.device, dtype=features.dtype),
    )
    raster = raster / counts.clamp_min(1.0)
    return raster.transpose(0, 1).reshape(1, features.shape[-1], grid_h, grid_w)


class ASGCNReconstructor(nn.Module):
    def __init__(
        self,
        architecture_version: int = PAPER_CORE_VERSION,
        graph_operator: str = "spline",
        spline_backend: str = "torch",
        spline_pseudo: str = "distance_over_radius",
        spline_is_open: bool = True,
        hidden_dim: int = 64,
        graph_layers: int = 6,
        event_sampling_factor: int = 1,
        graph_radius: float = 0.08,
        graph_position_dims: int = 3,
        graph_chunk_size: int = 512,
        max_graph_edges: int | None = 2_000_000,
        spline_kernel_size: int = 5,
        spline_degree: int = 1,
        spline_root_weight: bool = True,
        snn_dynamics: str = "literal_eq15",
        raster_downsample: int = 4,
        decoder_channels: int = 48,
        output_channels: int = 1,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        if int(architecture_version) != PAPER_CORE_VERSION:
            raise ValueError(
                f"architecture_version must be {PAPER_CORE_VERSION}; legacy edge-MLP "
                "checkpoints are intentionally incompatible"
            )
        if graph_operator != "spline":
            raise ValueError("graph_operator must be 'spline' for the ASGCN paper core")
        if spline_backend != "torch":
            raise ValueError("Only the portable pure-PyTorch spline backend is supported")
        if spline_pseudo != "distance_over_radius":
            raise ValueError(
                "spline_pseudo must be 'distance_over_radius'; this explicit "
                "reparameterization maps the paper's scalar distance to the "
                "SplineConv [0,1] domain"
            )
        if not bool(spline_is_open):
            raise ValueError("Only open B-spline bases are supported")
        if (
            isinstance(event_sampling_factor, bool)
            or int(event_sampling_factor) != event_sampling_factor
        ):
            raise ValueError("event_sampling_factor must be an integer")
        if int(event_sampling_factor) < 1:
            raise ValueError("event_sampling_factor must be at least 1")
        if not math.isfinite(float(graph_radius)) or float(graph_radius) <= 0:
            raise ValueError("graph_radius must be positive and finite")
        if int(graph_position_dims) not in {1, 2, 3, 4}:
            raise ValueError("graph_position_dims must be one of 1, 2, 3, or 4")
        if int(graph_chunk_size) < 1:
            raise ValueError("graph_chunk_size must be at least 1")
        if max_graph_edges is not None and (
            isinstance(max_graph_edges, bool)
            or int(max_graph_edges) != max_graph_edges
            or int(max_graph_edges) < 1
        ):
            raise ValueError("max_graph_edges must be a positive integer or null")
        if snn_dynamics not in {"literal_eq15", "standard_if"}:
            raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
        if int(raster_downsample) < 1:
            raise ValueError("raster_downsample must be at least 1")
        self.architecture_version = PAPER_CORE_VERSION
        self.encoder = ASGCNEncoder(
            hidden_dim,
            graph_layers,
            spline_kernel_size=spline_kernel_size,
            spline_degree=spline_degree,
            spline_root_weight=spline_root_weight,
        )
        self.decoder = RasterDecoder(hidden_dim, decoder_channels, output_channels, recurrent)
        self.event_sampling_factor = int(event_sampling_factor)
        self.graph_radius = float(graph_radius)
        self.graph_position_dims = int(graph_position_dims)
        self.graph_chunk_size = int(graph_chunk_size)
        self.max_graph_edges = int(max_graph_edges) if max_graph_edges is not None else None
        self.snn_dynamics = snn_dynamics
        self.raster_downsample = int(raster_downsample)

    def _graph(self, sample: dict[str, Any]) -> EventGraph:
        return build_event_graph(
            sample["events"],
            sample["sensor_size"],
            event_sampling_factor=self.event_sampling_factor,
            graph_radius=self.graph_radius,
            graph_position_dims=self.graph_position_dims,
            graph_chunk_size=self.graph_chunk_size,
            max_graph_edges=self.max_graph_edges,
        )

    def forward_sample(
        self,
        sample: dict[str, Any],
        inference_mode: str = "ann",
        simulation_steps: int = 16,
        return_activations: bool = False,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
            raise ValueError("simulation_steps must be an integer")
        simulation_steps = int(simulation_steps)
        graph = self._graph(sample)
        if inference_mode == "ann":
            features, activations = self.encoder.forward_ann(graph, return_activations)
            firing_rates: list[torch.Tensor] = []
        elif inference_mode == "snn":
            features, firing_rates = self.encoder.forward_snn(
                graph,
                simulation_steps,
                dynamics=self.snn_dynamics,
            )
            # Express normalized spike amplitudes in the analog decoder's trained
            # lambda_L units. For literal Eq. (15), this is dimensional rescaling,
            # not a claim of proven finite-T ANN-rate equivalence.
            features = features * self.encoder.output_activation_scale(features)
            activations = []
        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}")
        raster = rasterize_features(features, graph, sample["sensor_size"], self.raster_downsample)
        prediction, next_state = self.decoder(raster, sample["sensor_size"], recurrent_state)
        dataset_sampling_ratio = float(
            sample.get("metadata", {}).get("dataset_sampling_ratio", 1.0)
        )
        node_count = int(graph.node_features.shape[0])
        edge_count = int(graph.edge_index.shape[1])
        if node_count:
            degree = torch.bincount(graph.edge_index[1], minlength=node_count)
            isolated_nodes = (degree == 0).sum()
            max_degree = degree.max()
        else:
            isolated_nodes = graph.node_features.new_zeros((), dtype=torch.long)
            max_degree = graph.node_features.new_zeros((), dtype=torch.long)
        firing_rate_denominators = (
            [simulation_steps * node_count * layer.out_channels for layer in self.encoder.layers]
            if inference_mode == "snn"
            else []
        )
        diagnostics = {
            "paper_core_version": self.architecture_version,
            "nodes": node_count,
            "edges": edge_count,
            "isolated_nodes": isolated_nodes,
            "isolate_ratio": isolated_nodes.to(graph.node_features.dtype)
            / float(max(1, node_count)),
            "max_degree": max_degree,
            "edge_feature": "normalized_scalar_distance",
            "event_sampling_factor": self.event_sampling_factor,
            "dataset_sampling_ratio": dataset_sampling_ratio,
            "effective_sampling_ratio": (dataset_sampling_ratio * self.event_sampling_factor),
            "snn_dynamics": self.snn_dynamics if inference_mode == "snn" else None,
            "decoder_input_lambda_applied": inference_mode == "snn",
            "firing_rates": firing_rates,
            "firing_rate_denominators": firing_rate_denominators,
            "spike_counts": [
                rate * denominator
                for rate, denominator in zip(
                    firing_rates,
                    firing_rate_denominators,
                    strict=True,
                )
            ],
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
        if momentum != -1.0:
            raise ValueError(
                "ASGCN paper-core calibration uses exact feature-wise maxima; momentum must be -1"
            )
        graph = self._graph(sample)
        _, activations = self.encoder.forward_ann(graph, return_activations=True)
        self.encoder.update_activation_maxima(activations)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.encoder.fold_batch_norm()

    @torch.no_grad()
    def reset_activation_maxima(self) -> None:
        self.encoder.reset_activation_maxima()

    @torch.no_grad()
    def apply_parameter_normalization(self) -> None:
        self.encoder.apply_parameter_normalization()

    def calibration_summary(self) -> dict[str, list[int] | int]:
        return self.encoder.calibration_summary()
~~~~~~~~

# src/asgcn_recon/utils.py

~~~~~~~~python
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


def resolve_experiment_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Return a copy with filesystem paths anchored to the checkout root."""
    resolved = copy.deepcopy(config)
    base_dir = experiment_base_dir(config_path)
    path_locations = (
        ("dataset", "root"),
        ("dataset", "val_root"),
        ("dataset", "split_manifest"),
        ("dataset", "file_manifest"),
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
~~~~~~~~

# tests/__init__.py

~~~~~~~~python
"""Test-only package; never installed with asgcn_recon."""
~~~~~~~~

# tests/fixtures.py

~~~~~~~~python
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
~~~~~~~~

# tests/test_asgcn_paper_core.py

~~~~~~~~python
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from asgcn_recon.graph import (
    ASGCNEncoder,
    EventGraph,
    PaperSplineConv,
    build_event_graph,
    build_radius_graph,
    linear_open_bspline_basis,
    prepare_event_nodes,
    uniformly_sample_events,
)
from asgcn_recon.model import ASGCNReconstructor


def _single_node_graph() -> EventGraph:
    return EventGraph(
        node_features=torch.zeros((1, 4)),
        positions=torch.zeros((1, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
    )


def _normalized_single_layer_encoder() -> ASGCNEncoder:
    encoder = ASGCNEncoder(
        hidden_dim=1,
        graph_layers=1,
        spline_kernel_size=2,
        spline_root_weight=True,
    ).eval()
    encoder.fold_batch_norm()
    encoder.layers[0].activation_max.fill_(1.0)
    encoder.calibration_samples_seen.fill_(1)
    encoder.apply_parameter_normalization()
    return encoder


def test_uniform_event_sampling_and_polarity_encoding() -> None:
    events = torch.tensor(
        [[index, 0, index, index % 2] for index in range(10)], dtype=torch.float32
    )
    sampled = uniformly_sample_events(events, factor=3)
    torch.testing.assert_close(sampled, events[[0, 3, 6, 9]])
    with pytest.raises(ValueError, match="at least 1"):
        uniformly_sample_events(events, factor=0)

    node_features, positions = prepare_event_nodes(events[:2], sensor_size=(2, 10))
    torch.testing.assert_close(node_features[:, 3], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(positions[:, 3], torch.tensor([0.0, 1.0]))

    graph = build_event_graph(
        events,
        (2, 10),
        event_sampling_factor=3,
        graph_radius=2.0,
        graph_position_dims=3,
        graph_chunk_size=2,
    )
    assert graph.node_features.shape == (4, 4)
    torch.testing.assert_close(graph.node_features[:, 0], sampled[:, 0] / 9.0)


def test_radius_graph_is_exactly_undirected_without_self_edges() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 1.0],
            [0.9, 0.0, 0.0, 0.0],
        ]
    )
    edge_index, edge_attr = build_radius_graph(
        positions,
        radius=0.5,
        position_dims=3,
        chunk_size=2,
    )

    torch.testing.assert_close(edge_index, torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(edge_attr, torch.tensor([[0.6], [0.6]]))
    assert edge_attr.shape == (2, 1)
    assert torch.all(edge_index[0] != edge_index[1])
    assert torch.all((0.0 <= edge_attr) & (edge_attr <= 1.0))

    boundary_edges, _ = build_radius_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        radius=0.5,
        position_dims=3,
        chunk_size=1,
    )
    assert boundary_edges.shape == (2, 0), "The paper uses distance < D, not <= D"

    with pytest.raises(RuntimeError, match="max_graph_edges=5"):
        build_radius_graph(
            torch.zeros((4, 3)),
            radius=1.0,
            position_dims=3,
            chunk_size=2,
            max_edges=5,
        )


def test_linear_open_bspline_endpoints_and_partition_of_unity() -> None:
    pseudo = torch.tensor([[0.0], [0.125], [0.5], [0.875], [1.0]])
    indices, weights = linear_open_bspline_basis(pseudo, kernel_size=5)

    torch.testing.assert_close(indices[0], torch.tensor([0, 1]))
    torch.testing.assert_close(weights[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(indices[-1], torch.tensor([4, 0]))
    torch.testing.assert_close(weights[-1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(5))
    assert torch.all(weights >= 0)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        linear_open_bspline_basis(torch.tensor([[1.01]]), kernel_size=5)


def test_linear_bspline_exact_endpoint_matches_official_pyg_pseudo_gradient() -> None:
    """The official degree-1 backend wraps the inactive right basis at u=1."""
    layer = PaperSplineConv(
        1,
        1,
        kernel_size=2,
        root_weight=False,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[3.0]], [[5.0]]]))
    features = torch.tensor([[8.0]])
    edge_index = torch.tensor([[0], [0]])
    edge_attr = torch.tensor([[1.0]], requires_grad=True)

    output = layer.spline_aggregate(features, edge_index, edge_attr)
    output.sum().backward()

    torch.testing.assert_close(output, torch.tensor([[40.0]]))
    # At u=1 PyG uses bases [K-1, 0], so d/du = x * (W_0 - W_1).
    torch.testing.assert_close(edge_attr.grad, torch.tensor([[-16.0]]))


def test_spline_parameter_initialization_matches_official_pyg_bounds() -> None:
    layer = PaperSplineConv(4, 7, kernel_size=5, root_weight=True, bias=True)
    spline_bound = 1.0 / (5 * 4) ** 0.5
    root_bound = 1.0 / 4**0.5

    assert torch.all(layer.weight.abs() <= spline_bound)
    assert layer.root is not None
    assert torch.all(layer.root.abs() <= root_bound)
    torch.testing.assert_close(layer.bias, torch.zeros(7))


def test_spline_mean_aggregation_matches_hand_calculation_and_gradients() -> None:
    layer = PaperSplineConv(
        1,
        1,
        kernel_size=2,
        root_weight=False,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[3.0]], [[5.0]]]))
    features = torch.tensor([[2.0], [4.0], [8.0]], requires_grad=True)
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]])
    edge_attr = torch.tensor([[0.0], [1.0], [0.25]])

    output = layer.spline_aggregate(features, edge_index, edge_attr)
    torch.testing.assert_close(output, torch.tensor([[0.0], [23.0], [14.0]]))
    output.sum().backward()

    torch.testing.assert_close(features.grad, torch.tensor([[1.5], [3.5], [2.5]]))
    torch.testing.assert_close(layer.weight.grad, torch.tensor([[[4.0]], [[5.0]]]))


def test_batch_norm_folding_preserves_preactivation() -> None:
    generator = torch.Generator().manual_seed(33)
    layer = PaperSplineConv(2, 3, kernel_size=3, root_weight=True, bias=True).eval()
    with torch.no_grad():
        layer.norm.running_mean.copy_(torch.tensor([0.3, -0.2, 0.7]))
        layer.norm.running_var.copy_(torch.tensor([0.5, 2.0, 4.0]))
        layer.norm.weight.copy_(torch.tensor([1.2, -0.7, 0.4]))
        layer.norm.bias.copy_(torch.tensor([-0.1, 0.5, 0.2]))
    features = torch.rand((4, 2), generator=generator)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([[0.0], [0.25], [0.75], [1.0]])

    before = layer.preactivation(features, edge_index, edge_attr)
    layer.fold_batch_norm()
    after = layer.preactivation(features, edge_index, edge_attr)

    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
    assert layer.bn_bypassed.item() is True
    assert layer._bn_is_folded is True


def test_equation_6_scales_kernel_root_and_bias_per_feature() -> None:
    layer = PaperSplineConv(2, 2, kernel_size=2, root_weight=True, bias=True).eval()
    with pytest.raises(RuntimeError, match="Fold BatchNorm"):
        layer.apply_parameter_normalization(torch.ones(2), torch.ones(2))

    layer.fold_batch_norm()
    folded_weight = layer.weight.detach().clone()
    folded_root = layer.root.detach().clone()
    folded_bias = layer.bias.detach().clone()
    input_scale = torch.tensor([2.0, 4.0])
    output_scale = torch.tensor([5.0, 10.0])
    layer.apply_parameter_normalization(input_scale, output_scale)

    torch.testing.assert_close(
        layer.weight,
        folded_weight * input_scale.view(1, -1, 1) / output_scale.view(1, 1, -1),
    )
    torch.testing.assert_close(
        layer.root,
        folded_root * input_scale.view(-1, 1) / output_scale.view(1, -1),
    )
    torch.testing.assert_close(layer.bias, folded_bias / output_scale)
    torch.testing.assert_close(layer.activation_max, output_scale)
    torch.testing.assert_close(layer.threshold, torch.ones(2))
    assert layer.snn_normalized.item() is True


def test_equation_6_requires_nonempty_calibration_and_uses_unit_for_dead_channels() -> None:
    empty_encoder = ASGCNEncoder(hidden_dim=2, graph_layers=1, spline_kernel_size=2).eval()
    empty_encoder.fold_batch_norm()
    empty_encoder.reset_activation_maxima()
    _, activations = empty_encoder.forward_ann(
        EventGraph(
            node_features=torch.empty((0, 4)),
            positions=torch.empty((0, 4)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
        ),
        return_activations=True,
    )
    empty_encoder.update_activation_maxima(activations)
    assert empty_encoder.calibration_summary()["minimum_valid_samples"] == 0
    with pytest.raises(RuntimeError, match="no non-empty calibration"):
        empty_encoder.apply_parameter_normalization()

    encoder = ASGCNEncoder(hidden_dim=2, graph_layers=1, spline_kernel_size=2).eval()
    encoder.fold_batch_norm()
    encoder.reset_activation_maxima()
    encoder.layers[0].activation_max.copy_(torch.tensor([2.0, 0.0]))
    encoder.calibration_samples_seen.fill_(1)
    encoder.apply_parameter_normalization()
    torch.testing.assert_close(
        encoder.layers[0].activation_max,
        torch.tensor([2.0, 1.0]),
    )


def test_explicit_if_uses_half_threshold_initialization_and_threshold_spikes() -> None:
    encoder = _normalized_single_layer_encoder()
    encoder.layers[0].threshold.fill_(2.0)
    current = torch.ones((1, 1))

    with patch.object(encoder.layers[0], "affine", return_value=current) as affine:
        output, firing_rates = encoder.forward_snn(_single_node_graph(), simulation_steps=1)

    # v(0)=theta/2=1 and I(1)=1 reaches theta exactly; the emitted spike is theta=2.
    torch.testing.assert_close(output, torch.tensor([[2.0]]))
    torch.testing.assert_close(firing_rates[0], torch.tensor(1.0))
    assert affine.call_count == 1


def test_explicit_if_loop_keeps_soft_reset_residual() -> None:
    encoder = _normalized_single_layer_encoder()
    currents = [torch.tensor([[0.8]]), torch.tensor([[-0.2]])]

    with patch.object(encoder.layers[0], "affine", side_effect=currents) as affine:
        output, firing_rates = encoder.forward_snn(_single_node_graph(), simulation_steps=2)

    # t1: 0.5+0.8 -> spike, residual 0.3. t2: 0.3-0.2+previous spike -> spike.
    # A hard reset to zero would not fire at t2.
    torch.testing.assert_close(output, torch.tensor([[1.0]]))
    torch.testing.assert_close(firing_rates[0], torch.tensor(1.0))
    assert affine.call_count == 2


def test_snn_reuses_one_spline_basis_across_all_timesteps() -> None:
    encoder = _normalized_single_layer_encoder()
    graph = EventGraph(
        node_features=torch.zeros((2, 4)),
        positions=torch.zeros((2, 4)),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.25], [0.25]]),
    )

    with patch(
        "asgcn_recon.graph.linear_open_bspline_basis",
        wraps=linear_open_bspline_basis,
    ) as basis:
        encoder.forward_snn(graph, simulation_steps=16)

    assert basis.call_count == 1


def test_literal_equation_15_and_standard_if_are_explicitly_distinct() -> None:
    encoder = _normalized_single_layer_encoder()
    current = torch.tensor([[0.1]])

    with patch.object(encoder.layers[0], "affine", return_value=current):
        literal_output, literal_rates = encoder.forward_snn(
            _single_node_graph(),
            simulation_steps=100,
            dynamics="literal_eq15",
        )
    with patch.object(encoder.layers[0], "affine", return_value=current):
        standard_output, standard_rates = encoder.forward_snn(
            _single_node_graph(),
            simulation_steps=100,
            dynamics="standard_if",
        )

    # The written +h(t-1) recurrence self-reinjects every prior spike. This
    # regression test prevents it from being silently presented as standard IF.
    torch.testing.assert_close(literal_output, torch.tensor([[0.96]]))
    torch.testing.assert_close(literal_rates[0], torch.tensor(0.96))
    torch.testing.assert_close(standard_output, torch.tensor([[0.1]]))
    torch.testing.assert_close(standard_rates[0], torch.tensor(0.1))

    with pytest.raises(ValueError, match="snn_dynamics"):
        encoder.forward_snn(_single_node_graph(), dynamics="ambiguous")


def test_snn_rejects_unnormalized_encoder_and_invalid_steps() -> None:
    encoder = ASGCNEncoder(hidden_dim=1, graph_layers=1, spline_kernel_size=2)
    graph = _single_node_graph()
    with pytest.raises(RuntimeError, match=r"Eq\. \(6\)"):
        encoder.forward_snn(graph, simulation_steps=2)
    with pytest.raises(ValueError, match="simulation_steps"):
        encoder.forward_snn(graph, simulation_steps=0)
    with pytest.raises(ValueError, match="integer"):
        encoder.forward_snn(graph, simulation_steps=1.9)
    with pytest.raises(ValueError, match="integer"):
        encoder.forward_snn(graph, simulation_steps=True)


def test_real_equation_6_standard_if_lambda_boundary_matches_ann_activation() -> None:
    encoder = ASGCNEncoder(
        hidden_dim=1,
        graph_layers=1,
        spline_kernel_size=2,
        spline_root_weight=True,
    ).eval()
    graph = EventGraph(
        node_features=torch.tensor([[0.5, 0.0, 0.0, 0.0]]),
        positions=torch.zeros((1, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
    )
    with torch.no_grad():
        encoder.layers[0].weight.zero_()
        encoder.layers[0].root.zero_()
        encoder.layers[0].root[0, 0] = 1.0
        encoder.layers[0].bias.zero_()
        encoder.fold_batch_norm()
        encoder.reset_activation_maxima()
        ann_output, activations = encoder.forward_ann(graph, return_activations=True)
        encoder.update_activation_maxima(activations)
        encoder.apply_parameter_normalization()
        normalized_spikes, _ = encoder.forward_snn(
            graph,
            simulation_steps=8,
            dynamics="standard_if",
        )
        decoder_units = normalized_spikes * encoder.output_activation_scale(normalized_spikes)

    torch.testing.assert_close(ann_output, torch.tensor([[0.5]]))
    torch.testing.assert_close(decoder_units, ann_output)


def test_empty_and_single_node_graphs_are_finite_and_differentiable() -> None:
    empty = build_event_graph(
        torch.empty((0, 4)),
        (8, 8),
        event_sampling_factor=1,
        graph_radius=0.08,
        graph_position_dims=3,
        graph_chunk_size=4,
    )
    assert empty.node_features.shape == (0, 4)
    assert empty.edge_index.shape == (2, 0)
    assert empty.edge_attr.shape == (0, 1)

    single = build_event_graph(
        torch.tensor([[3.0, 4.0, 0.0, 1.0]]),
        (8, 8),
        event_sampling_factor=1,
        graph_radius=0.08,
        graph_position_dims=3,
        graph_chunk_size=4,
    )
    assert single.node_features.shape == (1, 4)
    assert single.edge_index.shape == (2, 0)

    encoder = ASGCNEncoder(hidden_dim=2, graph_layers=1, spline_kernel_size=2)
    empty_output, _ = encoder.forward_ann(empty)
    single_output, _ = encoder.forward_ann(single)
    assert empty_output.shape == (0, 2)
    assert torch.isfinite(single_output).all()
    single_output.sum().backward()
    assert encoder.layers[0].root.grad is not None
    assert torch.isfinite(encoder.layers[0].root.grad).all()

    snn_encoder = _normalized_single_layer_encoder()
    empty_snn, firing_rates = snn_encoder.forward_snn(empty, simulation_steps=3)
    assert empty_snn.shape == (0, 1)
    torch.testing.assert_close(firing_rates[0], torch.tensor(0.0))


def test_legacy_edge_mlp_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy edge-MLP"):
        ASGCNReconstructor(architecture_version=1)

    with pytest.raises(ValueError, match="distance_over_radius"):
        ASGCNReconstructor(spline_pseudo="distance")


def test_snn_restores_last_layer_lambda_before_analog_decoder() -> None:
    model = ASGCNReconstructor(
        hidden_dim=2,
        graph_layers=1,
        spline_kernel_size=2,
        decoder_channels=4,
        recurrent=False,
    ).eval()
    sample = {
        "events": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "target": torch.zeros((1, 4, 4)),
        "sensor_size": (4, 4),
        "sample_id": "known-scale",
        "metadata": {"dataset_sampling_ratio": 1.0},
    }
    captured: dict[str, torch.Tensor] = {}

    def capture_raster(features, graph, sensor_size, downsample):
        del graph, sensor_size, downsample
        captured["features"] = features.detach().clone()
        return torch.zeros((1, 2, 1, 1))

    with (
        patch.object(
            model.encoder,
            "forward_snn",
            return_value=(torch.ones((1, 2)), [torch.tensor(0.0)]),
        ),
        patch.object(
            model.encoder,
            "output_activation_scale",
            return_value=torch.tensor([2.0, 3.0]),
        ),
        patch("asgcn_recon.model.rasterize_features", side_effect=capture_raster),
        patch.object(
            model.decoder,
            "forward",
            return_value=(torch.zeros((1, 1, 4, 4)), None),
        ),
    ):
        _, diagnostics = model.forward_sample(sample, inference_mode="snn")

    torch.testing.assert_close(captured["features"], torch.tensor([[2.0, 3.0]]))
    assert diagnostics["decoder_input_lambda_applied"] is True
    assert diagnostics["isolated_nodes"].item() == 1
    assert diagnostics["max_degree"].item() == 0
~~~~~~~~

# tests/test_data_validation.py

~~~~~~~~python
from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from asgcn_recon.data import (
    EventAidRZipDataset,
    EventHDRDataset,
    build_dataset,
    load_eventhdr_split_manifest,
)
from asgcn_recon.engine import _dataset_coverage_summary
from tests.fixtures import make_eventaid, make_eventhdr


def _remove_events_group(h5: h5py.File) -> None:
    del h5["events"]


def _remove_images_group(h5: h5py.File) -> None:
    del h5["images"]


def _remove_polarity_array(h5: h5py.File) -> None:
    del h5["events/ps"]


def _shorten_y_array(h5: h5py.File) -> None:
    values = h5["events/ys"][:-1]
    del h5["events/ys"]
    h5["events"].create_dataset("ys", data=values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_remove_events_group, "required group 'events'"),
        (_remove_images_group, "required group 'images'"),
        (_remove_polarity_array, "events/ps"),
        (_shorten_y_array, "equal lengths"),
    ],
)
def test_eventhdr_rejects_missing_or_misaligned_event_arrays(
    tmp_path: Path,
    mutation: Callable[[h5py.File], None],
    message: str,
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        mutation(h5)

    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


@pytest.mark.parametrize("attribute", ["event_idx", "timestamp"])
def test_eventhdr_requires_image_boundary_attributes(tmp_path: Path, attribute: str) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        del h5["images/image000000000"].attrs[attribute]

    with pytest.raises(ValueError, match=rf"missing '{attribute}'"):
        EventHDRDataset(path.parent)


def test_eventhdr_uses_relative_paths_for_nested_duplicate_basenames(tmp_path: Path) -> None:
    make_eventhdr(tmp_path / "one")
    make_eventhdr(tmp_path / "two")

    dataset = EventHDRDataset(tmp_path)
    assert len(dataset) == 8
    assert {dataset[0]["metadata"]["scene"], dataset[4]["metadata"]["scene"]} == {
        "one/test.h5",
        "two/test.h5",
    }
    assert dataset[0]["sample_id"] != dataset[4]["sample_id"]


def test_eventhdr_distinguishes_h5_and_hdf5_stems(tmp_path: Path) -> None:
    first = make_eventhdr(tmp_path / "hdr")
    first.rename(first.with_name("same.h5"))
    second = make_eventhdr(tmp_path / "other")
    second.rename(tmp_path / "hdr" / "same.hdf5")

    dataset = EventHDRDataset(tmp_path / "hdr")
    scenes = {dataset[0]["metadata"]["scene"], dataset[4]["metadata"]["scene"]}
    assert scenes == {"same.h5", "same.hdf5"}


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_final_eventhdr_manifest_normalizes_physical_scene_groups(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "FINAL",
            "scene_groups": {
                "night-drive-a": ["chunk_02.h5", "nested\\chunk_01.hdf5"],
                "night-drive-b": ["validation.h5"],
            },
            "train_scenes": ["night-drive-a"],
            "val_scenes": ["night-drive-b"],
        },
    )

    manifest = load_eventhdr_split_manifest(manifest_path)

    assert manifest["status"] == "final"
    assert manifest["split_schema"] == "physical_scenes_v1"
    assert manifest["train_files"] == ["chunk_02.h5", "nested/chunk_01.hdf5"]
    assert manifest["val_files"] == ["validation.h5"]
    assert manifest["file_to_scene"] == {
        "chunk_02.h5": "night-drive-a",
        "nested/chunk_01.hdf5": "night-drive-a",
        "validation.h5": "night-drive-b",
    }


def test_official_separate_root_manifest_maps_overlapping_names_to_sequence_groups(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
        },
    )

    manifest = load_eventhdr_split_manifest(manifest_path)

    assert manifest["split_schema"] == "official_separate_roots_v1"
    assert manifest["group_semantics"] == "h5_sequence_file_not_physical_scene"
    assert manifest["train_files"] == ["1.h5"]
    assert manifest["val_files"] == ["1.h5"]
    assert manifest["file_to_group"] == {
        "train": {"1.h5": "official-train-h5::1.h5"},
        "val": {"1.h5": "official-eval-h5::1.h5"},
    }


def test_factory_uses_split_local_sequence_groups_for_overlapping_official_names(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "eval"
    make_eventhdr(train_root).rename(train_root / "1.h5")
    make_eventhdr(val_root).rename(val_root / "1.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(train_root),
        "val_root": str(val_root),
        "split_manifest": str(manifest_path),
    }

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")
    try:
        assert train_dataset.group_semantics == "h5_sequence_file_not_physical_scene"
        assert val_dataset.group_semantics == "h5_sequence_file_not_physical_scene"
        assert train_dataset[0]["metadata"]["scene"] == "official-train-h5::1.h5"
        assert val_dataset[0]["metadata"]["scene"] == "official-eval-h5::1.h5"
        coverage = _dataset_coverage_summary(val_dataset, config)
        assert coverage["quality_grouping"] == "source_h5_sequence_file"
    finally:
        train_dataset.close()
        val_dataset.close()

    make_eventhdr(tmp_path / "extra").rename(val_root / "2.h5")
    with pytest.raises(ValueError, match=r"dataset\.val_root.*undeclared: 2\.h5"):
        build_dataset(config, split="val")


def test_official_separate_root_manifest_requires_distinct_roots(tmp_path: Path) -> None:
    root = tmp_path / "eventhdr"
    make_eventhdr(root).rename(root / "1.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
        },
    )

    with pytest.raises(ValueError, match="requires distinct"):
        build_dataset(
            {
                "type": "eventhdr",
                "root": str(root),
                "val_root": str(root),
                "split_manifest": str(manifest_path),
            },
            split="train",
        )


def test_official_sequence_file_schema_rejects_physical_scene_fields(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
            "scene_groups": {"unsupported-claim": ["1.h5"]},
            "train_scenes": ["unsupported-claim"],
            "val_scenes": ["unsupported-claim"],
        },
    )

    with pytest.raises(ValueError, match="must not declare physical-scene fields"):
        load_eventhdr_split_manifest(manifest_path)


def test_checked_in_full_eventhdr_protocol_uses_official_roots_and_all_frames() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_eventhdr_split_manifest(repository / "manifests/eventhdr_split.json")
    config = json.loads((repository / "configs/hdr_train.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "final"
    assert manifest["split_schema"] == "official_separate_roots_v1"
    assert manifest["group_semantics"] == "h5_sequence_file_not_physical_scene"
    assert set(manifest["train_files"]) == {f"{index}.h5" for index in range(1, 52)}
    assert set(manifest["val_files"]) == {f"{index}.h5" for index in range(1, 20)}
    assert config["dataset"]["root"] == "data/EventHDR/train"
    assert config["dataset"]["val_root"] == "data/EventHDR/eval"
    assert config["dataset"]["frame_stride"] == 1
    assert config["dataset"]["crop_size"] is None
    assert config["train"]["max_train_samples"] is None
    assert config["train"]["max_val_samples"] is None
    assert config["train"]["validate_every"] is None


def test_final_eventhdr_manifest_rejects_legacy_file_lists(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "train_files": ["train.h5"],
            "val_files": ["val.h5"],
        },
    )

    with pytest.raises(ValueError, match="requires scene_groups"):
        load_eventhdr_split_manifest(manifest_path)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"val_scenes": ["scene-a"]}, "leaks physical scenes"),
        (
            {
                "scene_groups": {
                    "scene-a": ["shared.h5"],
                    "scene-b": ["shared.h5"],
                }
            },
            "multiple physical scenes",
        ),
        ({"val_scenes": ["undefined"]}, "undefined physical scenes"),
        ({"scene_groups": {"scene-a": [], "scene-b": ["b.h5"]}}, "non-empty list"),
        (
            {
                "scene_groups": {
                    "scene-a": ["a.h5"],
                    "scene-b": ["b.h5"],
                    "scene-c": ["c.h5"],
                }
            },
            "leaves physical scenes unassigned",
        ),
    ],
)
def test_physical_scene_manifest_rejects_leakage_and_invalid_ownership(
    tmp_path: Path, update: dict, message: str
) -> None:
    payload = {
        "status": "final",
        "scene_groups": {"scene-a": ["a.h5"], "scene-b": ["b.h5"]},
        "train_scenes": ["scene-a"],
        "val_scenes": ["scene-b"],
    }
    payload.update(update)
    manifest_path = _write_manifest(tmp_path / "split.json", payload)

    with pytest.raises(ValueError, match=message):
        load_eventhdr_split_manifest(manifest_path)


def test_physical_scene_manifest_rejects_incomplete_schema(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "provisional",
            "scene_groups": {"scene-a": ["a.h5"]},
            "train_files": ["a.h5"],
            "val_files": ["b.h5"],
        },
    )

    with pytest.raises(ValueError, match=r"incomplete.*train_scenes, val_scenes"):
        load_eventhdr_split_manifest(manifest_path)


def test_factory_assigns_physical_scene_and_retains_source_file(tmp_path: Path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root / "chunk-one")
    make_eventhdr(data_root / "chunk-two")
    make_eventhdr(data_root / "held-out")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "physical-night-drive": [
                    "chunk-two/test.h5",
                    "chunk-one/test.h5",
                ],
                "physical-day-drive": ["held-out/test.h5"],
            },
            "train_scenes": ["physical-night-drive"],
            "val_scenes": ["physical-day-drive"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(data_root),
        "split_manifest": str(manifest_path),
        "max_events": 8,
    }

    train_dataset = build_dataset(config, split="train")
    first = train_dataset[0]
    second_file = train_dataset[4]

    assert train_dataset.samples[0]["scene"] == "physical-night-drive"
    assert first["metadata"]["scene"] == "physical-night-drive"
    assert first["metadata"]["source_file"] == "chunk-two/test.h5"
    assert second_file["metadata"]["source_file"] == "chunk-one/test.h5"
    assert first["sample_id"] != second_file["sample_id"]

    val_sample = build_dataset(config, split="val")[0]
    assert val_sample["metadata"]["scene"] == "physical-day-drive"
    assert val_sample["metadata"]["source_file"] == "held-out/test.h5"


def test_final_manifest_must_cover_every_h5_under_root(tmp_path: Path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root / "train").rename(data_root / "train" / "a.h5")
    make_eventhdr(data_root / "val").rename(data_root / "val" / "b.h5")
    make_eventhdr(data_root / "extra").rename(data_root / "extra" / "c.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "scene-a": ["train/a.h5"],
                "scene-b": ["val/b.h5"],
            },
            "train_scenes": ["scene-a"],
            "val_scenes": ["scene-b"],
        },
    )

    with pytest.raises(ValueError, match="must cover every H5.*extra/c.h5"):
        build_dataset(
            {
                "type": "eventhdr",
                "root": str(data_root),
                "split_manifest": str(manifest_path),
            },
            split="train",
        )


def test_final_manifest_checks_separate_roots_without_collapsing_same_names(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    make_eventhdr(train_root).rename(train_root / "a.h5")
    make_eventhdr(val_root).rename(val_root / "b.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "training-scene": ["a.h5"],
                "validation-scene": ["b.h5"],
            },
            "train_scenes": ["training-scene"],
            "val_scenes": ["validation-scene"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(train_root),
        "val_root": str(val_root),
        "split_manifest": str(manifest_path),
    }

    build_dataset(config, split="train").close()
    build_dataset(config, split="val").close()

    make_eventhdr(tmp_path / "extra").rename(val_root / "a.h5")
    with pytest.raises(ValueError, match=r"dataset\.val_root.*undeclared: a\.h5"):
        build_dataset(config, split="val")


def test_final_manifest_excludes_nested_validation_root_from_training_coverage(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "dataset"
    val_root = train_root / "validation"
    make_eventhdr(train_root).rename(train_root / "train.h5")
    make_eventhdr(val_root).rename(val_root / "val.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "training-scene": ["train.h5"],
                "validation-scene": ["val.h5"],
            },
            "train_scenes": ["training-scene"],
            "val_scenes": ["validation-scene"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(train_root),
        "val_root": str(val_root),
        "split_manifest": str(manifest_path),
    }

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")
    assert [path.name for path in train_dataset.files] == ["train.h5"]
    assert [path.name for path in val_dataset.files] == ["val.h5"]
    train_dataset.close()
    val_dataset.close()


def test_eventaid_fixed_manifest_rejects_partial_external_eval(tmp_path: Path) -> None:
    data_root = tmp_path / "aid"
    make_eventaid(data_root)
    manifest_path = _write_manifest(
        tmp_path / "aid.json",
        {
            "files": [
                {"scene": "R-bear"},
                {"scene": "R-ball"},
            ]
        },
    )

    with pytest.raises(ValueError, match="coverage does not match"):
        build_dataset(
            {
                "type": "eventaid_r_zip",
                "root": str(data_root),
                "file_manifest": str(manifest_path),
            },
            split="eval",
        )


def test_provisional_legacy_manifest_keeps_file_identity_as_scene(tmp_path: Path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "provisional",
            "train_files": ["test.h5"],
            "val_files": ["unused.h5"],
        },
    )

    manifest = load_eventhdr_split_manifest(manifest_path)
    dataset = build_dataset(
        {
            "type": "eventhdr",
            "root": str(data_root),
            "split_manifest": str(manifest_path),
        },
        split="train",
    )

    assert manifest["split_schema"] == "legacy_files_v1"
    assert dataset[0]["metadata"]["scene"] == "test.h5"
    assert dataset[0]["metadata"]["source_file"] == "test.h5"


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("event_idx", -1, "outside"),
        ("event_idx", 10_000, "outside"),
        ("event_idx", 1.5, "must be an integer"),
        ("timestamp", np.nan, "must be finite"),
    ],
)
def test_eventhdr_rejects_invalid_image_boundaries(
    tmp_path: Path, attribute: str, value: float, message: str
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        h5["images/image000000000"].attrs[attribute] = value

    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("event_idx", "event_idx values must be monotonically"),
        ("timestamp", "image timestamps must be monotonically"),
    ],
)
def test_eventhdr_rejects_nonmonotonic_image_boundaries(
    tmp_path: Path, attribute: str, message: str
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        first = h5["images/image000000000"].attrs[attribute]
        h5["images/image000000001"].attrs[attribute] = first - 1

    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


def _replace_h5_array(path: Path, name: str, update: Callable[[np.ndarray], None]) -> None:
    with h5py.File(path, "a") as h5:
        values = np.asarray(h5[f"events/{name}"][:], dtype=np.float64)
        update(values)
        del h5[f"events/{name}"]
        h5["events"].create_dataset(name, data=values)


def _set_nan(values: np.ndarray) -> None:
    values[1] = np.nan


def _reverse_timestamp(values: np.ndarray) -> None:
    values[1] = values[0] - 1.0


def _set_out_of_range_y(values: np.ndarray) -> None:
    values[1] = 32


def _set_invalid_polarity(values: np.ndarray) -> None:
    values[1] = 2


@pytest.mark.parametrize(
    ("array_name", "update", "message"),
    [
        ("ts", _set_nan, "timestamps must be finite"),
        ("ts", _reverse_timestamp, "timestamps must be monotonically"),
        ("xs", _set_nan, "coordinates must be finite"),
        ("ys", _set_out_of_range_y, "coordinates must lie within"),
        ("ps", _set_invalid_polarity, "polarity values must be"),
    ],
)
def test_eventhdr_rejects_malformed_loaded_event_blocks(
    tmp_path: Path,
    array_name: str,
    update: Callable[[np.ndarray], None],
    message: str,
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    _replace_h5_array(path, array_name, update)
    dataset = EventHDRDataset(path.parent, max_events=None)

    with pytest.raises(ValueError, match=message):
        dataset[0]


def _replace_zip_member(path: Path, member: str, replacement: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w") as destination:
        for info, content in entries:
            destination.writestr(info, replacement if info.filename == member else content)
    temporary.replace(path)


def _remove_zip_member(path: Path, member: str) -> None:
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w") as destination:
        for info, content in entries:
            if info.filename != member:
                destination.writestr(info, content)
    temporary.replace(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"10 0 0\n", "expected four values"),
        (b"10 0 0 1 junk 0 0 1\n", "every token must be numeric"),
        (b"nan 0 0 1\n", "timestamps must be finite"),
        (b"10 0 0 1\n9 0 0 1\n", "timestamps must be monotonically"),
        (b"10 nan 0 1\n", "coordinates must be finite"),
        (b"10 48 0 1\n", "coordinates must lie within"),
        (b"10 0 0 nan\n", "polarity must be finite"),
        (b"10 0 0 2\n", "polarity values must be"),
    ],
)
def test_eventaid_rejects_malformed_event_text(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "event/000001.txt", content)
    dataset = EventAidRZipDataset(path.parent, max_events=None)

    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_eventaid_rejects_nonmonotonic_frame_timestamps(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "timestamps.txt", b"100\n99\n101\n102\n")

    with pytest.raises(ValueError, match="timestamps.*strictly increasing"):
        EventAidRZipDataset(path.parent)


def test_eventaid_requires_timestamps_file(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _remove_zip_member(path, "timestamps.txt")

    with pytest.raises(ValueError, match="timestamps.txt is missing"):
        EventAidRZipDataset(path.parent)


def test_eventaid_rejects_internal_gt_gap(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _remove_zip_member(path, "gt/000003_img.png")

    with pytest.raises(ValueError, match="GT IDs are not contiguous"):
        EventAidRZipDataset(path.parent)


def test_eventaid_requires_timestamp_for_every_pair(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "timestamps.txt", b"100\n200\n")

    with pytest.raises(ValueError, match="does not cover"):
        EventAidRZipDataset(path.parent)
~~~~~~~~

# tests/test_e2e.py

~~~~~~~~python
from __future__ import annotations

import torch

from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.engine import _model_state_sha256, benchmark, evaluate
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
        "architecture_version": 2,
        "graph_operator": "spline",
        "spline_backend": "torch",
        "spline_pseudo": "distance_over_radius",
        "spline_is_open": True,
        "hidden_dim": 8,
        "graph_layers": 2,
        "event_sampling_factor": 1,
        "graph_radius": 2.0,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
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
    model_state = model.state_dict()
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "epoch": 0,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": model_config,
        },
        checkpoint,
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
    assert timing["mean_raw_events"] == 80
    assert timing["mean_retained_events"] == 32
    assert timing["retention_ratio"] == 32 / 80
    assert timing["raw_events_per_second"] > timing["retained_events_per_second"]
    assert timing["graph_nodes_per_second"] == timing["retained_events_per_second"]
    assert timing["events_per_second"] == timing["retained_events_per_second"]
    assert timing["recurrent_context_frames"] == 1
    assert timing["state_resets"] == 0
    assert timing["state_reset_ratio"] == 0.0
    ann_output = output_dir / "ann"
    assert (ann_output / "metrics.json").is_file()
    assert (ann_output / "frames.csv").is_file()
    assert (ann_output / "benchmark.json").is_file()
    assert len(list((ann_output / "predictions").glob("*_pred.png"))) == 1

    hdr.close()
    aid.close()
~~~~~~~~

# tests/test_engine_integrity.py

~~~~~~~~python
from __future__ import annotations

import copy
import sys

import pytest
import torch

from asgcn_recon.engine import (
    _centralize_gradients,
    _clip_and_validate_gradients,
    _ensure_finite_loss,
    _model_state_sha256,
    _prediction_artifact_stem,
    _training_protocol,
    _validate_snn_request,
    _validate_training_protocol,
    load_model_checkpoint,
)
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import atomic_torch_save
from scripts import check_env


def _config() -> dict:
    return {
        "seed": 23,
        "train": {
            "epochs": 10,
            "batch_size": 1,
            "num_workers": 0,
            "amp": True,
            "learning_rate": 1e-3,
            "weight_decay": 5e-3,
            "grad_clip": 2.0,
            "max_train_samples": 20,
            "validate_every": 2,
            "log_every": 10,
            "loss_weights": {
                "charbonnier": 1.0,
                "ssim": 0.25,
                "gradient": 0.05,
                "temporal": 0.1,
            },
        },
    }


def test_training_protocol_captures_trajectory_but_allows_run_control_changes() -> None:
    config = _config()
    protocol = _training_protocol(config, torch.device("cpu"))
    assert protocol["optimizer"]["name"] == "AdamW"
    assert protocol["optimizer"]["learning_rate"] == pytest.approx(1e-3)
    assert protocol["mixed_precision"] == {
        "requested": True,
        "effective": False,
        "autocast_dtype": None,
        "gradient_scaler": False,
    }
    assert len(protocol["source"]["source_tree_sha256"]) == 64
    assert protocol["runtime"]["gpu_name"] is None
    assert protocol["runtime"]["compute_capability"] is None
    assert protocol["version"] == 3

    allowed = copy.deepcopy(config)
    allowed["train"].update({"epochs": 99, "log_every": 1, "resume": "/another/last.pt"})
    assert _training_protocol(allowed, torch.device("cpu")) == protocol

    changed = copy.deepcopy(config)
    changed["train"]["learning_rate"] = 2e-3
    with pytest.raises(ValueError, match=r"training protocol differs.*optimizer"):
        _validate_training_protocol(
            {"training_protocol": protocol},
            _training_protocol(changed, torch.device("cpu")),
        )


def test_training_protocol_can_reserve_validation_for_the_final_epoch() -> None:
    config = _config()
    config["train"]["validate_every"] = None
    protocol = _training_protocol(config, torch.device("cpu"))
    assert protocol["validate_every"] is None
    assert protocol["checkpoint_selection"] == "single_final_epoch"


def test_paper_optimizer_mode_records_gc_and_milestone_schedule() -> None:
    config = _config()
    config["train"].update({"optimizer": "adam_gc", "lr_milestones": [8, 4], "lr_gamma": 0.2})
    protocol = _training_protocol(config, torch.device("cpu"))
    assert protocol["optimizer"]["name"] == "Adam"
    assert protocol["optimizer"]["gradient_centralization"] is True
    assert protocol["scheduler"] == {
        "name": "MultiStepLR",
        "milestones": [4, 8],
        "gamma": 0.2,
        "step_unit": "epoch",
        "step_timing": "after_epoch",
    }

    model = torch.nn.Linear(3, 2)
    model.weight.grad = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    model.bias.grad = torch.tensor([1.0, 2.0])
    original_bias = model.bias.grad.clone()
    _centralize_gradients(model)
    assert torch.allclose(model.weight.grad.mean(dim=1), torch.zeros(2))
    assert torch.equal(model.bias.grad, original_bias)


def test_exact_resume_rejects_checkpoint_without_training_protocol() -> None:
    with pytest.raises(ValueError, match="missing training_protocol"):
        _validate_training_protocol({}, _training_protocol(_config(), torch.device("cpu")))


def test_nonfinite_loss_components_and_gradients_fail_fast() -> None:
    with pytest.raises(FloatingPointError, match="total loss"):
        _ensure_finite_loss(
            torch.tensor(float("nan")),
            {"charbonnier": 1.0, "total": float("nan")},
            epoch=1,
            step=2,
            sample_id="sample-a",
        )
    with pytest.raises(FloatingPointError, match="charbonnier"):
        _ensure_finite_loss(
            torch.tensor(1.0),
            {"charbonnier": float("inf"), "total": 1.0},
            epoch=1,
            step=2,
            sample_id="sample-a",
        )

    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="gradients after clipping"):
        _clip_and_validate_gradients(model, 1.0, epoch=1, step=2, sample_id="sample-a")


def test_snn_requires_paper_core_parameter_normalization() -> None:
    checkpoint = {
        "checkpoint_type": "snn_inference",
        "batch_norm_folded": True,
        "snn_calibration_samples": 1,
        "paper_core_version": 2,
    }
    with pytest.raises(ValueError, match="parameter_normalized"):
        _validate_snn_request("snn", 4, checkpoint)
    checkpoint["parameter_normalized"] = True
    checkpoint["snn_calibration_valid_samples"] = 1
    _validate_snn_request("snn", 4, checkpoint)


def test_checkpoint_loader_rejects_unversioned_legacy_model(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": {}, "model_config": {}}, checkpoint)
    with pytest.raises(ValueError, match="architecture_version"):
        load_model_checkpoint(checkpoint, torch.device("cpu"), {})

    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model = ASGCNReconstructor(**model_config)
    mismatch = tmp_path / "mismatch.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": model_config},
        mismatch,
    )
    changed = dict(model_config, recurrent=True)
    with pytest.raises(ValueError, match="model_config differs"):
        load_model_checkpoint(mismatch, torch.device("cpu"), changed)


def test_checkpoint_loader_cross_checks_conversion_metadata_and_layer_state(tmp_path) -> None:
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model = ASGCNReconstructor(**model_config)
    model_state = model.state_dict()
    path = tmp_path / "tampered.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "snn_inference",
            "model_config": model_config,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "batch_norm_folded": True,
            "parameter_normalized": True,
        },
        path,
    )
    with pytest.raises(ValueError, match="batch_norm_folded metadata disagrees"):
        load_model_checkpoint(path, torch.device("cpu"), model_config)


def test_checkpoint_loader_rejects_finite_model_tensor_tampering(tmp_path) -> None:
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model_state = ASGCNReconstructor(**model_config).state_dict()
    checkpoint = {
        "checkpoint_type": "ann_inference",
        "model_config": model_config,
        "model": model_state,
        "model_state_sha256": _model_state_sha256(model_state),
    }
    valid_path = tmp_path / "valid.pt"
    atomic_torch_save(checkpoint, valid_path)
    load_model_checkpoint(valid_path, torch.device("cpu"), model_config)

    tensor_name = next(name for name, value in model_state.items() if value.is_floating_point())
    model_state[tensor_name] = model_state[tensor_name].clone()
    model_state[tensor_name].view(-1)[0] += 0.25
    tampered_path = tmp_path / "finite-tampered.pt"
    atomic_torch_save(checkpoint, tampered_path)
    with pytest.raises(ValueError, match="does not match tensor bytes"):
        load_model_checkpoint(tampered_path, torch.device("cpu"), model_config)


def test_prediction_artifact_stems_are_cross_platform_safe_and_collision_resistant() -> None:
    first = _prediction_artifact_stem("a/b_c:CON?*", 3)
    second = _prediction_artifact_stem(r"a_b/c:CON?*\\tail", 3)
    repeated_at_other_index = _prediction_artifact_stem("a/b_c:CON?*", 4)

    assert first != second
    assert first != repeated_at_other_index
    assert first.startswith("00000003_")
    assert len(first) <= 86
    assert all(
        character.isascii() and (character.isalnum() or character in "._-") for character in first
    )


@pytest.mark.parametrize(
    ("flag", "required_key", "absent_keys"),
    [
        (
            "--require-eventhdr-train",
            "eventhdr_train_h5",
            ("eventhdr_eval_h5", "eventaid_r_zip"),
        ),
        (
            "--require-eventhdr-eval",
            "eventhdr_eval_h5",
            ("eventhdr_train_h5", "eventaid_r_zip"),
        ),
        (
            "--require-eventaid-all",
            "eventaid_r_zip",
            ("eventhdr_train_h5", "eventhdr_eval_h5"),
        ),
    ],
)
def test_check_env_dataset_requirements_are_independent(
    tmp_path, monkeypatch, flag: str, required_key: str, absent_keys: tuple[str, ...]
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(runs_root),
            flag,
        ],
    )
    with pytest.raises(SystemExit) as error:
        check_env.main()
    message = str(error.value)
    assert required_key in message
    assert all(key not in message for key in absent_keys)


def test_check_env_full_data_preserves_all_requirements(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-full-data",
        ],
    )
    with pytest.raises(SystemExit) as error:
        check_env.main()
    message = str(error.value)
    assert "eventhdr_train_h5" in message
    assert "eventhdr_eval_h5" in message
    assert "eventaid_r_zip" in message


@pytest.mark.parametrize(
    ("flag", "subdirectory", "expected_count"),
    [
        ("--require-eventhdr-train", "train", 51),
        ("--require-eventhdr-eval", "eval", 19),
    ],
)
def test_check_env_requires_exact_official_eventhdr_names(
    tmp_path, monkeypatch, flag: str, subdirectory: str, expected_count: int
) -> None:
    root = tmp_path / "data" / "EventHDR" / subdirectory
    root.mkdir(parents=True)
    for index in range(1, expected_count):
        (root / f"{index}.h5").touch()
    (root / f"{expected_count + 1}.h5").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            flag,
        ],
    )

    with pytest.raises(SystemExit) as error:
        check_env.main()

    message = str(error.value)
    assert f"missing={expected_count}.h5" in message
    assert f"extra={expected_count + 1}.h5" in message


def test_check_env_accepts_exact_official_eventhdr_names(tmp_path, monkeypatch) -> None:
    train_root = tmp_path / "data" / "EventHDR" / "train"
    eval_root = tmp_path / "data" / "EventHDR" / "eval"
    train_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    for index in range(1, 52):
        (train_root / f"{index}.h5").touch()
    for index in range(1, 20):
        (eval_root / f"{index}.h5").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-eventhdr-train",
            "--require-eventhdr-eval",
        ],
    )

    check_env.main()
~~~~~~~~

# tests/test_get_hdr.py

~~~~~~~~python
from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "get_hdr.py"
    spec = importlib.util.spec_from_file_location("get_hdr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_hdr = _load_script()


def _write_split(root: Path, split: str, *, missing: str | None = None) -> Path:
    directory = root / split
    directory.mkdir(parents=True)
    for name in get_hdr.EXPECTED[split]:
        if name != missing:
            (directory / name).write_bytes(get_hdr.HDF5_MAGIC + name.encode("ascii"))
    return directory


def _write_complete(root: Path) -> Path:
    _write_split(root, "train")
    _write_split(root, "eval")
    return root


def test_copy_complete_source_and_idempotent_check(tmp_path: Path) -> None:
    source = _write_complete(tmp_path / "download" / "EventHDR")
    destination = tmp_path / "data" / "EventHDR"

    assert get_hdr.main(["--source", str(source), "--destination", str(destination)]) == 0
    assert get_hdr.main(["--check", "--destination", str(destination)]) == 0
    assert get_hdr.main(["--source", str(source), "--destination", str(destination)]) == 0
    assert len(list((destination / "train").glob("*.h5"))) == 51
    assert len(list((destination / "eval").glob("*.h5"))) == 19


def test_missing_source_is_rejected_before_destination_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "download"
    _write_split(source, "train", missing="51.h5")
    _write_split(source, "eval")
    destination = tmp_path / "data" / "EventHDR"

    assert get_hdr.main(["--source", str(source), "--destination", str(destination)]) == 1
    assert not destination.exists()
    assert "missing=51.h5" in capsys.readouterr().err


def test_extra_or_nested_h5_is_rejected(tmp_path: Path) -> None:
    source = _write_complete(tmp_path / "download")
    (source / "train" / "52.h5").write_bytes(get_hdr.HDF5_MAGIC)
    with pytest.raises(get_hdr.ImportError, match="extra=52.h5"):
        get_hdr.locate_source(source, ("train", "eval"))

    (source / "train" / "52.h5").unlink()
    nested = source / "train" / "nested"
    nested.mkdir()
    (nested / "1.h5").write_bytes(get_hdr.HDF5_MAGIC)
    with pytest.raises(get_hdr.ImportError, match="Nested HDF5"):
        get_hdr.validate_split_dir(source / "train", "train")


def test_separate_split_source_directory_is_supported(tmp_path: Path) -> None:
    source = _write_split(tmp_path / "browser-download", "eval")
    destination = tmp_path / "data" / "EventHDR"
    assert (
        get_hdr.main(
            [
                "--source",
                str(source),
                "--split",
                "eval",
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    assert len(list((destination / "eval").glob("*.h5"))) == 19
    assert not (destination / "train").exists()


def test_archive_import_streams_exact_members_without_extract_tree(tmp_path: Path) -> None:
    archive_path = tmp_path / "EventHDR.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for split, names in get_hdr.EXPECTED.items():
            for name in names:
                archive.writestr(f"EventHDR/{split}/{name}", get_hdr.HDF5_MAGIC + b"data")
        archive.writestr("EventHDR/pretrained/model.pt", b"ignored")
    destination = tmp_path / "data" / "EventHDR"

    assert (
        get_hdr.main(
            ["--archive", str(archive_path), "--destination", str(destination)]
        )
        == 0
    )
    assert len(list((destination / "train").glob("*.h5"))) == 51
    assert len(list((destination / "eval").glob("*.h5"))) == 19
    assert not (destination / "pretrained").exists()


def test_archive_rejects_unsafe_or_duplicate_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in get_hdr.EXPECTED["eval"]:
            archive.writestr(f"eval/{name}", get_hdr.HDF5_MAGIC)
        archive.writestr("../eval/other.h5", get_hdr.HDF5_MAGIC)
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(get_hdr.ImportError, match="Unsafe archive member"),
    ):
        get_hdr.locate_archive_members(archive, ("eval",))

    duplicate_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_path, "w") as archive:
        for name in get_hdr.EXPECTED["eval"]:
            archive.writestr(f"eval/{name}", get_hdr.HDF5_MAGIC)
        archive.writestr("prefix/eval/1.h5", get_hdr.HDF5_MAGIC)
    with (
        zipfile.ZipFile(duplicate_path) as archive,
        pytest.raises(get_hdr.ImportError, match="duplicate"),
    ):
        get_hdr.locate_archive_members(archive, ("eval",))


def test_check_rejects_bad_hdf5_magic(tmp_path: Path) -> None:
    destination = _write_complete(tmp_path / "data" / "EventHDR")
    (destination / "train" / "26.h5").write_bytes(b"not-hdf5")
    with pytest.raises(get_hdr.ImportError, match="Not an HDF5"):
        get_hdr.check_destination(destination, ("train", "eval"))


def test_link_mode_uses_shared_storage_without_copy(tmp_path: Path) -> None:
    source = _write_complete(tmp_path / "shared" / "EventHDR")
    destination = tmp_path / "repo" / "data" / "EventHDR"
    try:
        result = get_hdr.link_source(
            {"train": source / "train", "eval": source / "eval"}, destination
        )
    except OSError as error:
        if os.name == "nt":
            pytest.skip(f"Windows symlink privilege is unavailable: {error}")
        raise
    assert result == {"linked": 2, "kept": 0}
    assert (destination / "train").is_symlink()
    assert (destination / "eval").is_symlink()
    get_hdr.check_destination(destination, ("train", "eval"))
~~~~~~~~

# tests/test_graph_vectorized.py

~~~~~~~~python
from __future__ import annotations

import torch

from asgcn_recon.graph import build_radius_graph


def _reference_radius_graph(
    positions: torch.Tensor,
    radius: float,
    position_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    destinations: list[int] = []
    attributes: list[float] = []
    for source in range(positions.shape[0]):
        for destination in range(positions.shape[0]):
            if source == destination:
                continue
            distance = torch.linalg.vector_norm(
                positions[source, :position_dims] - positions[destination, :position_dims]
            ).item()
            if distance < radius:
                sources.append(source)
                destinations.append(destination)
                attributes.append(distance / radius)
    return (
        torch.tensor((sources, destinations), dtype=torch.long),
        torch.tensor(attributes, dtype=positions.dtype).unsqueeze(-1),
    )


def test_chunked_radius_graph_matches_pairwise_reference() -> None:
    generator = torch.Generator().manual_seed(2026)
    positions = torch.rand((23, 4), generator=generator)
    radius = 0.45
    expected = _reference_radius_graph(positions, radius, position_dims=3)

    for chunk_size in (1, 2, 7, 23, 64):
        actual = build_radius_graph(
            positions,
            radius,
            position_dims=3,
            chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_radius_graph_handles_empty_input() -> None:
    edge_index, edge_attr = build_radius_graph(
        torch.empty((0, 4)),
        radius=0.08,
        position_dims=3,
        chunk_size=8,
    )

    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 1)
~~~~~~~~

# tests/test_inspect_all.py

~~~~~~~~python
from __future__ import annotations

import json

import pytest

from asgcn_recon.cli import inspect_dataset
from tests.fixtures import make_eventhdr


def test_inspect_validate_all_reads_every_sample_but_limits_preview(tmp_path) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root, frames=4)
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(root),
            "target_channels": 1,
            "max_events": 8,
        }
    }

    result = inspect_dataset(config, samples=1, validate_all=True)

    assert result["samples"] == 4
    assert result["validated_samples"] == 4
    assert result["validation_complete"] is True
    assert len(result["preview"]) == 1


def test_inspect_rejects_negative_preview_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        inspect_dataset({"dataset": {}}, samples=-1)


def test_inspect_uses_separate_validation_root(tmp_path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_path = make_eventhdr(train_root)
    val_path = make_eventhdr(val_root)
    train_path.rename(train_root / "train.h5")
    val_path.rename(val_root / "val.h5")
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "final",
                "scene_groups": {
                    "train-scene": ["train.h5"],
                    "validation-scene": ["val.h5"],
                },
                "train_scenes": ["train-scene"],
                "val_scenes": ["validation-scene"],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(train_root),
            "val_root": str(val_root),
            "split_manifest": str(manifest),
            "max_events": 8,
        }
    }

    result = inspect_dataset(config, samples=1)

    assert result["splits"]["train"]["preview"][0]["metadata"]["source"].endswith("train.h5")
    assert result["splits"]["val"]["preview"][0]["metadata"]["source"].endswith("val.h5")
~~~~~~~~

# tests/test_metrics_ssim.py

~~~~~~~~python
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from asgcn_recon.metrics import structural_similarity


def _reference_gaussian_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Small unfold-based reference independent of the production convolution."""

    size = min(window_size, *prediction.shape[-2:])
    if size % 2 == 0:
        size -= 1
    coordinates = torch.arange(size, dtype=prediction.dtype)
    coordinates -= (size - 1) / 2
    kernel_1d = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    weights = torch.outer(kernel_1d, kernel_1d).flatten().view(1, 1, -1, 1)

    def patches(value: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = value.shape
        unfolded = F.unfold(value, kernel_size=size)
        return unfolded.view(batch, channels, size * size, -1)

    x = patches(prediction)
    y = patches(target)
    mu_x = (x * weights).sum(dim=2)
    mu_y = (y * weights).sum(dim=2)
    sigma_x = (x.square() * weights).sum(dim=2) - mu_x.square()
    sigma_y = (y.square() * weights).sum(dim=2) - mu_y.square()
    sigma_xy = (x * y * weights).sum(dim=2) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean().clamp(-1.0, 1.0)


def test_ssim_matches_gaussian_window_reference() -> None:
    generator = torch.Generator().manual_seed(2026)
    prediction = torch.rand((2, 2, 9, 8), generator=generator, dtype=torch.float64)
    target = torch.rand((2, 2, 9, 8), generator=generator, dtype=torch.float64)

    actual = structural_similarity(prediction, target, window_size=5)
    expected = _reference_gaussian_ssim(prediction, target, window_size=5)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("height,width", [(1, 1), (1, 7), (2, 2), (5, 8), (10, 10)])
def test_ssim_is_finite_and_exact_for_identical_small_images(height: int, width: int) -> None:
    image = torch.linspace(0.0, 1.0, height * width).reshape(1, 1, height, width)

    result = structural_similarity(image, image)

    assert torch.isfinite(result)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0.0, atol=1e-6)


def test_ssim_small_image_has_finite_gradient() -> None:
    prediction = torch.tensor([[[[0.1, 0.4], [0.8, 0.2]]]], requires_grad=True)
    target = torch.tensor([[[[0.2, 0.3], [0.7, 0.1]]]])

    loss = 1.0 - structural_similarity(prediction, target)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_ssim_promotes_mixed_precision_inputs_for_amp_training() -> None:
    prediction = torch.rand((1, 1, 8, 8), dtype=torch.bfloat16, requires_grad=True)
    target = torch.rand((1, 1, 8, 8), dtype=torch.float32)

    result = structural_similarity(prediction, target)
    result.backward()

    assert result.dtype == torch.float32
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_ssim_disables_outer_autocast_for_stable_local_statistics() -> None:
    prediction = torch.zeros((1, 1, 16, 16), requires_grad=True)
    target = torch.zeros_like(prediction)

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        result = structural_similarity(prediction, target)
    result.backward()

    torch.testing.assert_close(result, torch.ones_like(result), rtol=0.0, atol=1e-6)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
~~~~~~~~

# tests/test_p0_engine.py

~~~~~~~~python
from __future__ import annotations

import json
import shutil
from collections import Counter

import pytest
import torch

from asgcn_recon.data import EventHDRDataset, load_eventhdr_split_manifest
from asgcn_recon.engine import (
    _balanced_contiguous_indices,
    _balanced_sample_indices,
    _centralize_gradients,
    _continues_sequence,
    _dataset_content_fingerprint,
    _enforce_training_split_status,
    _macro_ssim,
    _model_state_sha256,
    _prefix_context_schedule,
    _representative_schedule,
    _resume_best_macro_ssim,
    _sample_event_counts,
    _sampling_summary,
    _validate_resume_best_pair,
    _validate_snn_request,
    benchmark,
    calibrate,
    evaluate,
    load_model_checkpoint,
)
from asgcn_recon.graph import PaperSplineConv
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import atomic_torch_save
from tests.fixtures import make_eventhdr


class _GroupedIndexDataset:
    def __init__(self) -> None:
        self.samples = [
            *({"scene": "long"} for _ in range(8)),
            *({"scene": "medium"} for _ in range(4)),
            *({"scene": "short"} for _ in range(2)),
        ]

    def __len__(self) -> int:
        return len(self.samples)


def _model_config() -> dict:
    return {
        "architecture_version": 2,
        "graph_operator": "spline",
        "spline_backend": "torch",
        "spline_pseudo": "distance_over_radius",
        "spline_is_open": True,
        "hidden_dim": 4,
        "graph_layers": 1,
        "event_sampling_factor": 1,
        "graph_radius": 1.0,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": False,
    }


def _eval_config(root, output_dir) -> dict:
    return {
        "seed": 19,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(root),
            "target_channels": 1,
            "max_events": 16,
            "crop_size": None,
            "tone_map": "log",
        },
        "model": _model_config(),
        "eval": {
            "num_workers": 0,
            "max_samples": 1,
            "save_predictions": 0,
            "output_dir": str(output_dir),
        },
    }


def test_balanced_indices_cover_groups_before_repeating() -> None:
    dataset = _GroupedIndexDataset()
    indices = _balanced_sample_indices(dataset, limit=6, seed=3)
    counts = Counter(dataset.samples[index]["scene"] for index in indices)
    assert counts == {"long": 2, "medium": 2, "short": 2}
    assert indices == sorted(indices)
    summary = _sampling_summary(dataset, indices)
    assert summary["selected_samples"] == 6
    assert summary["selected_groups"] == summary["available_groups"] == 3
    long_indices = [index for index in indices if dataset.samples[index]["scene"] == "long"]
    assert long_indices == [0, 7]


def test_representative_schedule_has_exact_length_and_balances_groups() -> None:
    dataset = _GroupedIndexDataset()
    schedule = _representative_schedule(dataset, count=6, seed=3)
    assert len(schedule) == 6
    counts = Counter(dataset.samples[index]["scene"] for index in schedule)
    assert counts == {"long": 2, "medium": 2, "short": 2}


def test_contiguous_sampler_balances_groups_without_state_gaps() -> None:
    dataset = _GroupedIndexDataset()
    indices = _balanced_contiguous_indices(dataset, limit=6, seed=3, require_all_groups=True)
    grouped = {
        scene: [index for index in indices if dataset.samples[index]["scene"] == scene]
        for scene in ("long", "medium", "short")
    }
    assert {scene: len(values) for scene, values in grouped.items()} == {
        "long": 2,
        "medium": 2,
        "short": 2,
    }
    assert all(values[1] == values[0] + 1 for values in grouped.values())

    with pytest.raises(ValueError, match="every validation group"):
        _balanced_contiguous_indices(dataset, limit=2, seed=3, require_all_groups=True)


def test_prefix_context_replays_unscored_predecessors() -> None:
    dataset = _GroupedIndexDataset()
    schedule, score_positions = _prefix_context_schedule(dataset, [6, 7, 12])
    assert schedule == [0, 1, 2, 3, 4, 5, 6, 7, 12]
    assert score_positions == {6, 7, 8}
    bounded, bounded_scores = _prefix_context_schedule(dataset, [6, 7], max_context_frames=2)
    assert bounded == [4, 5, 6, 7]
    assert bounded_scores == {2, 3}
    with pytest.raises(ValueError, match="non-negative"):
        _prefix_context_schedule(dataset, [6], max_context_frames=-1)


def test_content_fingerprint_is_path_independent_and_detects_changes(tmp_path) -> None:
    source = make_eventhdr(tmp_path / "one")
    destination_root = tmp_path / "two"
    destination_root.mkdir()
    destination = destination_root / source.name
    shutil.copy2(source, destination)
    first = EventHDRDataset(source.parent, max_events=8)
    second = EventHDRDataset(destination.parent, max_events=8)

    assert _dataset_content_fingerprint(first) == _dataset_content_fingerprint(second)

    with destination.open("ab") as handle:
        handle.write(b"changed")
    assert _dataset_content_fingerprint(first) != _dataset_content_fingerprint(second)
    first.close()
    second.close()


def test_macro_ssim_is_the_checkpoint_selection_score() -> None:
    validation = {
        "micro": {"ssim": 0.95},
        "macro": {"ssim": 0.61},
        "per_scene": {},
    }
    assert _macro_ssim(validation) == pytest.approx(0.61)
    assert _macro_ssim({"ssim": 0.72}) == pytest.approx(0.72)


def test_resume_rejects_legacy_micro_best_score() -> None:
    with pytest.raises(ValueError, match="predates macro-SSIM"):
        _resume_best_macro_ssim({"best_ssim": 0.9, "val": {"ssim": 0.9}})
    assert _resume_best_macro_ssim(
        {"best_metric": "macro_ssim", "best_ssim": 0.7}
    ) == pytest.approx(0.7)


def test_checkpoint_loader_rejects_legacy_graph_architecture(tmp_path) -> None:
    legacy = tmp_path / "legacy.pt"
    atomic_torch_save(
        {
            "model": {},
            "model_config": {"hidden_dim": 4, "graph_layers": 1},
        },
        legacy,
    )
    with pytest.raises(ValueError, match="architecture_version"):
        load_model_checkpoint(legacy, torch.device("cpu"), _model_config())


def test_resume_rejects_unrelated_historical_best_checkpoint() -> None:
    model_digest = "a" * 64
    resume = {
        "epoch": 2,
        "model_config": _model_config(),
        "validation_protocol": {"version": 2},
        "training_protocol": {"version": 1},
        "paper_core_version": 2,
        "best_metric": "macro_ssim",
        "best_ssim": 0.7,
        "best_model_state_sha256": model_digest,
        "val": {},
    }
    best = {
        **resume,
        "epoch": 1,
        "model_state_sha256": model_digest,
        "val": {"macro": {"ssim": 0.7}},
    }
    _validate_resume_best_pair(resume, best)

    best["validation_protocol"] = {"version": 99}
    with pytest.raises(ValueError, match="different validation protocol"):
        _validate_resume_best_pair(resume, best)


def test_training_rejects_nonfinal_split(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    split = {
        "status": "provisional",
        "train_files": ["train.h5"],
        "val_files": ["val.h5"],
    }
    manifest.write_text(json.dumps(split), encoding="utf-8")
    config = {
        "dataset": {"split_manifest": str(manifest)},
        "train": {},
    }
    with pytest.raises(ValueError, match="cannot be used for training"):
        _enforce_training_split_status(config)

    split["status"] = "final"
    manifest.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="requires scene_groups"):
        _enforce_training_split_status(config)

    split = {
        "status": "final",
        "scene_groups": {
            "train-scene": ["train.h5"],
            "validation-scene": ["val.h5"],
        },
        "train_scenes": ["train-scene"],
        "val_scenes": ["validation-scene"],
    }
    manifest.write_text(json.dumps(split), encoding="utf-8")
    _enforce_training_split_status(config)


def test_split_manifest_rejects_exact_train_val_overlap(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "provisional",
                "train_files": ["shared.h5"],
                "val_files": ["shared.h5"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leaks files across train/val"):
        load_eventhdr_split_manifest(manifest)


def test_sequence_continuity_requires_adjacent_index_and_matching_shape() -> None:
    assert _continues_sequence("scene", 8, (32, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("scene", 9, (32, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("scene", 8, (16, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("other", 8, (32, 48), "scene", 7, (32, 48))


def test_event_count_fallback_uses_retained_tensor_for_custom_datasets() -> None:
    sample = {"events": torch.zeros((7, 4))}
    assert _sample_event_counts(sample) == (7, 7)
    sample["metadata"] = None
    assert _sample_event_counts(sample) == (7, 7)
    sample["metadata"] = {"raw_event_count": "12"}
    assert _sample_event_counts(sample) == (12, 7)
    sample["metadata"] = {"raw_event_count": 3}
    assert _sample_event_counts(sample) == (7, 7)


def test_gradient_centralization_respects_spline_output_axis() -> None:
    class MixedWeights(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spline = PaperSplineConv(2, 3, kernel_size=2)
            self.linear = torch.nn.Linear(4, 3, bias=False)

    model = MixedWeights()
    model.spline.weight.grad = torch.arange(
        model.spline.weight.numel(), dtype=torch.float32
    ).reshape_as(model.spline.weight)
    model.spline.root.grad = torch.arange(
        model.spline.root.numel(), dtype=torch.float32
    ).reshape_as(model.spline.root)
    model.linear.weight.grad = torch.arange(
        model.linear.weight.numel(), dtype=torch.float32
    ).reshape_as(model.linear.weight)

    _centralize_gradients(model)

    torch.testing.assert_close(
        model.spline.weight.grad.mean(dim=(0, 1)),
        torch.zeros(model.spline.out_channels),
    )
    torch.testing.assert_close(
        model.spline.root.grad.mean(dim=0),
        torch.zeros(model.spline.out_channels),
    )
    torch.testing.assert_close(
        model.linear.weight.grad.mean(dim=1),
        torch.zeros(model.linear.out_features),
    )


def test_snn_request_requires_steps_and_calibration_metadata() -> None:
    with pytest.raises(ValueError, match="simulation_steps"):
        _validate_snn_request("snn", 0)
    with pytest.raises(ValueError, match="integer"):
        _validate_snn_request("snn", 1.5)
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        _validate_snn_request("snn", 4, {"model": {}}, "ann.pt")
    _validate_snn_request(
        "snn",
        4,
        {
            "checkpoint_type": "snn_inference",
            "batch_norm_folded": True,
            "snn_calibration_samples": 1,
            "snn_calibration_valid_samples": 1,
            "paper_core_version": 2,
            "parameter_normalized": True,
        },
        "snn.pt",
    )


def test_ann_request_rejects_parameter_normalized_snn_checkpoint() -> None:
    _validate_snn_request("ann", 16, {"checkpoint_type": "ann_inference"}, "ann.pt")
    with pytest.raises(ValueError, match="ANN checkpoint"):
        _validate_snn_request(
            "ann",
            16,
            {"checkpoint_type": "snn_inference"},
            "snn.pt",
        )
    with pytest.raises(ValueError, match=r"Eq\. \(6\)-normalized"):
        _validate_snn_request("ann", 16, {"parameter_normalized": True}, "snn.pt")


def test_calibration_is_balanced_and_writes_clean_inference_checkpoint(tmp_path) -> None:
    root = tmp_path / "hdr"
    first = make_eventhdr(root / "scene_a")
    second = make_eventhdr(root / "scene_b")
    first.rename(first.with_name("a.h5"))
    second.rename(second.with_name("b.h5"))

    model_config = _model_config()
    source = tmp_path / "training.pt"
    model = ASGCNReconstructor(**model_config)
    model_state = model.state_dict()
    atomic_torch_save(
        {
            "checkpoint_type": "training",
            "epoch": 7,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": model_config,
            "paper_core_version": 2,
            "optimizer": {"large": "training-only"},
            "scaler": {"training-only": True},
            "history": [1, 2, 3],
            "rng_state": {"training-only": True},
        },
        source,
    )
    config = _eval_config(root, tmp_path / "eval")
    output = tmp_path / "snn.pt"
    calibrate(config, source, output, samples=2)
    with pytest.raises(FileExistsError, match="already exists"):
        calibrate(config, source, output, samples=1)

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_type"] == "snn_inference"
    assert checkpoint["model_state_sha256"] == _model_state_sha256(checkpoint["model"])
    assert checkpoint["batch_norm_folded"] is True
    assert checkpoint["paper_core_version"] == 2
    assert checkpoint["parameter_normalized"] is True
    assert checkpoint["snn_calibration_samples"] == 2
    assert checkpoint["snn_calibration_valid_samples"] == 2
    assert checkpoint["snn_calibration_summary"]["minimum_valid_samples"] == 2
    assert checkpoint["snn_calibration_summary"]["valid_samples_per_layer"] == [2]
    assert checkpoint["snn_calibration_sampling"]["selected_groups"] == 2
    assert set(checkpoint["snn_calibration_sampling"]["per_group"].values()) == {1}
    for training_key in ("optimizer", "scaler", "history", "rng_state", "config", "val"):
        assert training_key not in checkpoint

    config["eval"]["max_samples"] = 2
    result = evaluate(config, output, inference_mode="snn", simulation_steps=2)
    assert result["quality"]["frames"] == 2
    assert result["quality"]["micro"]["temporal_l1"] >= 0
    assert result["quality"]["per_scene"]["scene_a/a.h5"]["temporal_l1_frames"] == 1
    assert result["gpu_memory"] == {
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    assert result["snn_dynamics"] == "literal_eq15"
    assert 0.0 <= result["graph_topology"]["isolate_ratio"] <= 1.0
    timing = benchmark(
        config,
        output,
        warmup=0,
        steps=2,
        inference_mode="snn",
        simulation_steps=2,
    )
    assert timing["snn_dynamics"] == "literal_eq15"
    assert len(timing["layer_firing_rates"]) == 1
    assert timing["mean_firing_rate"] == pytest.approx(timing["layer_firing_rates"][0])
    standard_timing = benchmark(
        config,
        output,
        warmup=0,
        steps=1,
        inference_mode="snn",
        simulation_steps=2,
        snn_dynamics="standard_if",
    )
    assert standard_timing["snn_dynamics"] == "standard_if"

    tampered = torch.load(output, map_location="cpu", weights_only=False)
    tampered["model"]["encoder.layers.0.activation_max"][0] = float("nan")
    tampered["model_state_sha256"] = _model_state_sha256(tampered["model"])
    nonfinite_path = tmp_path / "snn_nonfinite.pt"
    torch.save(tampered, nonfinite_path)
    with pytest.raises(ValueError, match="non-finite state"):
        evaluate(config, nonfinite_path, inference_mode="snn", simulation_steps=2)

    inconsistent = torch.load(output, map_location="cpu", weights_only=False)
    inconsistent["snn_calibration_summary"]["valid_samples_per_layer"] = [999]
    inconsistent_path = tmp_path / "snn_inconsistent.pt"
    torch.save(inconsistent, inconsistent_path)
    with pytest.raises(ValueError, match="calibration metadata disagrees"):
        evaluate(config, inconsistent_path, inference_mode="snn", simulation_steps=2)

    wrong_threshold = torch.load(output, map_location="cpu", weights_only=False)
    wrong_threshold["model"]["encoder.layers.0.threshold"][0] = 0.5
    wrong_threshold["model_state_sha256"] = _model_state_sha256(wrong_threshold["model"])
    wrong_threshold_path = tmp_path / "snn_wrong_threshold.pt"
    torch.save(wrong_threshold, wrong_threshold_path)
    with pytest.raises(ValueError, match="unit threshold"):
        evaluate(config, wrong_threshold_path, inference_mode="snn", simulation_steps=2)
    with pytest.raises(ValueError, match="ANN checkpoint"):
        evaluate(config, output, inference_mode="ann")


def test_public_snn_paths_reject_invalid_requests(tmp_path) -> None:
    with pytest.raises(ValueError, match="simulation_steps"):
        evaluate({}, tmp_path / "unused.pt", inference_mode="snn", simulation_steps=0)
    with pytest.raises(ValueError, match="calibration samples"):
        calibrate({}, tmp_path / "unused.pt", tmp_path / "out.pt", samples=0)
    same_path = tmp_path / "same.pt"
    same_path.touch()
    with pytest.raises(ValueError, match="must be different"):
        calibrate({}, same_path, same_path, samples=1, overwrite=True)

    root = tmp_path / "hdr"
    make_eventhdr(root)
    model_config = _model_config()
    uncalibrated = tmp_path / "ann.pt"
    uncalibrated_state = ASGCNReconstructor(**model_config).state_dict()
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "epoch": 1,
            "model": uncalibrated_state,
            "model_state_sha256": _model_state_sha256(uncalibrated_state),
            "model_config": model_config,
            "paper_core_version": 2,
        },
        uncalibrated,
    )
    config = _eval_config(root, tmp_path / "eval")
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        evaluate(config, uncalibrated, inference_mode="snn", simulation_steps=2)
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        benchmark(
            config,
            uncalibrated,
            warmup=0,
            steps=1,
            inference_mode="snn",
            simulation_steps=2,
        )


def test_balanced_sampler_uses_eventhdr_files(tmp_path) -> None:
    root = tmp_path / "hdr"
    first = make_eventhdr(root / "a", frames=4)
    second = make_eventhdr(root / "b", frames=2)
    first.rename(first.with_name("a.h5"))
    second.rename(second.with_name("b.h5"))
    dataset = EventHDRDataset(root, max_events=8)
    indices = _balanced_sample_indices(dataset, limit=4, seed=5)
    paths = Counter(str(dataset.samples[index]["path"]) for index in indices)
    assert sorted(paths.values()) == [2, 2]
    dataset.close()
~~~~~~~~

# tests/test_pipeline.py

~~~~~~~~python
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from asgcn_recon.cli import inspect_dataset
from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.data.common import stratified_subsample, uniform_cap_ratio
from asgcn_recon.data.factory import build_dataset
from asgcn_recon.engine import _data_loader, _model_state_sha256, benchmark, train
from asgcn_recon.graph import build_radius_graph, prepare_event_nodes
from asgcn_recon.losses import ReconstructionLoss
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import (
    load_json,
    resolve_experiment_paths,
)
from tests.fixtures import make_eventaid, make_eventhdr


def _paper_model_config(
    hidden_dim: int = 4,
    graph_layers: int = 1,
    *,
    recurrent: bool = True,
) -> dict:
    return {
        "architecture_version": 2,
        "graph_operator": "spline",
        "spline_backend": "torch",
        "spline_pseudo": "distance_over_radius",
        "spline_is_open": True,
        "hidden_dim": hidden_dim,
        "graph_layers": graph_layers,
        "event_sampling_factor": 1,
        "graph_radius": 2.0,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": recurrent,
    }


def test_eventhdr_loader(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=32)
    sample = dataset[0]
    assert sample["events"].shape == (32, 4)
    assert sample["target"].shape == (1, 32, 48)
    assert sample["events"][:, 2].min() >= 0
    assert sample["events"][:, 2].max() <= 1
    assert sample["metadata"]["raw_event_count"] == 96
    assert sample["metadata"]["cropped_event_count"] == 96
    assert sample["metadata"]["retained_event_count"] == 32
    assert sample["metadata"]["dataset_sampling_ratio"] == 3.0
    assert dataset[1]["metadata"]["dt_us"] == 2_000


def test_eventhdr_stride_aggregates_intervals(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=None, frame_stride=2)
    assert len(dataset) == 2
    assert dataset.samples[1]["end_idx"] - dataset.samples[1]["start_idx"] == 192
    assert dataset[1]["metadata"]["dt_us"] == 4_000
    assert dataset[1]["metadata"]["raw_event_count"] == 192
    assert dataset[1]["metadata"]["cropped_event_count"] == 192
    assert dataset[1]["metadata"]["retained_event_count"] == 192
    assert dataset[1]["metadata"]["dataset_sampling_ratio"] == 1.0


def test_eventhdr_preserves_zero_event_target_intervals(tmp_path):
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "r+") as h5:
        first_end = int(h5["images/image000000000"].attrs["event_idx"])
        h5["images/image000000001"].attrs["event_idx"] = first_end

    dataset = EventHDRDataset(path.parent, max_events=None)
    assert len(dataset) == 4
    assert dataset.zero_event_intervals == 1
    empty = dataset[1]
    assert empty["events"].shape == (0, 4)
    assert empty["metadata"]["zero_event_interval"] is True
    assert empty["metadata"]["raw_event_count"] == 0
    assert empty["metadata"]["sequence_index"] == 1
    assert empty["metadata"]["dt_us"] == 2_000


def test_eventaid_next_frame_alignment(tmp_path):
    make_eventaid(tmp_path / "eventaid")
    dataset = EventAidRZipDataset(tmp_path / "eventaid", max_events=32)
    assert len(dataset) == 3
    assert dataset.samples[0]["frame_id"] == 1
    assert dataset.samples[0]["target_name"].endswith("000002_img.png")
    assert dataset[0]["metadata"]["dt_us"] == 10_000
    assert dataset[0]["metadata"]["raw_event_count"] == 80
    assert dataset[0]["metadata"]["cropped_event_count"] == 80
    assert dataset[0]["metadata"]["retained_event_count"] == 32
    assert dataset[0]["metadata"]["dataset_sampling_ratio"] == 2.5


def test_max_events_uses_exact_size_uniform_sampling() -> None:
    events = np.arange(13 * 4, dtype=np.float32).reshape(13, 4)
    assert uniform_cap_ratio(len(events), max_events=5) == 2.6
    retained = stratified_subsample(events, max_events=5)
    np.testing.assert_array_equal(retained, events[[0, 3, 6, 9, 12]])
    assert uniform_cap_ratio(len(events), max_events=None) == 1.0
    np.testing.assert_array_equal(stratified_subsample(events, None), events)


def test_max_events_has_no_one_event_boundary_collapse() -> None:
    at_cap = np.arange(8192 * 4, dtype=np.float32).reshape(8192, 4)
    over_cap = np.arange(8193 * 4, dtype=np.float32).reshape(8193, 4)
    assert len(stratified_subsample(at_cap, 8192)) == 8192
    assert len(stratified_subsample(over_cap, 8192)) == 8192
    assert uniform_cap_ratio(len(over_cap), 8192) == pytest.approx(8193 / 8192)


@pytest.mark.parametrize("dataset_name", ["eventhdr", "eventaid"])
def test_random_crop_is_deterministic_and_sequence_aligned(tmp_path, dataset_name):
    root = tmp_path / dataset_name
    if dataset_name == "eventhdr":
        make_eventhdr(root)
        dataset_class = EventHDRDataset
    else:
        make_eventaid(root)
        dataset_class = EventAidRZipDataset

    arguments = {
        "max_events": None,
        "crop_size": [8, 8],
        "random_crop": True,
        "seed": 41,
    }
    first = dataset_class(root, **arguments)
    first_samples = [first[index] for index in range(len(first))]
    first_crops = [
        (sample["metadata"]["crop"]["top"], sample["metadata"]["crop"]["left"])
        for sample in first_samples
    ]
    assert all(
        sample["metadata"]["raw_event_count"]
        >= sample["metadata"]["cropped_event_count"]
        == sample["metadata"]["retained_event_count"]
        and sample["metadata"]["dataset_sampling_ratio"] == 1.0
        for sample in first_samples
    )
    assert any(
        sample["metadata"]["cropped_event_count"] < sample["metadata"]["raw_event_count"]
        for sample in first_samples
    )
    repeated_crop = first[0]["metadata"]["crop"]
    first.close()

    reopened = dataset_class(root, **arguments)
    reopened_crops = [
        (sample["metadata"]["crop"]["top"], sample["metadata"]["crop"]["left"])
        for sample in (reopened[index] for index in range(len(reopened)))
    ]
    assert reopened[0]["metadata"]["crop"] == repeated_crop
    reopened.close()

    assert reopened_crops == first_crops
    assert len(set(first_crops)) == 1


def test_event_graph_is_undirected_instead_of_causal():
    events = torch.tensor([[i, i, i, i % 2] for i in range(12)], dtype=torch.float32)
    _, positions = prepare_event_nodes(events, (16, 16))
    edge_index, edge_attr = build_radius_graph(positions, radius=2.0, position_dims=3, chunk_size=4)
    pairs = set(map(tuple, edge_index.transpose(0, 1).tolist()))
    assert len(pairs) == len(events) * (len(events) - 1)
    assert all(source != destination for source, destination in pairs)
    assert all((destination, source) in pairs for source, destination in pairs)
    assert edge_attr.shape == (len(pairs), 1)


def test_empty_event_interval_uses_zero_node_graph():
    sample = {
        "events": torch.empty((0, 4), dtype=torch.float32),
        "target": torch.zeros((1, 8, 8), dtype=torch.float32),
        "sensor_size": (8, 8),
        "sample_id": "empty/0",
        "metadata": {},
    }
    model = ASGCNReconstructor(**_paper_model_config())
    prediction, diagnostics = model.forward_sample(sample)
    prediction.mean().backward()
    assert torch.isfinite(prediction).all()
    assert diagnostics["nodes"] == 0
    assert diagnostics["edges"] == 0


def test_model_forward_backward(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model = ASGCNReconstructor(**_paper_model_config(hidden_dim=8, graph_layers=2))
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert prediction.shape == (1, 1, 32, 48)
    assert diagnostics["edges"] == diagnostics["nodes"] * (diagnostics["nodes"] - 1)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_bn_folding_and_explicit_snn_path(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model_config = _paper_model_config(hidden_dim=8, graph_layers=2, recurrent=False)
    model = ASGCNReconstructor(**model_config).eval()
    with torch.no_grad():
        ann_before, _ = model.forward_sample(sample)
        model.fold_batch_norm()
        ann_after, _ = model.forward_sample(sample)
        restored = ASGCNReconstructor(**model_config).eval()
        restored.load_state_dict(model.state_dict())
        ann_restored, _ = restored.forward_sample(sample)
        model.reset_activation_maxima()
        model.calibrate_sample(sample)
        model.apply_parameter_normalization()
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
    model = ASGCNReconstructor(**_paper_model_config())
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


def test_eventhdr_manifest_accepts_nested_relative_paths(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root / "scene")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": ["scene/test.h5"],
                "val_files": ["unused.h5"],
            }
        ),
        encoding="utf-8",
    )
    dataset = build_dataset(
        {
            "type": "eventhdr",
            "root": str(data_root),
            "split_manifest": str(manifest_path),
        },
        split="train",
    )
    assert len(dataset) == 4
    assert dataset[0]["metadata"]["scene"] == "scene/test.h5"


def test_factory_uses_val_root_for_validation_split(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_path = make_eventhdr(train_root, frames=2)
    val_path = make_eventhdr(val_root, frames=4)
    train_path.rename(train_root / "train.h5")
    val_path.rename(val_root / "val.h5")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": ["train.h5"],
                "val_files": ["val.h5"],
            }
        ),
        encoding="utf-8",
    )
    dataset = build_dataset(
        {
            "type": "eventhdr",
            "root": str(train_root),
            "val_root": str(val_root),
            "split_manifest": str(manifest_path),
        },
        split="val",
    )
    assert len(dataset) == 4
    assert dataset[0]["metadata"]["source"].endswith("val.h5")


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
        "model": _paper_model_config(),
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
    assert all(key in first for key in ("optimizer", "scheduler", "scaler", "rng_state"))
    assert (tmp_path / "run/.data_hash_cache.json").is_file()
    protocol_text = json.dumps(first["validation_protocol"])
    assert str(data_root) not in protocol_text
    assert "mtime_ns" not in protocol_text
    best = torch.load(tmp_path / "run/best.pt", map_location="cpu", weights_only=False)
    assert best["checkpoint_type"] == "ann_inference"
    assert first["checkpoint_type"] == "training"
    assert first["model_state_sha256"] == _model_state_sha256(first["model"])
    assert len(first["best_model_state_sha256"]) == 64
    assert best["model_state_sha256"] == first["best_model_state_sha256"]
    for training_key in (
        "optimizer",
        "scheduler",
        "scaler",
        "history",
        "rng_state",
        "config",
    ):
        assert training_key not in best

    config["train"]["epochs"] = 2
    train(config, resume_from=tmp_path / "run/last.pt")
    resumed = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]

    config["train"]["epochs"] = 3
    config["seed"] = 18
    with pytest.raises(ValueError, match="validation protocol differs"):
        train(config, resume_from=tmp_path / "run/last.pt")


def test_null_validation_interval_scores_only_the_single_final_epoch(tmp_path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    config["train"].update({"epochs": 2, "validate_every": None})

    train(config)

    checkpoint = torch.load(
        tmp_path / "run/last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["history"][0]["val"] == {}
    assert checkpoint["history"][1]["val"]["frames"] == 1
    assert checkpoint["checkpoint_selection"] == "single_final_epoch"
    assert checkpoint["validation_protocol"]["selection_metric"] == (
        "single_final_epoch_macro_ssim"
    )


def test_training_rejects_resume_into_a_different_run_directory(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)
    source = tmp_path / "run/last.pt"

    config["train"]["epochs"] = 2
    config["output"]["run_dir"] = str(tmp_path / "other-run")
    with pytest.raises(ValueError, match="inside the configured run_dir"):
        train(config, resume_from=source)


def test_exact_resume_rejects_missing_state_and_tampered_historical_best(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)

    for missing_key in ("scaler", "rng_state"):
        run_root = tmp_path / missing_key
        config = _tiny_training_config(run_root, data_root)
        train(config)
        last_path = run_root / "run/last.pt"
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        checkpoint.pop(missing_key)
        torch.save(checkpoint, last_path)
        config["train"]["epochs"] = 2
        expected_message = "GradScaler state" if missing_key == "scaler" else "RNG state"
        with pytest.raises(ValueError, match=expected_message):
            train(config, resume_from=last_path)

    digest_root = tmp_path / "digest"
    config = _tiny_training_config(digest_root, data_root)
    train(config)
    best_path = digest_root / "run/best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    tensor_name = next(name for name, value in best["model"].items() if value.is_floating_point())
    best["model"][tensor_name] = best["model"][tensor_name].clone()
    best["model"][tensor_name].view(-1)[0] += 1
    torch.save(best, best_path)
    config["train"]["epochs"] = 2
    with pytest.raises(ValueError, match="does not match tensor bytes"):
        train(config, resume_from=digest_root / "run/last.pt")

    last_digest_root = tmp_path / "last-digest"
    config = _tiny_training_config(last_digest_root, data_root)
    train(config)
    last_path = last_digest_root / "run/last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    tensor_name = next(name for name, value in last["model"].items() if value.is_floating_point())
    last["model"][tensor_name] = last["model"][tensor_name].clone()
    last["model"][tensor_name].view(-1)[0] += 1
    torch.save(last, last_path)
    config["train"]["epochs"] = 2
    with pytest.raises(ValueError, match="does not match tensor bytes"):
        train(config, resume_from=last_path)


def test_training_can_resume_before_first_validation_checkpoint(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)

    last_path = tmp_path / "run/last.pt"
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    checkpoint["best_ssim"] = float("-inf")
    checkpoint["val"] = {}
    checkpoint["history"] = [
        {"epoch": 1, "train_loss": checkpoint["history"][0]["train_loss"], "val": {}}
    ]
    torch.save(checkpoint, last_path)
    (tmp_path / "run/best.pt").unlink()

    config["train"]["epochs"] = 2
    train(config, resume_from=last_path)
    resumed = torch.load(last_path, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert (tmp_path / "run/best.pt").is_file()


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
~~~~~~~~

# tests/test_server_orchestration.py

~~~~~~~~python
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_full_script_covers_the_complete_ordered_experiment_matrix() -> None:
    script = _text("scripts/full.sh")

    preflight = script.index("[1/5]")
    inspection = script.index("[2/5]")
    training = script.index("[3/5]")
    calibration = script.index("[4/5]")
    evaluation = script.index("[5/5]")
    assert preflight < inspection < training < calibration < evaluation

    for config in (
        "configs/hdr_train.json",
        "configs/hdr_ann.json",
        "configs/hdr_snn.json",
        "configs/aid_ann.json",
        "configs/aid_snn.json",
    ):
        assert config in script
    assert 'SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"' in script
    assert "for dynamics in literal_eq15 standard_if" in script
    assert "RUN_BENCHMARK=1" in script
    assert 'CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"' in script
    assert 'for config_path in "${TRAIN_CONFIG}" "${AID_ANN_CONFIG}"' in script


def test_calibration_wrapper_defaults_to_all_samples_and_protects_output() -> None:
    script = _text("scripts/calibrate.sh")
    assert 'CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"' in script
    assert 'OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"' in script
    assert '--samples "${CALIBRATION_SAMPLES}"' in script
    assert "calibrated output already exists" in script
    assert "CALIBRATE_ARGS+=(--overwrite)" in script
    assert "rm -f" not in script


def test_all_wrappers_support_optional_validate_all_preflight() -> None:
    for relative in ("scripts/train.sh", "scripts/eval.sh", "scripts/calibrate.sh"):
        script = _text(relative)
        assert "INSPECT_VALIDATE_ALL" in script
        assert "INSPECT_ARGS+=(--validate-all)" in script


def test_calibration_has_slurm_and_pbs_entrypoints_with_dependency_examples() -> None:
    for relative in ("server/calibrate.sbatch", "server/calibrate.pbs"):
        script = _text(relative)
        assert script.startswith("#!/usr/bin/env bash\n")
        assert "scripts/calibrate.sh" in script
        assert "CALIBRATION_SAMPLES" in script
        assert "depend" in script


def test_eventaid_downloader_defaults_to_the_complete_release() -> None:
    script = _text("scripts/get_aid.sh")
    assert "the complete 14-scene release is downloaded" in script
    assert "SCENES=(R-bear)" not in script
    assert 'if ((DOWNLOAD_ALL == 0)) && ((${#SCENES[@]} == 0)); then\n  DOWNLOAD_ALL=1' in script
~~~~~~~~

# tests/test_temporal_metric.py

~~~~~~~~python
from __future__ import annotations

import pytest
import torch

from asgcn_recon.metrics import MetricAccumulator, temporal_consistency_error


def test_temporal_consistency_error_compares_frame_changes() -> None:
    previous_prediction = torch.zeros((1, 1, 2, 2))
    prediction = torch.full((1, 1, 2, 2), 0.5)
    previous_target = torch.zeros((1, 1, 2, 2))
    target = torch.full((1, 1, 2, 2), 0.25)

    result = temporal_consistency_error(prediction, previous_prediction, target, previous_target)

    assert float(result) == pytest.approx(0.25)


def test_metric_accumulator_supports_temporal_metric_after_first_frame() -> None:
    accumulator = MetricAccumulator()
    accumulator.update("scene-a", "a/0", {"ssim": 0.8})
    accumulator.update("scene-a", "a/1", {"ssim": 0.9, "temporal_l1": 0.1})
    accumulator.update("scene-b", "b/0", {"ssim": 0.7})
    accumulator.update("scene-b", "b/1", {"ssim": 0.6, "temporal_l1": 0.3})

    summary = accumulator.summary()

    assert summary["micro"]["temporal_l1"] == pytest.approx(0.2)
    assert summary["macro"]["temporal_l1"] == pytest.approx(0.2)
    assert summary["per_scene"]["scene-a"]["temporal_l1_frames"] == 1
~~~~~~~~
