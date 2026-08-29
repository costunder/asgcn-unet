# 프로젝트 코드 전체 스냅샷

- 생성 기준: 현재 Git 작업 트리
- 포함 범위: code_summary.md 자체를 제외한 Git 추적/신규 텍스트 파일 65개
- hand_off.md 포함
- 로컬 검증: Python 3.12.13, torch 2.13.0+cpu, pytest 136 passed
- 각 `# 파일경로` 아래의 fenced block은 해당 파일 전체 원문

# .dockerignore

~~~~~~text
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
~~~~~~

# .editorconfig

~~~~~~text
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
~~~~~~

# .env.example

~~~~~~text
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
~~~~~~

# .gitattributes

~~~~~~text
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
~~~~~~

# .github/ISSUE_TEMPLATE/bug_report.yml

~~~~~~yaml
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
~~~~~~

# .github/pull_request_template.md

~~~~~~markdown
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
~~~~~~

# .github/workflows/ci.yml

~~~~~~yaml
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
~~~~~~

# .gitignore

~~~~~~text
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
~~~~~~

# compose.yaml

~~~~~~yaml
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
~~~~~~

# configs/aid_ann.json

~~~~~~json
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
~~~~~~

# configs/aid_smoke.json

~~~~~~json
{
  "purpose": "non-reportable partial EventAid-R loader smoke",
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventaid_r_zip",
    "root": "data/EventAid-R",
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
    "num_workers": 0,
    "recurrent_context_frames": 8,
    "max_samples": 8,
    "save_predictions": 4,
    "output_dir": "runs/eventaid_r_smoke"
  }
}
~~~~~~

# configs/aid_snn.json

~~~~~~json
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
~~~~~~

# configs/hdr_ann.json

~~~~~~json
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
~~~~~~

# configs/hdr_smoke.json

~~~~~~json
{
  "seed": 2026,
  "device": "auto",
  "dataset": {
    "type": "eventhdr",
    "root": "data/EventHDR/train",
    "val_root": "data/EventHDR/train",
    "split_manifest": "manifests/eventhdr_smoke.json",
    "target_channels": 1,
    "max_events": 8192,
    "crop_size": [256, 256],
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
    "epochs": 1,
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "optimizer": "adam_gc",
    "learning_rate": 0.001,
    "weight_decay": 0.005,
    "lr_milestones": [20, 30],
    "lr_gamma": 0.1,
    "grad_clip": 1.0,
    "amp": true,
    "log_every": 4,
    "validate_every": 1,
    "resume": null,
    "max_train_samples": 32,
    "max_val_samples": 32,
    "validation_context_frames": 8,
    "rehash_data": false,
    "allow_provisional_split": true,
    "loss_weights": {
      "charbonnier": 1.0,
      "ssim": 0.2,
      "gradient": 0.1,
      "temporal": 0.2
    }
  },
  "output": {
    "run_dir": "runs/smoke"
  }
}
~~~~~~

# configs/hdr_snn.json

~~~~~~json
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
~~~~~~

# configs/hdr_train.json

~~~~~~json
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
    "validate_every": 1,
    "resume": null,
    "max_train_samples": null,
    "max_val_samples": 500,
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
~~~~~~

# constraints/py312.txt

~~~~~~text
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
~~~~~~

# CONTRIBUTING.md

~~~~~~markdown
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
~~~~~~

# Dockerfile

~~~~~~dockerfile
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
~~~~~~

# docs/ASGCN.md

~~~~~~markdown
# ASGCN 구현 범위

이 저장소의 graph core는 AAAI 2025 논문에 공개된 ASGCN 수식 중 정적 graph/ANN→SNN core를
직접 구현하지만,
저자 공식 코드를 그대로 실행한 **완전 재현본**은 아니다. 2026-08-29 기준 공개 배포된 저자
코드는 확인할 수 없었다. 확인 가능한
[AAAI 공식 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32154)와
[공식 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)에는 저자 코드 링크가
없고, 아래에 적은 구현 세부값 일부도 논문에 명시되지 않았다. 따라서 논문으로 확인되는 부분과
이 저장소의 가정·과제용 확장을 분리해 기록한다.

## 논문에 정의된 graph/SNN core

- 입력 event를 sampling factor `R`로 균일하게 줄인다.
- event 좌표 사이의 유클리드 거리가 `D`보다 작은 두 node를 잇는 simple undirected radius graph를
  만들고, scalar distance를 edge pseudo-coordinate로 쓴다.
- 식 (11)은 실제 이웃 수로 평균한 weighted tensor-product B-spline message aggregation을,
  ANN update는 ReLU를 정의한다.
- 식 (6)은 layer-wise lambda를 이용한 ANN parameter normalization을 정의한다. 별도 threshold
  설명은 feature dimension별 maximum activation을 사용한다.
- batch normalization은 식 (13)–(14)에 따라 convolution parameter에 fold한다.
- SNN 경로는 별도 rate encoding 없이 IF membrane을 timestep마다 전개한다. 초기 membrane은
  `theta/2`, 발화량은 `theta`, 발화 뒤에는 soft reset을 적용한다. 기본
  `snn_dynamics=literal_eq15`는 식 (15)의 `+h_i^l(t-1)`까지 문자 그대로 실행한다.
- 학습 optimizer는 Adam과 gradient centralization을 사용하며, 초기 learning rate는 `1e-3`, L2
  weight decay는 `5e-3`이다. 논문은 milestone decay를 명시하지만 정확한 epoch와 gamma는 공개하지
  않는다.

`architecture_version: 2` checkpoint만 이 의미론을 나타낸다. 현재 공통 설정은
`graph_operator: spline`, `spline_backend: torch`,
`spline_pseudo: distance_over_radius`다. 마지막 이름은 논문이 말한 원시 거리값 자체가 아니라
SplineConv 정의역에 맞춘 `distance / D` 재매개화임을 의도적으로 드러낸다.

## 논문에 없는 구현 가정

논문만으로 단일한 정답을 정할 수 없는 값은 config에 노출했다. 기본값은 다음과 같다.

| 항목 | 이 저장소의 명시적 가정 |
|---|---|
| graph position | `x, y, t`를 각각 `[0,1]`로 정규화한 3차원 좌표 |
| polarity node feature | 음극성 `-1`, 양극성 `+1` |
| graph radius/pseudo | 정규화 좌표에서 `D=0.08`, scalar distance를 `D`로 나눠 `[0,1]` pseudo-coordinate로 사용 |
| B-spline/update | open degree-1 spline, kernel size 5, 별도 root weight·affine bias와 현재 ReLU/BN 적용 순서 |
| graph width/depth | hidden feature 64, graph layer 6; 논문이 복원 과제용 값을 제공하지 않아 설정값으로 고정 |
| paper sampling | model의 고정 `event_sampling_factor=1`로 기본 비축소 |
| adaptive safety cap | crop 뒤 `ceil(N/max_events)` 정수 stride로 먼저 줄이는 reconstruction/server 가정이며 논문의 공식 `R`이 아님 |
| graph construction | 결과를 바꾸지 않는 512-node chunk 계산 |
| ANN→SNN scale | layer-wise lambda와 feature-wise threshold 결합이 모호해 feature-wise lambda와 정규화 뒤 unit threshold 사용 |
| first-layer scale | lambda^0은 `[1,1,1,1]`; 이후 lambda는 calibration activation maxima. `[0,1]`의 `x,y,t`와 `±1` polarity를 근거로 한 저장소 선택이며 논문에는 없음 |
| dead activation channel | calibration ReLU maximum이 0인 feature는 epsilon으로 나누지 않고 lambda 1을 사용하며, 변환 전 dead-channel 수를 checkpoint metadata에 기록 |
| gradient centralization axes | spline `[K,Cin,Cout]`와 root `[Cin,Cout]`는 마지막 output axis 외 차원, Conv/Linear는 첫 output axis 외 차원에서 평균을 제거; 저자 코드로 확인되지 않은 tensor-layout 선택 |
| edge memory guard | directed edge가 2,000,000개를 넘으면 자르지 않고 실패; 8,192-node 밀집 graph의 O(N²) OOM을 막는 서버 안전장치 |
| connectivity | `D`가 만든 graph를 강제로 연결하지 않고 평가 결과에 isolated node 비율과 max degree를 기록 |

실제 전처리 순서는 spatial crop → adaptive `max_events` stride → model의 고정 paper sampling factor
`R`이다. 두 factor와 raw/cropped/retained count를 분리해 기록하며 adaptive cap을 ASGCN 논문의 공식
sampling 설정으로 보고하지 않는다.

순수 PyTorch spline 구현은 Linux 서버 설치를 단순하게 하기 위한 선택이다. B-spline
pseudo-coordinate의 `[0,1]` 범위, open spline, degree, kernel과 root-weight 개념은
[PyTorch SplineConv 공식 연산자 소스](https://github.com/rusty1s/pytorch_spline_conv)를 참고했다.
지원하는 scalar/open/degree-1 범위에서는 endpoint index·pseudo gradient와 weight/root 초기화 bound도
공식 연산자 동작에 맞추고 회귀 테스트로 고정했다. 고정 graph의 basis/index는 한 번만 계산해 모든
layer와 IF timestep에서 재사용한다.
이는 ASGCN 저자 코드와 동일하다는 증거가 아니며, 논문에 없는 kernel size·degree·root 설정을
공식값으로 오인하면 안 된다.

### 식 (15)와 rate conversion의 공개 모호성

논문 식 (15)는 `v_tilde(t)=v(t-1)+c(t)+h(t-1)`로 적혀 있다. 이 `+h(t-1)` self-feedback을
문자 그대로 실행하면 작은 양의 정전류도 첫 발화 뒤 firing rate 1에 가까워질 수 있어, 표준
soft-reset IF의 `firing rate ≈ normalized ANN activation` 유도와 양립하지 않는다. 저자 코드나
인덱스 설명이 없어 이를 임의로 오타 처리하지 않았다.

- `literal_eq15`: 저장소 기본값. 공개 식 (15)–(17)을 그대로 실행한다.
- `standard_if`: `+h(t-1)`를 제거한 rate-conversion 대조군이며, 공식 ASGCN 값이라고 주장하지 않는다.

Eq. (6) 뒤 마지막 `lambda_L`를 곱하는 단계는 spike 출력을 학습된 analog decoder의 단위로 보내기
위한 차원 변환이다. `standard_if`에서는 보정 ANN activation과의 parity test가 있지만,
`literal_eq15`에서 유한 T의 ANN-rate 동등성을 보장한다는 뜻은 아니다. 두 dynamics의 장기-T 차이를
단위 테스트로 고정하고 checkpoint의 model config에 선택값을 보존한다.

## Event-to-frame 과제용 확장

원 논문은 event graph를 pooling한 뒤 MLP로 분류한다. 이 저장소는 분류기를 재현하는 대신
EventHDR에서 luminance frame을 복원하도록 다음 모듈을 연결한다.

```text
uniform event sampling
  -> undirected radius graph
  -> B-spline graph encoder (ANN 또는 literal-Eq15/standard-IF timestep)
  -> feature rasterization
  -> residual U-Net + analog ConvGRU
  -> luminance frame
```

따라서 residual U-Net, ConvGRU, rasterization, EventHDR/EventAid-R loader, tone mapping,
Charbonnier·SSIM·gradient·temporal loss는 ASGCN 논문의 구성요소가 아니라 연구과제에 맞춘
재구성 확장이다. SNN 전환은 graph encoder에만 적용되고 decoder와 ConvGRU는 analog 연산이다.

## 아직 구현하지 않은 논문·하드웨어 범위

- 새 event가 들어올 때 영향받는 K-hop ego-network만 갱신하는 동적 asynchronous 실행
- 논문 식 (18)–(19)의 node clustering, graph pooling과 edge remapping
- 원 논문의 classification MLP와 N-MNIST/CIFAR10-DVS/N-Caltech101/N-Cars 성능표 재현
- 논문의 연산 수 기반 energy 추정 및 실제 FPGA/ASIC 전력·에너지 측정
- event compression·transport protocol과 반도체 RTL

현재 코드는 논문의 공개 정의에 맞춘 **ASGCN paper-core 기반 event-to-frame 연구
프로토타입**이다. 누락된 저자 구현 세부정보와 다른 출력 과제 때문에 “공식 ASGCN 완전 재현”이나
원 논문 성능 재현으로 표현하지 않는다. 해당 주장은 저자 코드·동일 데이터 전처리·분류 head·공식
checkpoint를 확보해 parity를 검증한 뒤에만 가능하다.
~~~~~~

# docs/EXPERIMENT.md

~~~~~~markdown
# 실험 프로토콜

## 1. 고정할 연구 질문

1. EventHDR의 실제 이벤트에서 luminance frame을 얼마나 잘 복원하는가?
2. ASGCN paper-core의 graph depth와 radius가 복원 품질·비용에 어떤 영향을 주는가?
3. EventAid-R의 다른 장면·운동에서 품질이 얼마나 유지되는가?
4. ANN, 논문 식 (15)를 문자 그대로 실행한 IF, 표준 IF 대조군의 품질·연산 지연 차이는 무엇인가?

현재 `snn` 모드는 graph encoder의 IF membrane을 timestep마다 전개하지만 U-Net·ConvGRU decoder는
analog이고 GPU에서 기능을 모사한다. 따라서 저전력·에너지 우위를 검증하는 질문에는 답할 수 없으며,
해당 주장은 연산량 모델 또는 hardware 측정 후에만 추가한다. 논문 재현 범위는
[ASGCN 구현 범위](ASGCN.md)를 따른다.

기본 `literal_eq15`의 `+h(t-1)` self-feedback은 표준 rate-conversion IF와 동등하지 않다. 따라서
ANN↔SNN 오차는 `snn_dynamics`와 T를 함께 표기한다. `standard_if`는 모호성 분석용 대조군이지
ASGCN 저자 공식 구현이라고 간주하지 않는다.

## 2. 데이터 역할과 누수 방지

| 단계 | 데이터 | 허용되는 사용 |
|---|---|---|
| 학습 | EventHDR train | weight 최적화·crop |
| 검증 | EventHDR holdout | 물리 scene split 확정 후 macro SSIM checkpoint 선택 |
| 보정 | EventHDR train | BN folding·feature-wise threshold·parameter normalization |
| 내부 최종시험 | EventHDR 공식 eval | 학습 종료 후 1회 |
| 외부 최종시험 | EventAid-R | 학습·보정·threshold 선택 금지 |

`manifests/eventhdr_split.json`은 현재 물리 scene 대응표가 없는 legacy file-list
`provisional` 상태다. 최종 manifest는 아래처럼 동일 물리 장면의 파일을 `scene_groups`로 묶고,
겹치지 않는 scene ID를 `train_scenes`와 `val_scenes`에 넣어야 한다.

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

예시는 schema 설명일 뿐 실제 mapping이 아니다. `status`만 `final`로 바꾸거나 legacy 목록만 남기면
검증에서 거부된다. `configs/hdr_smoke.json`만 비보고용 provisional legacy schema를 허용한다.
smoke는 별도 `manifests/eventhdr_smoke.json`의 train `1.h5`, `2.h5`와 validation `48.h5`,
`49.h5`만 사용하고 content fingerprint도 이 네 파일만 hash한다.
final manifest는 `data/EventHDR/train` 아래의 모든 H5를 정확히 한 scene에 소유시키고 모든 scene을
train 또는 validation에 배정해야 한다. 누락·미선언 파일과 미배정·중복 scene/file은 거부한다.

validation sample limit은 채점 frame 수에만 적용된다. `hdr_train`은 최대 500개, `hdr_smoke`는
최대 32개를 채점하며 아래 recurrent context frame은 이 수에 포함하지 않는다.

1. group(final은 physical scene, provisional은 H5 file)별 quota를 round-robin으로 배정한다.
2. 각 group의 dataset index에서 deterministic contiguous window를 선택한다.
3. recurrent 모델은 window 앞의 같은 group predecessor를 `validation_context_frames` 한도에서
   metric 없이 replay해 streaming ConvGRU state를 예열한다. 기본값은 본학습 64, smoke 8이며
   `null`이면 전체 prefix다. non-recurrent 모델은 context를 replay하지 않는다.
4. sample limit이 group 수보다 작으면 일부 group을 버리지 않고 오류로 중단한다.
5. checkpoint 선택에는 scene별 SSIM 평균의 평균인 macro SSIM을 쓴다.

calibration은 recurrent state를 쓰지 않으므로 각 group(final은 physical scene, provisional은 H5
file)의 전체 index 범위를 `linspace`로 덮는다.
benchmark는 recurrent 모델이면 group별 연속 window와 최대 `eval.recurrent_context_frames`개의
unmeasured predecessor(현재 eval config 기본 32), 비순환 모델이면 time-spread sample을 사용한다.
`--warmup`은 recurrent context가 아니라 device/kernel warmup이다.
장면, sequence index, sensor shape가 끊기는 경계에서는 state를 초기화하며 benchmark 결과에 reset 수와
비율을 기록한다.

random crop의 RNG는 scene·source file/member를 합친 안정적인 sequence identity로 결정한다. 따라서
worker 수나 resume 여부와 무관하고, 같은 연속 sequence의 모든 frame은 동일한 sensor ROI를 사용해
ConvGRU pixel state와 temporal loss를 정렬한다. 이는 epoch마다 바뀌는 random augmentation이 아니다.

event 전처리는 spatial crop → adaptive `max_events` cap → model의 고정 paper sampling factor `R`
순서다. adaptive cap은 crop 뒤 event 수 `N`에 대해 `ceil(N/max_events)` 정수 stride를 사용하며,
서버 메모리를 위한 재구성 가정이지 논문의 공식 `R`이 아니다. sample metadata는 cap 전후를
`raw_event_count`, `cropped_event_count`, `retained_event_count`, `dataset_sampling_factor`로 분리한다.

## 3. 학습 optimizer protocol

논문에 명시된 Adam, gradient centralization, 초기 learning rate `1e-3`, L2 weight decay `5e-3`를
`optimizer=adam_gc`, `learning_rate=0.001`, `weight_decay=0.005`로 고정한다. 논문은 milestone에서
learning rate를 낮춘다고만 쓰고 정확한 epoch와 gamma를 공개하지 않는다. 따라서 40-epoch 복원
적응에서는 `lr_milestones=[20,30]`, `lr_gamma=0.1`을 명시적 가정으로 사용한다. 1-epoch smoke도
동일한 protocol을 기록하지만 milestone에 도달하지 않는다.
gradient centralization의 axis도 구현 가정이다. spline `[K,Cin,Cout]`와 root `[Cin,Cout]`는 마지막
output axis 외 차원에서, Conv/Linear는 첫 output axis 외 차원에서 gradient 평균을 제거한다.

## 4. output domain

EventHDR와 EventAid-R 모두 다음 target 변환을 쓴다.

```text
integer image -> dtype range로 [0,1] 정규화
RGB이면 BT.709 luminance
y = log1p(5000*x) / log1p(5000)
```

이는 output 수치 domain만 통일한다. 센서 response와 exposure가 동일하다는 보장은 없으므로 두
dataset의 절대 PSNR/SSIM을 동일 분포처럼 해석하지 않는다.
EventAid-R은 event block `i`를 GT `i+1`과 짝짓는 `target_offset=1`을 명시적 protocol 가정으로
사용한다. 이 정렬을 바꾼 결과는 같은 외부시험으로 합치지 않는다.

## 5. 실행 순서

```bash
# 전체 EventAid-R 전 작은 ZIP 하나로 loader만 비보고 점검
bash scripts/get_aid.sh R-bear
asgcn-recon inspect --config configs/aid_smoke.json --samples 2 --validate-all

# smoke용 EventHDR train과 GPU만 먼저 확인
python scripts/check_env.py --require-cuda --require-eventhdr-smoke \
  --lock constraints/py312.txt

# smoke manifest의 네 H5만 decode/hash하는 1 epoch 점검
asgcn-recon inspect --config configs/hdr_smoke.json --samples 2 --validate-all
asgcn-recon train --config configs/hdr_smoke.json

# 본학습·고정 내부/외부 평가 전 전체 파일 수 확인
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt

# 모든 event block 검증
asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all

# scene split final 이후 ANN 본학습
asgcn-recon train --config configs/hdr_train.json

# EventHDR train만으로 IF-SNN 보정·변환
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500

# 고정 내부시험
asgcn-recon evaluate \
  --config configs/hdr_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics literal_eq15

# 같은 calibrated checkpoint를 쓰는 비공식 standard-IF 대조군
asgcn-recon evaluate \
  --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics standard_if

# 고정 외부시험
asgcn-recon evaluate \
  --config configs/aid_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/aid_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics literal_eq15
```

SNN 명령은 ANN checkpoint, 모든 graph layer에서 유효한 non-empty calibration observation이 0개인
checkpoint, BN 미fold·parameter 미정규화 상태, 비정수 또는 `simulation_steps < 1`을 거부한다.
checkpoint metadata는 persistent graph-layer flag 및 변환 뒤 unit threshold와도 교차검증한다.
모든 ANN/SNN/training checkpoint는 model tensor byte의 SHA-256을 저장하고 load 전에 검증한다.
반대로 Eq. (6)이
적용된 SNN checkpoint를 ANN 모드에 넣는 것도 거부하므로 ANN 평가는 변환 전 `best.pt`를 쓴다.
`scripts/eval.sh`에서는 동일한 override를 `SNN_DYNAMICS=literal_eq15` 또는
`SNN_DYNAMICS=standard_if`로 전달한다.

최종 EventHDR eval config는 H5 정확히 19개를, EventAid-R final config는 manifest와 이름이 같은
ZIP 정확히 14개를 강제한다. 일부 EventAid-R만 허용하는 `aid_smoke.json` 결과는 보고용 외부시험이
아니다.

## 6. 품질 지표

- PSNR: `[0,1]` data range
- SSIM: 11×11, σ=1.5 Gaussian valid window; 작은 영상은 fitting odd window
- RMSE
- `temporal_l1`: 같은 scene·sensor shape에서 sequence index가 정확히 1 증가하는 frame 사이에만
  `L1((pred_t-pred_t-1), (gt_t-gt_t-1))`
- LPIPS: `eval.lpips=true`일 때만 선택적으로 실행

결과는 micro, group macro, per-group으로 계산한다. final holdout의 group은 physical scene,
provisional/EventHDR 공식 eval은 H5 파일, EventAid-R은 `R-*.zip` scene이다. JSON의 기존
`macro`/`per_scene` 이름은 호환성을 위해 유지한다. 첫 frame과 장면·index·shape 불연속 뒤 첫
frame은 `temporal_l1` 집계에서 제외되고 CSV에는 null이 들어간다.

기존 논문과 SSIM을 비교할 때는 해당 논문의 구현, crop, border, color space, tone mapping까지
동일하게 맞춰 별도 검증한다. 현재 Gaussian 구현을 사용했다는 이유만으로 공식 수치와 완전히
동일하다고 가정하지 않는다.

## 7. 지연·메모리 지표

- evaluate: graph 생성과 model forward latency, 첫 frame cold start 포함
- benchmark: dataset read와 host-to-device 이동 제외, warmup 이후 CUDA Event 측정
- mean, p50, p90, p95, p99, max, FPS
- `raw_events_per_second`: spatial crop/cap 전 source interval event 수 / model compute time
- `retained_events_per_second`: crop과 adaptive cap 뒤 event 수 / model compute time
- `graph_nodes_per_second`: model의 추가 factor `R` 뒤 graph node 수 / model compute time
- `mean_raw_events`, `mean_retained_events`, `retention_ratio`, 평균 edge
- isolate 비율과 max degree; `max_graph_edges=2,000,000` 초과는 graph를 자르지 않고 실패
- SNN layer별 `총 spike / 총 neuron-step` 발화율과 전체 가중 발화율
- timestamp 기반 RTF, deadline miss ratio
- peak allocated/reserved GPU memory

세 rate 모두 benchmark가 제외한 dataset read/I/O throughput이 아니라, 같은 model compute 시간으로
정규화한 workload rate다. 기존 `events_per_second`는 deprecated 하위 호환 alias이며 항상
`retained_events_per_second`와 같다. metadata가 없는 custom dataset은 tensor에 실제 남아 있는 event
수를 raw와 retained 양쪽의 안전한 fallback으로 쓴다.

`snn` 경로는 T번 IF membrane timestep을 실제로 실행하고 fixed edge의 B-spline basis/index는
sample당 한 번 계산해 layer와 timestep에서 재사용한다. 따라서 `T=4/8/16/32`는 이 PyTorch
구현에서 dynamics별 timestep 수에 따른 품질·GPU latency ablation이다. decoder는 analog이므로 이를
neuromorphic accelerator의 latency나 에너지로 환산하지 않는다.

artifact는 `<eval.output_dir>/ann/` 또는
`<eval.output_dir>/snn_<dynamics>_T<steps>/`에 `metrics.json`, `frames.csv`, `predictions/`,
`benchmark.json`으로 나뉜다. 같은 mode/dynamics/T artifact가 이미 있으면 덮어쓰지 않고 실패한다.
`benchmark.json`은 benchmark가, 나머지는 evaluate가 기록한다.
prediction PNG는 평가 순번, 안전한 slug, 전체 sample ID hash를 결합해 파일명 충돌과 OS 금지 문자를
차단한다.

## 8. 최소 비교표

1. `graph_layers=3` vs 6
2. ANN vs `literal_eq15` vs `standard_if`, 각각 `T=4/8/16/32`
3. `max_events=4096/8192/16384`
4. 정규화된 `x,y,t`에서 `graph_radius=0.04/0.08/0.12`
5. ConvGRU on/off
6. EventHDR 내부 macro/per-scene vs EventAid-R 외부 macro/per-scene

모든 비교는 split, seed, tone mapping, crop, 해상도, checkpoint selection rule을 고정한다. 비교마다
config 원문, Git commit, `check_env.py` 출력, GPU 이름, CUDA/PyTorch, peak memory와 wall-clock을 함께
보존한다.

exact resume protocol은 선택 frame identity, group 길이, transform, manifest와 선택된
train/validation 원본의 SHA-256을 저장한다. smoke에서는 네 H5만, 본학습에서는 최종 manifest의
모든 H5를 hash한다. 절대경로와 mtime은 protocol에서 비교하지 않아 상대 파일 identity와 byte가
같은 복사본은 다른 mount에서도 재개할 수 있다. 같은 경로의 resume은 run 폴더 sidecar에서
size/mtime/ctime이 모두 같은 파일의 기존 full hash를 재사용한다.
원본을 교체·복원했거나 전수 확인하려면 `rehash_data=true`로 cache를 무시한다.

## 9. 중단 조건

- manifest가 provisional이면 본학습 금지; `status`만 바꾸고 final scene schema를 생략해도 금지
- 전체 dataset validation 실패 시 해당 파일 제외가 아니라 원본 재다운로드/검증
- A100 10GB smoke OOM이면 full training 전에 기본 graph/model 설정을 재검토
- NaN loss/metric, 비단조 timestamp, 범위 밖 좌표 발생 시 결과 폐기
- EventAid-R 결과를 본 뒤 hyperparameter나 threshold를 바꾸면 기존 결과를 잠금시험으로 표기 금지
- A6000/A100 latency를 FPGA/ASIC latency 또는 에너지로 환산해 주장 금지
~~~~~~

# docs/SERVER.md

~~~~~~markdown
# Linux GPU 서버 실행 가이드

MobaXterm은 SSH/SFTP 클라이언트이며 계산은 접속한 Linux 서버에서 수행된다. 아래 명령은 저장소
루트 기준이다.

## 지원 경로

- 직접 접속 GPU 서버: `tmux` + Bash wrapper
- interactive GPU allocation: 학교의 `ssai_agpu -g=1` 같은 명령 안에서 Bash wrapper
- SLURM: `server/*.sbatch`
- PBS/Torque: `server/*.pbs`
- Docker + NVIDIA Container Toolkit

현재 실측 완료 범위는 로컬 CPU다. GitHub Actions workflow는 구성했지만 최종 push commit의 성공
여부는 Actions run에서 별도로 확인해야 한다. A100/A6000 CUDA 결과도 서버에서 생성해야 한다.

## 설치

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

python3.12 --version
python3.12 -c "import venv, ensurepip; print('venv/ensurepip: OK')"
curl --version | head -n 1
tmux -V
ldd --version | head -n 1

cp .env.example .env
read -r -p "Official PyTorch wheel index URL: " TORCH_INDEX_URL
export TORCH_INDEX_URL
bash scripts/setup.sh

source .venv/bin/activate
python scripts/check_env.py --lock constraints/py312.txt
python -m pip check
python -m pytest -q
```

`venv`/`ensurepip`은 설치, `curl`은 EventAid-R downloader에 필요하다. `tmux`는 직접 GPU 서버
경로에서만 필요하므로 scheduler만 쓰면 생략할 수 있다. 빠진 명령은 학교 module 또는 OS package로
준비한다.

`constraints/py312.txt`는 Python 3.12.13, torch 2.13.0에서 검증한 core/dev profile이다. CUDA
wheel build는 `nvidia-smi`의 driver와 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)
결과에 맞춘다. 선택한 index에 torch 2.13.0 wheel이 실제로 있는지 확인해야 한다.

설치 스크립트는 다음을 강제한다.

- Python 3.10 이상
- `py312.txt` 사용 시 venv Python 정확히 3.12
- command-line environment가 `.env`보다 우선
- 기존 venv의 실제 Python 재검사
- Linux locked torch 2.13.0 profile에서 glibc 2.28 이상
- 선택한 constraints를 torch와 editable install 양쪽에 적용
- 마지막 `pip check`

LPIPS는 기본 경로에 포함하지 않는다. 필요할 때만 torch와 맞는 torchvision을 확인하고 `eval`
extra를 설치한다.

```bash
python -m pip install -e '.[eval]'
```

`scripts/setup.sh`은 Linux glibc 2.28 미만에서 locked torch 2.13.0 조합을 venv 생성·network
download 전에 fail-fast하고, `scripts/check_env.py --lock constraints/py312.txt`도 같은 조건을
검사한다. 해당 구형 HPC OS에서는 무리하게 source build를 시작하지 말고 학교의 최신
container/module 또는 검증된 별도 환경을 사용한다.

## 데이터와 저장 공간

```text
data/
├── EventHDR/
│   ├── train/*.h5
│   └── eval/*.h5
└── EventAid-R/
    └── R-*.zip
```

대용량 데이터가 shared storage에 있으면 설치 스크립트가 만든 빈 폴더만 제거하고 symlink한다.

```bash
rmdir data/EventHDR/train data/EventHDR/eval data/EventHDR data/EventAid-R
ln -s /shared/datasets/EventHDR data/EventHDR
ln -s /shared/datasets/EventAid-R data/EventAid-R
```

`rmdir`은 폴더가 비어 있지 않으면 실패하므로 기존 데이터를 지우지 않는다. data와 runs가 서로
다른 filesystem일 수 있어 환경 진단은 두 위치의 남은 공간을 각각 표시한다.

전체 EventAid-R를 받기 전 `R-bear` 하나로 ZIP loader만 확인할 때는 final 14-file guard가 있는
`aid_ann.json` 대신 비보고용 `aid_smoke.json`을 쓴다.

```bash
bash scripts/get_aid.sh R-bear
asgcn-recon inspect --config configs/aid_smoke.json --samples 2 --validate-all
```

```bash
python scripts/check_env.py --require-full-data
asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all
```

`--validate-all`은 모든 selected frame/event block을 decode하므로 50GB 전체에서는 오래 걸린다.
`hdr_ann/hdr_snn`은 EventHDR eval H5 정확히 19개를, `aid_ann/aid_snn`은 manifest와 일치하는
EventAid-R ZIP 정확히 14개를 강제한다.

## GPU allocation 검사와 smoke

로그인 노드가 GPU를 숨겨도 정상일 수 있다. 실제 compute allocation 안에서 다음을 실행한다.

```bash
python scripts/check_env.py --require-cuda --require-eventhdr-smoke \
  --lock constraints/py312.txt
mkdir -p logs
bash scripts/train.sh configs/hdr_smoke.json 2>&1 | tee logs/smoke.log
```

smoke는 실제 EventHDR에서 최대 32 train sample과 32 scored validation sample, 1 epoch를 사용한다.
validation에는 group당 최대 8개의 unscored recurrent context frame이 추가될 수 있다. 임시 split을
허용한 비보고용 검사다. `manifests/eventhdr_smoke.json`이 지정한 `1.h5`, `2.h5`, `48.h5`,
`49.h5`만 dataset content hash 대상으로 읽으며, EventHDR eval과 EventAid-R는 smoke에 필요하지 않다.
`runs/smoke/history.json`에서 CUDA peak allocated/reserved memory를 확인한다.

## 직접 GPU 서버

물리 scene split manifest가 `final`인 경우에만 본학습이 열린다.
최종 manifest는 `scene_groups`에 동일 물리 장면의 H5 목록을 묶고, 겹치지 않는 scene ID를
`train_scenes`와 `val_scenes`에 배정해야 한다. 예를 들면 다음 schema다.

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

예시는 실제 scene mapping이 아니다. legacy `train_files`/`val_files`를 유지한 채 `status`만 바꾸면
거부된다. legacy file-list schema는 `manifests/eventhdr_smoke.json`의 provisional smoke에서만 쓴다.
final manifest는 `data/EventHDR/train` 아래 모든 H5를 정확히 한 scene에 포함하고 모든 scene을
train/validation 중 하나에 배정해야 한다. root의 누락·미선언 H5도 본학습 전에 거부한다.

```bash
tmux new-session -s asgcn -c "$PWD" \
  "bash -lc 'source .venv/bin/activate && bash scripts/train.sh configs/hdr_train.json'"
```

중단 후 resume:

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/train.sh configs/hdr_train.json
```

첫 실행은 train/validation 원본의 full SHA-256을 계산한다. 같은 경로의 resume은 run 폴더 sidecar에서
size/mtime/ctime이 모두 같은 파일의 hash를 재사용한다. 원본을 교체·복원했다면
`train.rehash_data=true`로 한 번 전수 hash한다. 절대 데이터 root와 filesystem mtime/ctime은
checkpoint protocol의 비교 항목이 아니며 상대 파일 identity와 full digest가 같아야 한다.

## Interactive PBS/Torque wrapper

학교에서 다음처럼 GPU shell을 발급한다면:

```bash
ssai_agpu -g=1
```

발급된 shell 안에서:

```bash
cd /absolute/path/to/asgcn-event-reconstruction
source .venv/bin/activate
python scripts/check_env.py --require-cuda --lock constraints/py312.txt
bash scripts/train.sh configs/hdr_smoke.json
```

을 실행한다. wrapper 인자와 walltime 정책은 학교 문서를 따른다.

## PBS/Torque batch

`#PBS -l select=...:ngpus=...`의 resource 이름은 클러스터마다 다르므로 제출 전에 header를
수정한다. 저장소 루트에서 제출하는 것이 기본이다.

```bash
qsub server/train.pbs
```

저장소 밖에서 제출할 때는 절대 경로를 넘긴다. 스크립트는 잘못된 root를 자동으로 사용하지 않고
중단한다.

```bash
qsub -v PROJECT_ROOT=/absolute/path/to/repo server/train.pbs
```

SNN 외부평가 예시:

```bash
qsub -v PROJECT_ROOT="$PWD",CONFIG_PATH=configs/aid_snn.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,\
SIMULATION_STEPS=16,SNN_DYNAMICS=literal_eq15 server/eval.pbs
```

`SNN_DYNAMICS=standard_if`로 바꾸면 같은 calibrated checkpoint의 비공식 standard-IF 대조군을
실행한다. 결과는 dynamics와 timestep이 포함된 별도 하위 폴더에 저장된다.

`#PBS -V`는 login shell의 불필요한 credential/module까지 전달할 수 있어 사용하지 않는다. 필요한
값만 `-v`로 전달한다.

## SLURM

저장소 루트에서 제출한다. Slurm은 compute node에서 job script의 spool 사본을 실행할 수 있으므로
스크립트는 `PROJECT_ROOT`, `SLURM_SUBMIT_DIR` 순으로 root를 찾고 `pyproject.toml`과 wrapper를
검증한다. 저장소 밖에서 제출할 때는 절대 경로를 명시한다.

```bash
sbatch server/train.sbatch

sbatch --export=ALL,PROJECT_ROOT="$PWD",CONFIG_PATH=configs/aid_ann.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt server/eval.sbatch

sbatch --export=ALL,PROJECT_ROOT="$PWD",CONFIG_PATH=configs/aid_snn.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,\
SIMULATION_STEPS=16,SNN_DYNAMICS=standard_if server/eval.sbatch
```

기본 요청은 GPU 1개, CPU 8개, RAM 32GB다. partition, account, GPU type, walltime과 module은
클러스터 정책에 맞춰 수정한다. 본학습 40 epoch는 현재 serial frame 처리 때문에 기본 2일을 넘을
수 있으므로 smoke에서 측정한 step time으로 walltime을 먼저 계산한다.

평가 artifact는 `<eval.output_dir>/ann/` 또는
`<eval.output_dir>/snn_<dynamics>_T<steps>/` 아래에 저장된다. `metrics.json`, `frames.csv`,
`predictions/`는 evaluate가, `benchmark.json`은 benchmark가 쓴다. 같은 mode/dynamics/T 결과가
있으면 덮어쓰지 않고 실패하므로 기존 결과를 옮기거나 새 output directory를 사용한다.

## Docker

Dockerfile은 Python 3.12와 `constraints/py312.txt`를 사용한다.

```bash
docker build \
  --build-arg TORCH_VERSION=2.13.0 \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t asgcn-event-reconstruction .

docker compose run --rm experiment \
  inspect --config configs/hdr_train.json --samples 2
```

`compose.yaml`은 data를 read-only, runs를 writable volume으로 연결한다.

## 흔한 오류

- `CUDA available: false`: GPU allocation 안인지, CPU torch wheel인지, driver와 wheel index가
  맞는지 확인한다.
- `glibc 2.28 or newer` 또는 `No matching distribution`: locked torch 2.13.0 profile에는 Linux
  glibc 2.28 이상이 필요하다. 학교의 최신 container/module과 CUDA index 조합을 확인한다.
- `Dependency profile requires Python 3.12`: 기존 `.venv`를 삭제해야 한다면 그 폴더가 정확히 이
  저장소의 venv인지 확인한 뒤 재생성한다.
- `status='provisional'` 또는 `requires scene_groups`: 본학습 차단이 정상이다. 최종 manifest에
  `scene_groups`, `train_scenes`, `val_scenes`를 모두 작성하거나 smoke config를 쓴다. status만
  `final`로 바꾸는 것은 허용되지 않는다.
- `missing manifest files`: EventHDR 1–51 배치와 symlink 위치를 확인한다.
- SNN calibrated checkpoint 오류: ANN `best.pt`를 먼저 `calibrate`해 `best_snn.pt`를 만든다.
- OOM: 기본값에서 먼저 측정하고 `max_events`, `graph_radius`, `decoder_channels` 순으로 낮춘다.
  `graph_radius`를 바꾸면 graph 구조도 달라지므로 해당 run의 config를 함께 보존한다.
- `max_graph_edges` 오류: 기본 2,000,000 directed-edge 메모리 guard가 밀집 graph를 탐지한 것이다.
  edge를 임의로 버리지 말고 `graph_radius`/`max_events`를 낮춰 재측정한다. guard 상향은 peak reserved
  memory를 확인한 별도 config에서만 수행한다.
- SSH 종료: interactive shell이 아니라 tmux 또는 scheduler job을 사용한다.
~~~~~~

# hand_off.md

~~~~~~markdown
# ASGCN Event Reconstruction 인수인계

## 1. 현재 결론

이 저장소는 EventHDR로 학습하고 EventHDR 공식 eval과 EventAid-R에서 평가하는
**ASGCN paper-core 기반 event-to-frame 연구 프로토타입**이다. MobaXterm으로 Linux GPU 서버에
SSH 접속한 뒤 clone, 설치, 데이터 배치, 검사, 학습, ANN→SNN 보정, 평가, benchmark까지 수행할 수
있다.

이 저장소는 ASGCN 저자의 공식 저장소가 아니다. 2026-08-29 기준 확인 가능한
[AAAI 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32154)와
[공식 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)에서 공개 저자 코드나
checkpoint 링크를 확인하지 못했다. 따라서 다음 표현만 사용한다.

- 가능한 표현: `ASGCN paper-core 기반 복원 적응`, `공개 수식의 정적 graph/SNN core 구현`
- 금지 표현: `저자 공식 코드`, `공식 ASGCN 완전 재현`, `논문 분류 성능 재현`

원 논문은 event graph classification을 수행하지만 본 과제는 event-to-frame 복원이다. 논문의
sampling, radius graph, B-spline graph convolution, BN folding, ANN→SNN parameter normalization,
IF recurrence를 graph encoder에 사용하고, 그 뒤에 과제용 rasterization, residual U-Net, analog
ConvGRU를 연결했다.

비공개 원격 저장소:

```text
https://github.com/costunder/asgcn-event-reconstruction
```

## 2. 최종 파이프라인

```text
EventHDR/EventAid-R event interval [N, x,y,t,p]
  -> spatial crop
  -> adaptive integer-stride max_events safety cap
  -> fixed paper sampling factor R
  -> normalized event nodes [x,y,t,p]
  -> strict d(i,j) < D undirected radius graph
  -> scalar pseudo-coordinate u=d/D
  -> degree-1 open B-spline graph encoder
       ANN: affine + BN + ReLU
       SNN: BN fold + Eq.(6) calibration + explicit IF timesteps
  -> graph feature rasterization
  -> residual U-Net + analog ConvGRU
  -> [0,1] luminance frame
  -> PSNR / Gaussian SSIM / RMSE / temporal L1 / latency
```

전처리의 두 sampling은 서로 다르다.

1. dataset의 adaptive cap은 crop 뒤 `ceil(N/max_events)` 정수 stride를 사용한다. 서버 메모리
   안전장치이며 논문의 공식 `R`이 아니다.
2. model의 `event_sampling_factor`가 논문의 고정 `R`이다. 기본값은 1이다.

각 sample에는 `raw_event_count`, `cropped_event_count`, `retained_event_count`,
`dataset_sampling_factor`가 들어가며 model diagnostics에는 model factor와 두 값의 곱도 기록된다.

## 3. 공개 논문과 맞춘 부분

### 3.1 event graph

- event 하나를 node 하나로 사용한다.
- node feature는 `x,y,t,p`다.
- 기본 graph 거리는 `[0,1]`로 정규화한 `x,y,t` 3차원에서 계산한다.
- 두 node의 Euclidean distance가 `D`보다 작을 때만 edge를 만든다.
- self edge 없이 두 방향을 모두 materialize해 simple undirected graph를 표현한다.
- edge message는 실제 incoming degree로 평균한다.
- `distance/D`는 PyG SplineConv의 `[0,1]` pseudo-domain에 맞춘 명시적 재매개화다. 논문의 원시
  distance를 그대로 넣었다고 주장하지 않는다.

### 3.2 B-spline operator

순수 PyTorch `PaperSplineConv`를 사용하므로 Linux에서 `torch-spline-conv` C++/CUDA extension을
별도로 빌드하지 않는다. 지원 범위는 scalar pseudo-coordinate, open degree 1, mean aggregation이다.
기본 kernel size는 5다.

공식 `pytorch_spline_conv` source와 맞춘 항목:

- degree-1 basis index와 partition of unity
- exact `u=1` modulo endpoint와 pseudo-coordinate gradient
- spline weight bound `1/sqrt(K*Cin)`
- root weight bound `1/sqrt(Cin)`
- mean message aggregation, root transform, output bias

root weight, bias, degree, openness, kernel size는 논문이 공개하지 않은 선택이다. root/bias는 식 (12)의
update 선택 및 공식 PyG 기본 동작에 대응하지만 식 (11)에 문자 그대로 쓰여 있는 항은 아니다.
고정 edge basis/index는 sample당 한 번 계산해 모든 graph layer와 IF timestep에서 재사용한다.

### 3.3 ANN→SNN 변환

1. ANN을 일반 역전파로 학습한다.
2. graph layer BN을 식 (13)–(14)처럼 kernel/root/bias에 정확히 fold한다.
3. EventHDR train calibration sample에서 feature별 ReLU maximum `lambda_l`를 측정한다.
4. 식 (6)의 `W_l * lambda_(l-1)/lambda_l`, `b_l/lambda_l`를 kernel, root, bias에 적용한다.
5. 유효한 non-empty calibration sample이 없는 layer는 변환을 거부한다.
6. calibration에서 항상 0이었던 dead channel은 epsilon으로 폭증시키지 않고 unit scale을 사용하고,
   변환 전 dead-channel 수를 checkpoint metadata에 보존한다.
7. 마지막 normalized spike output에는 `lambda_L`를 곱해 학습된 analog decoder 단위로 보낸다.

SNN checkpoint metadata와 각 layer의 persistent `bn_bypassed`, `snn_normalized`, 변환 뒤 exact-unit
threshold state가 다르면 load를 거부한다. 모든 checkpoint는 model tensor byte SHA-256도 load 전에
재계산한다. 변환한 `best_snn.pt`를 ANN mode로 읽는 것도 거부한다.

## 4. 식 (15)의 중요한 모호성

논문 식 (15)는 다음 self-feedback을 포함한다.

```text
v_tilde(t) = v(t-1) + c(t) + h(t-1)
```

이 식을 문자 그대로 실행하면 작은 양의 정전류도 첫 발화 뒤 이전 spike가 재주입되어 장기 firing
rate가 1에 가까워질 수 있다. 이는 표준 soft-reset IF의 `firing rate ≈ normalized ANN activation`
유도와 수학적으로 양립하지 않는다. 저자 코드나 정정 자료가 없으므로 임의로 오타 처리하지 않았다.

- `snn_dynamics=literal_eq15`: 모든 기본 config. 식 (15)–(17)을 문자 그대로 실행한다.
- `snn_dynamics=standard_if`: `+h(t-1)`를 제거한 rate-conversion 대조군이다. 공식 ASGCN 값이 아니다.

테스트는 `c=0.1`, `theta=1`, `T=100`에서 literal 결과 0.96과 standard 결과 0.10을 각각 고정한다.
따라서 literal 결과에 대해 “ANN activation을 정확히 rate로 근사한다”고 쓰면 안 된다. 마지막
`lambda_L` 곱은 decoder 단위 변환이며 literal recurrence의 ANN parity 증명이 아니다.

## 5. 논문에 없는 과제용 가정

기본 config의 주요 값:

| 항목 | 값 | 상태 |
|---|---:|---|
| architecture version | 2 | 구형 edge-MLP checkpoint 차단 |
| hidden features | 64 | 복원 과제 가정 |
| graph layers | 6 | 복원 과제 가정 |
| normalized radius D | 0.08 | ablation 필요 |
| graph dimensions | x,y,t | polarity는 node feature만 사용 |
| B-spline | open degree 1, K=5 | 논문 미공개값 |
| root weight | true | 식 (12)/PyG 기반 선택 |
| raster downsample | 4 | 품질·메모리 ablation 필요 |
| decoder channels | 48 | 복원 과제 가정 |
| recurrent decoder | true | analog ConvGRU |
| max events | 8192 | dataset/server safety cap |
| max directed edges | 2,000,000 | OOM fail-fast guard |
| SNN dynamics | literal_eq15 | 공개 식 우선 선택 |

random crop은 scene+source-file 단위로 고정한다. 같은 연속 sequence에서 frame마다 ROI를 바꾸면
ConvGRU pixel state와 temporal loss가 공간적으로 어긋나므로 per-frame crop을 사용하지 않는다.
현재 crop은 resume/worker에 대해 결정적이고 epoch별 증강은 아니다.

radius graph는 `D`가 만든 결과를 강제로 연결하지 않는다. isolated node는 root transform만 받는다.
평가 CSV와 benchmark에는 isolate 비율과 max degree가 기록된다. directed edge가 2,000,000개를 넘으면
edge를 몰래 잘라 다른 graph로 바꾸지 않고 오류로 중단한다.

## 6. 아직 구현하지 않은 범위

- 새 event의 K-hop ego-network만 갱신하는 동적 asynchronous 실행
- sliding window에서 만료된 node 제거와 membrane state 관리
- 식 (18)–(19)의 clustering, pooling, edge remapping
- 원 논문의 classification MLP와 원 데이터셋 성능표
- 식 (20)–(21)의 완전한 energy model과 실제 FPGA/ASIC 측정
- event compression/transport protocol, RTL, FPGA/ASIC accelerator
- full-resolution learned upsampling ablation
- 실제 센서 I/O, H2D를 포함한 end-to-end ingest throughput

이 항목 때문에 현재 결과를 반도체 구현 완료나 공식 논문 재현이라고 부를 수 없다.

## 7. 데이터셋 역할

| 데이터 | 저장 용량 참고 | 역할 |
|---|---:|---|
| EventHDR | 약 25.72 GB | ANN 학습, holdout validation, 공식 eval, SNN calibration |
| EventAid-R | 공식 ZIP 합계 약 24.68 GB | 학습·보정 없는 외부 일반화 평가 |

두 데이터 target은 `[0,1]` luminance로 변환한 뒤
`log1p(5000*x)/log1p(5000)` tone mapping을 사용한다. 이는 수치 output domain만 맞추며 센서의
radiometric response가 같다는 뜻은 아니다.

EventAid-R에서 event index `i`를 다음 영상 `i+1`의 복원 입력으로 해석하는 `target_offset=1`은
공개 파일 배열을 바탕으로 둔 구현 가정이다. 논문 저자 코드로 확인된 공식 pairing 규칙이 아니므로
최종 보고서에는 이 가정을 명시하고, 필요하면 offset ablation을 별도로 수행한다.

예상 배치:

```text
data/
├── EventHDR/
│   ├── train/*.h5
│   └── eval/*.h5
└── EventAid-R/
    └── R-*.zip
```

EventAid-R는 ZIP을 추출하지 않고 직접 읽는다. `scripts/get_aid.sh`와 `scripts/get_aid.ps1`은 공식
URL/manifest를 사용해 필요한 scene을 내려받고 검증한다. EventHDR는 README가 연결한 공식 저장소의
배포 링크에서 받은 H5를 위 경로에 둔다.

## 8. split과 데이터 누수 방지

`manifests/eventhdr_split.json`은 물리 scene 대응표가 확보되지 않아 `provisional`이다. 본학습 config는
이 상태에서 중단된다. 단순히 `status`만 `final`로 바꾸면 안 된다.

최종 schema:

```json
{
  "status": "final",
  "scene_groups": {
    "physical-scene-a": ["1.h5", "2.h5"],
    "physical-scene-b": ["48.h5", "49.h5"]
  },
  "train_scenes": ["physical-scene-a"],
  "val_scenes": ["physical-scene-b"]
}
```

실제 mapping을 확인한 뒤 scene 간 file 중복이 없도록 작성한다. final manifest에서는
`metadata.scene`이 H5 파일명이 아니라 physical scene ID가 된다.

final manifest는 train root 아래의 모든 H5를 정확히 한 번 `scene_groups`에 포함하고, 모든 scene을
train 또는 validation 중 정확히 한 split에 배정해야 한다. 누락 파일, root에 없는 선언, 중복 소속,
미배정 scene은 모두 거부된다.

smoke만 `manifests/eventhdr_smoke.json`의 provisional legacy file list를 허용한다. smoke는 train
`1.h5`, `2.h5`, validation `48.h5`, `49.h5`만 열고 hash하므로 전체 25 GB를 불필요하게 읽지 않는다.

## 9. MobaXterm/Linux 서버 시작 절차

MobaXterm은 SSH/SFTP 클라이언트다. 실제 명령은 접속한 Linux 서버에서 실행한다.

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

nvidia-smi
ldd --version | head -n 1
python3.12 --version

cp .env.example .env
read -r -p "Official PyTorch wheel index URL: " TORCH_INDEX_URL
export TORCH_INDEX_URL
bash scripts/setup.sh
source .venv/bin/activate
python scripts/check_env.py --require-cuda --lock constraints/py312.txt
python -m pytest -q
```

기본 locked profile은 Python 3.12, torch 2.13.0이다. Linux wheel은 glibc 2.28 이상을 요구하므로
Ubuntu 18.04 기반 image에서 실패할 수 있다. 이때 Ubuntu 22.04/24.04 계열 container를 사용한다.
`TORCH_INDEX_URL`에는 PyTorch 공식 selector가 제시하는 서버 driver 호환 wheel index URL을 넣는다.
torch 2.13.0 wheel이 해당 index에 실제로 존재해야 하며, 임의의 CUDA suffix를 가정하지 않는다.
정확한 명령은 `README.md`와 `docs/SERVER.md`를 우선한다.

## 10. 데이터 검사와 실행 명령

### smoke

```bash
chmod +x scripts/get_aid.sh
bash scripts/get_aid.sh bear
asgcn-recon inspect --config configs/aid_smoke.json --samples 2 --validate-all

python scripts/check_env.py --require-cuda --require-eventhdr-smoke \
  --lock constraints/py312.txt
asgcn-recon inspect --config configs/hdr_smoke.json --samples 2 --validate-all
asgcn-recon train --config configs/hdr_smoke.json
```

`aid_smoke.json`은 단일/부분 ZIP loader smoke 전용이며 결과 보고용이 아니다. 최종 EventAid-R 평가인
`aid_ann.json`과 `aid_snn.json`은 manifest의 14개 ZIP을 정확히 요구한다.

### 본학습 전 전체 검사

```bash
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt
asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all
```

### ANN 학습

```bash
asgcn-recon train --config configs/hdr_train.json
```

현재 provisional split guard 때문에 physical scene mapping 확정 전에는 위 본학습이 의도적으로
중단된다.

### ANN→SNN calibration

```bash
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500
```

calibration은 EventHDR train만 사용하며 EventHDR eval/EventAid-R를 사용하지 않는다.

### 평가

```bash
asgcn-recon evaluate --config configs/hdr_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics literal_eq15

asgcn-recon evaluate --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics standard_if

asgcn-recon evaluate --config configs/aid_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate --config configs/aid_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics literal_eq15
```

`scripts/eval.sh`/SLURM/PBS wrapper에서도 `SNN_DYNAMICS=literal_eq15` 또는 `standard_if`를 명시할
수 있다. 두 dynamics 결과는 별도 실험으로 보고한다.

## 11. 학습 protocol과 exact resume

논문이 공개한 optimizer 조건에 맞춰 본 config는 다음을 사용한다.

- Adam + gradient centralization
- learning rate `1e-3`
- L2 weight decay `5e-3`
- MultiStepLR `[20,30]`, gamma `0.1`은 논문 미공개값에 대한 복원 과제 가정

Spline weight `[K,Cin,Cout]`와 root `[Cin,Cout]`는 output axis를 제외한 dimension에서 gradient를
centralize한다. Conv/Linear는 output-first layout에 맞춰 첫 axis를 제외한다.

`last.pt`에는 model, optimizer, scheduler, GradScaler, epoch, history, RNG, validation/data fingerprint,
training protocol을 넣는다. resume 시 다음처럼 optimization trajectory를 바꾸는 값이 다르면 거부한다.

- optimizer, learning rate, weight decay, scheduler
- loss weights, grad clipping
- batch size, max train sample 수, validation 주기
- AMP 요청값과 실제 device 적용 상태
- model config, split, transforms, source SHA-256

`epochs`, `log_every`, resume path, output control처럼 이미 완료한 optimizer step을 바꾸지 않는 값만
resume에서 달라질 수 있다. `best.pt`는 ANN inference-only, `best_snn.pt`는 SNN inference-only다.

## 12. fail-fast 장치

- non-finite event coordinate/timestamp/polarity 거부
- 비단조 timestamp/event index 거부
- 범위 밖 image/event coordinate 거부
- loss component/total non-finite 즉시 중단
- gradient clipping 뒤 non-finite gradient 즉시 중단
- validation metric/macro SSIM non-finite 즉시 중단
- `simulation_steps`의 bool/float/0 입력 거부
- 빈 calibration layer 거부
- ANN/SNN checkpoint mode 혼용 거부
- checkpoint metadata/state flag 불일치 거부
- partial BN-fold/parameter normalization 거부
- final scene schema 누락/scene leakage 거부
- radius graph edge budget 초과 거부

## 13. 평가와 benchmark 의미

품질:

- PSNR, RMSE
- 11×11 sigma 1.5 Gaussian-window SSIM
- scene macro/micro/per-scene 집계
- 같은 scene에서 연속 index일 때만 temporal L1
- 선택적 LPIPS

품질 집계의 grouping은 구현 가정이다. final physical-scene manifest에서는 physical scene ID,
provisional/EventHDR official eval에서는 H5 파일, EventAid-R에서는 ZIP scene을 group으로 사용한다.
JSON의 `per_scene`/`scene_count` 이름은 하위 호환을 위해 유지한다.

성능:

- evaluate: graph 생성과 model forward를 포함, dataset read는 timer 전
- benchmark: dataset read와 H2D 제외, CUDA Event 사용
- p50/p90/p95/p99/max, FPS, RTF/deadline miss
- raw events/s, retained events/s, graph nodes/s
- isolate ratio, max degree, mean node/edge
- SNN layer별 `총 spike / 총 neuron-step` 발화율과 전체 가중 발화율
- peak allocated/reserved GPU memory

`events_per_second`는 deprecated compatibility alias이며 retained events/s와 같다. 어느 값도 센서
ingest 또는 반도체 전송 throughput으로 해석하지 않는다. PyTorch GPU IF latency/발화율도 FPGA/ASIC
energy 측정값이 아니다.

평가 산출물은 `<output_dir>/ann/` 또는
`<output_dir>/snn_<dynamics>_T<T>/` 아래의 `metrics.json`, `frames.csv`, `predictions/`,
`benchmark.json`으로 모드별 분리된다. 기존 산출물이 있는 run directory는 덮어쓰지 않고 거부하므로
새 output root를 쓰거나 이전 결과를 명시적으로 보존·이동해야 한다. 최종 EventHDR 평가는 19개 H5,
최종 EventAid-R 평가는 manifest의 14개 ZIP 전체 coverage guard를 통과해야 한다.
prediction PNG는 평가 순번과 full sample-ID hash를 포함한 cross-platform filename을 쓴다.

## 14. checkpoint 종류

| 파일 | 용도 | optimizer/RNG | BN fold | Eq.(6) |
|---|---|---|---|---|
| `last.pt` | exact training resume | 포함 | 아니오 | 아니오 |
| `best.pt` | ANN 평가 | 미포함 | 아니오 | 아니오 |
| `best_snn.pt` | SNN 평가 | 미포함 | 예 | 예 |

checkpoint는 embedded `model_config`와 `architecture_version=2`를 요구한다. 구형 edge-MLP/raw state
dict를 paper-core model로 조용히 읽지 않는다. 세 checkpoint 모두 `model_state_sha256`를 가지며
일반 평가, calibration, exact resume가 tensor byte와의 일치를 강제한다.

## 15. 로컬 검증 결과

검증 환경:

```text
Windows 11
Python 3.12.13
torch 2.13.0+cpu
CUDA unavailable on this host
```

마지막 검증 결과:

- Ruff: 통과
- Python compileall: 통과
- pytest: 136 passed
- JSON config 7개, manifest 3개 parse: 통과
- `scripts/check_env.py --lock constraints/py312.txt`: 통과
- HWP/HWPX와 dummy implementation 파일: 없음

현재 로컬에 있는 일부 실제 데이터도 전수 decode/finite 검사했다.

| 데이터 | 파일 | sample | 검사 결과 |
|---|---|---:|---|
| EventHDR | `data/EventHDR/train/26.h5` | 500 | 전부 통과 |
| EventAid-R | `data/EventAid-R/R-bear.zip` | 65 | 전부 통과 |

실제 EventHDR 마지막 sample의 기본 ANN CPU forward:

```text
sample: 26.h5/image000000499
output: [1,1,240,256], finite
nodes: 4134
directed edges: 90156
isolated nodes: 3
max degree: 57
model parameters: 4,409,617
CPU forward: 약 0.37 s (성능 보증값 아님)
```

실제 EventHDR sample에서 6개 layer calibration, Eq. (6), literal-Eq15 `T=2` SNN forward도 finite
output을 확인했다. GPU/CUDA AMP, A100/A6000 peak memory와 full dataset 40-epoch 학습은 이 host에서
실측하지 않았다.

## 16. 주요 파일 지도

| 경로 | 역할 |
|---|---|
| `README.md` | clone부터 데이터·학습·평가까지 사용자 실행 가이드 |
| `docs/ASGCN.md` | 논문 근거, 구현 가정, 식 (15) 모호성, 비재현 범위 |
| `docs/EXPERIMENT.md` | split, protocol, ablation, metric 해석 |
| `docs/SERVER.md` | MobaXterm/Linux/CUDA/SLURM/PBS/Docker 운영 |
| `src/asgcn_recon/graph.py` | radius graph, SplineConv, BN folding, Eq. (6), IF loop |
| `src/asgcn_recon/model.py` | graph encoder, rasterizer, U-Net/ConvGRU decoder |
| `src/asgcn_recon/engine.py` | train/resume/calibrate/evaluate/benchmark/checkpoint |
| `src/asgcn_recon/data/eventhdr.py` | EventHDR H5 direct loader/validation |
| `src/asgcn_recon/data/eventaid_r.py` | EventAid-R ZIP direct loader/validation |
| `src/asgcn_recon/data/factory.py` | split manifest, physical scene schema, leakage checks |
| `configs/*.json` | `hdr_smoke`/`aid_smoke`, train, internal/external ANN/SNN 실험 설정 |
| `manifests/*.json` | EventHDR split/smoke와 EventAid-R 파일 manifest |
| `scripts/setup.sh` | venv와 CPU/CUDA locked dependency 설치 |
| `scripts/check_env.py` | Python/torch/CUDA/glibc/data/disk/lock 검사 |
| `server/*` | SLURM/PBS template |
| `tests/*` | 수식, operator, loader, protocol, resume, end-to-end 회귀 |
| `code_summary.md` | ChatGPT 교차검증용 전체 text file snapshot |

## 17. 다음 담당자의 우선순위

1. EventHDR 배포 정보에서 물리 scene↔H5 mapping을 확보하고 final manifest를 작성한다.
2. A6000 또는 A100에서 `hdr_smoke`를 먼저 실행해 peak allocated/reserved memory와 edge 분포를 본다.
3. 기본 config의 isolate ratio와 edge budget hit 여부를 확인한다.
4. `literal_eq15`와 `standard_if`를 T=4/8/16/32로 분리 보고한다.
5. `graph_layers=3/6`, radius 0.04/0.08/0.12, raster downsample 1/2/4를 ablation한다.
6. full training 전에 학습/검증 scene이 겹치지 않는지 manifest validation 결과를 보존한다.
7. full training 뒤 EventHDR official eval과 EventAid-R를 한 번만 고정 평가한다.
8. 과제의 저지연·고효율 주장에는 실제 hardware 또는 명시적 operation/energy model을 추가한다.

## 18. 최종 주의사항

- provisional manifest의 `status`만 바꾸지 않는다.
- EventAid-R를 보고 hyperparameter/threshold를 조정한 뒤 외부 잠금시험이라고 부르지 않는다.
- edge guard를 단순 상향하기 전에 GPU peak reserved memory를 측정한다.
- literal Eq. (15)를 standard rate IF처럼 설명하지 않는다.
- PyTorch GPU latency를 FPGA/ASIC latency나 energy로 환산하지 않는다.
- 전체 공식 데이터와 GPU full run을 하지 않은 상태에서 실험 완료라고 쓰지 않는다.

현재 코드는 더미 골격이나 edge-MLP proxy가 아니라 공개 수식 기반 graph/SNN core와 복원 decoder가
실제로 실행되는 연구 코드다. 동시에 논문과 과제의 경계, 저자 코드 부재, 공개 식의 모호성, 미구현
hardware 범위를 숨기지 않는 것이 이 저장소의 재현성 원칙이다.
~~~~~~

# Makefile

~~~~~~makefile
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
~~~~~~

# manifests/eventaid_r.json

~~~~~~json
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
~~~~~~

# manifests/eventhdr_smoke.json

~~~~~~json
{
  "status": "provisional",
  "note": "Smoke-only file split. Do not report it as a physical-scene-disjoint validation result.",
  "train_files": ["1.h5", "2.h5"],
  "val_files": ["48.h5", "49.h5"]
}
~~~~~~

# manifests/eventhdr_split.json

~~~~~~json
{
  "status": "provisional",
  "note": "Provisional legacy file lists only. For final training replace them with scene_groups, train_scenes, and val_scenes, then set status to final; changing status alone is rejected.",
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
~~~~~~

# pyproject.toml

~~~~~~toml
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
~~~~~~

# README.md

~~~~~~markdown
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
~~~~~~

# requirements.txt

~~~~~~text
-c constraints/py312.txt
-e .[dev]
~~~~~~

# scripts/check_env.py

~~~~~~python
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
from pathlib import Path, PurePosixPath

import torch

from asgcn_recon.data import load_eventhdr_split_manifest


def _eventhdr_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([*root.rglob("*.h5"), *root.rglob("*.hdf5")])


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


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
    parser.add_argument("--require-eventhdr-smoke", action="store_true")
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
    smoke_manifest_path = project_root / "manifests" / "eventhdr_smoke.json"
    smoke_required: set[str] = set()
    if smoke_manifest_path.is_file():
        smoke_manifest = load_eventhdr_split_manifest(smoke_manifest_path)
        smoke_required = {
            PurePosixPath(str(value).replace("\\", "/")).as_posix()
            for value in (
                list(smoke_manifest.get("train_files", []))
                + list(smoke_manifest.get("val_files", []))
            )
        }
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
        "eventhdr_smoke_h5": len(smoke_required & train_present),
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
    expected_counts = {
        "eventhdr_train_h5": (51, require_eventhdr_train),
        "eventhdr_eval_h5": (19, require_eventhdr_eval),
        "eventaid_r_zip": (14, require_eventaid_all),
    }
    for key, (expected, required) in expected_counts.items():
        actual = int(report[key])
        if required and actual < expected:
            problems.append(f"{key} has {actual} files; at least {expected} are required")
    if args.require_eventhdr_smoke:
        if not smoke_required:
            problems.append(f"Smoke manifest has no required H5 files: {smoke_manifest_path}")
        smoke_missing = sorted(smoke_required - train_present)
        if smoke_missing:
            problems.append(
                "EventHDR train directory is missing smoke manifest files: "
                + ", ".join(smoke_missing)
            )
    if require_eventhdr_train:
        manifest_path = project_root / "manifests" / "eventhdr_split.json"
        if manifest_path.is_file():
            manifest = load_eventhdr_split_manifest(manifest_path)
            required = {
                PurePosixPath(str(value).replace("\\", "/")).as_posix()
                for value in (
                    list(manifest.get("train_files", [])) + list(manifest.get("val_files", []))
                )
            }
            missing = sorted(required - train_present)
            if missing:
                problems.append(
                    "EventHDR train directory is missing manifest files: "
                    + ", ".join(missing[:8])
                    + (" ..." if len(missing) > 8 else "")
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
            aid_missing = sorted(aid_required - aid_present)
            if aid_missing:
                problems.append(
                    "EventAid-R directory is missing manifest files: "
                    + ", ".join(aid_missing[:8])
                    + (" ..." if len(aid_missing) > 8 else "")
                )
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
~~~~~~

# scripts/eval.sh

~~~~~~bash
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
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${CONFIG_PATH}" --samples "${INSPECT_SAMPLES}"
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
~~~~~~

# scripts/get_aid.ps1

~~~~~~powershell
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
~~~~~~

# scripts/get_aid.sh

~~~~~~bash
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
~~~~~~

# scripts/setup.sh

~~~~~~bash
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
echo "Next: ./scripts/get_aid.sh R-bear"
echo "Then: ${VENV_PYTHON} -m asgcn_recon.cli inspect --config configs/aid_smoke.json --samples 2"
echo "Then place EventHDR H5 files under data/EventHDR/train and data/EventHDR/eval."
~~~~~~

# scripts/train.sh

~~~~~~bash
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
~~~~~~

# server/eval.pbs

~~~~~~text
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
~~~~~~

# server/eval.sbatch

~~~~~~text
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
~~~~~~

# server/train.pbs

~~~~~~text
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NCPUS:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "PBS job: ${PBS_JOBID:-interactive}"
echo "Config: ${CONFIG_PATH}"
nvidia-smi || true

bash "${PROJECT_ROOT}/scripts/train.sh" "${CONFIG_PATH}"
~~~~~~

# server/train.sbatch

~~~~~~text
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
~~~~~~

# src/asgcn_recon/__init__.py

~~~~~~python
"""ASGCN-style event-to-frame reconstruction."""

__version__ = "0.1.0"
~~~~~~

# src/asgcn_recon/cli.py

~~~~~~python
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
    return result


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
    calibrate_cmd.add_argument("--samples", type=int, default=100)
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
                )
            )
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
~~~~~~

# src/asgcn_recon/data/__init__.py

~~~~~~python
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
~~~~~~

# src/asgcn_recon/data/common.py

~~~~~~python
from __future__ import annotations

import math
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


def uniform_cap_factor(event_count: int, max_events: int | None) -> int:
    """Return the integer stride needed to keep at most ``max_events`` events."""
    if max_events is None or max_events <= 0 or event_count <= max_events:
        return 1
    return math.ceil(event_count / max_events)


def stratified_subsample(events: np.ndarray, max_events: int | None) -> np.ndarray:
    """Uniformly stride-sample events to bound graph memory without linspace jitter."""
    factor = uniform_cap_factor(len(events), max_events)
    if factor == 1:
        return events.astype(np.float32, copy=False)
    return events[::factor].astype(np.float32, copy=False)


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
~~~~~~

# src/asgcn_recon/data/eventaid_r.py

~~~~~~python
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
    uniform_cap_factor,
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
        dataset_sampling_factor = uniform_cap_factor(cropped_event_count, self.max_events)
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
                "dataset_sampling_factor": dataset_sampling_factor,
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
~~~~~~

# src/asgcn_recon/data/eventhdr.py

~~~~~~python
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
    uniform_cap_factor,
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
                    if frame_index % self.frame_stride == 0 and end_idx > selected_start_idx:
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
        dataset_sampling_factor = uniform_cap_factor(cropped_event_count, self.max_events)
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
                "dataset_sampling_factor": dataset_sampling_factor,
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
~~~~~~

# src/asgcn_recon/data/factory.py

~~~~~~python
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
        manifest["file_to_scene"] = {
            key: key for key in (*manifest["train_files"], *manifest["val_files"])
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
        if split_manifest and split in {"train", "val", "calibration"}:
            manifest_path = Path(split_manifest)
            manifest = load_eventhdr_split_manifest(manifest_path)
            if manifest["status"] == "final":
                if training_root.resolve() == validation_root.resolve():
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
            cfg["allowed_files"] = manifest[key]
            cfg["file_to_scene"] = {
                file_key: manifest["file_to_scene"][file_key] for file_key in manifest[key]
            }
        dataset = EventHDRDataset(root=root, **cfg)
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
~~~~~~

# src/asgcn_recon/engine.py

~~~~~~python
from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import statistics
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
    return {
        "version": 1,
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
        "validate_every": max(1, int(train_config.get("validate_every", 1))),
        "recurrent_state_detached_each_sample": True,
        "runtime": {
            "device_type": device.type,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "cudnn": (torch.backends.cudnn.version() if device.type == "cuda" else None),
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
        "version": 4,
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
        "selection_metric": "macro_ssim",
        "ssim": "gaussian_valid_11_sigma1.5",
    }


def _enforce_training_split_status(config: dict[str, Any]) -> None:
    manifest_path = config.get("dataset", {}).get("split_manifest")
    if not manifest_path:
        return
    manifest = load_eventhdr_split_manifest(manifest_path)
    status = str(manifest.get("status", "missing")).strip().lower()
    allow_provisional = bool(config.get("train", {}).get("allow_provisional_split", False))
    if status != "final" and not allow_provisional:
        raise ValueError(
            f"Training split manifest {manifest_path} has status='{status}', not 'final'. "
            "Finalize the scene-level split or explicitly set "
            "train.allow_provisional_split=true for a non-reportable smoke run."
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
    validate_every = max(1, int(train_config.get("validate_every", 1)))
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

        should_validate = epoch % validate_every == 0 or epoch == epochs
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
            "dataset_sampling_factor": diagnostics["dataset_sampling_factor"],
            "model_sampling_factor": diagnostics["event_sampling_factor"],
            "effective_sampling_factor": diagnostics["effective_sampling_factor"],
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
    samples: int = 100,
) -> Path:
    if int(samples) < 1:
        raise ValueError("calibration samples must be at least 1")
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
    calibration_indices = _balanced_sample_indices(
        dataset,
        min(int(samples), len(dataset)),
        seed=int(config.get("seed", 2026)),
    )
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
    output_path = Path(output_path)
    atomic_torch_save(inference_checkpoint, output_path)
    return output_path
~~~~~~

# src/asgcn_recon/graph.py

~~~~~~python
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
    """Build the paper's simple undirected distance graph without self edges.

    Every ordered edge direction is materialized so source-to-target aggregation is
    equivalent to an undirected graph. Pairwise distances are evaluated in chunks to
    bound temporary memory without changing the radius-graph result. The scalar edge
    pseudo-coordinate is distance / D in [0,1], as required by a B-spline kernel.
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
    sources: list[torch.Tensor] = []
    destination_chunks: list[torch.Tensor] = []
    distances_kept: list[torch.Tensor] = []
    retained_edge_count = 0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        distances = torch.cdist(coordinates[start:stop], coordinates)
        within_radius = distances < radius
        local_rows = torch.arange(stop - start, device=device)
        within_radius[local_rows, local_rows + start] = False
        chunk_edge_count = int(within_radius.sum().item()) if max_edges is not None else None
        if (
            max_edges is not None
            and chunk_edge_count is not None
            and retained_edge_count + chunk_edge_count > max_edges
        ):
            raise RuntimeError(
                "Radius graph exceeded max_graph_edges="
                f"{max_edges:,} while processing {count:,} nodes. Reduce "
                "graph_radius/max_events or raise the explicit memory guard after "
                "measuring accelerator memory."
            )
        local_sources, local_destinations = torch.nonzero(
            within_radius,
            as_tuple=True,
        )
        if local_sources.numel() == 0:
            continue
        sources_global = local_sources + start
        retained_edge_count += int(local_sources.numel())
        sources.append(sources_global)
        distances_kept.append(distances[local_sources, local_destinations])
        destination_chunks.append(local_destinations)

    if not sources:
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=positions.dtype),
        )
    source = torch.cat(sources)
    destination = torch.cat(destination_chunks)
    distance = torch.cat(distances_kept)
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
            for kernel_index in range(self.kernel_size):
                coefficient = torch.where(
                    indices[:, 0] == kernel_index,
                    basis[:, 0],
                    torch.zeros_like(basis[:, 0]),
                )
                coefficient = coefficient + torch.where(
                    indices[:, 1] == kernel_index,
                    basis[:, 1],
                    torch.zeros_like(basis[:, 1]),
                )
                # Project each node once per control point instead of materializing
                # an [E,Cin,Cout] kernel or repeating an edge-wise matrix product.
                projected = x @ self.weight[kernel_index]
                messages = projected[source] * coefficient[:, None].to(projected.dtype)
                messages = messages.to(output.dtype)
                output.index_add_(0, destination, messages)
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
~~~~~~

# src/asgcn_recon/losses.py

~~~~~~python
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
~~~~~~

# src/asgcn_recon/metrics.py

~~~~~~python
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
~~~~~~

# src/asgcn_recon/model.py

~~~~~~python
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
        dataset_sampling_factor = int(sample.get("metadata", {}).get("dataset_sampling_factor", 1))
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
            "dataset_sampling_factor": dataset_sampling_factor,
            "effective_sampling_factor": (dataset_sampling_factor * self.event_sampling_factor),
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
~~~~~~

# src/asgcn_recon/utils.py

~~~~~~python
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
~~~~~~

# tests/__init__.py

~~~~~~python
"""Test-only package; never installed with asgcn_recon."""
~~~~~~

# tests/fixtures.py

~~~~~~python
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
~~~~~~

# tests/test_asgcn_paper_core.py

~~~~~~python
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
        "metadata": {"dataset_sampling_factor": 1},
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
~~~~~~

# tests/test_data_validation.py

~~~~~~python
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
~~~~~~

# tests/test_e2e.py

~~~~~~python
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
    assert timing["mean_retained_events"] == 27
    assert timing["retention_ratio"] == 27 / 80
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
~~~~~~

# tests/test_engine_integrity.py

~~~~~~python
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
            "--require-eventhdr-smoke",
            "smoke manifest files",
            ("at least 51", "eventhdr_eval_h5", "eventaid_r_zip"),
        ),
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


def test_check_env_smoke_accepts_only_the_four_manifest_h5_files(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    train_root = data_root / "EventHDR" / "train"
    train_root.mkdir(parents=True)
    for name in ("1.h5", "2.h5", "48.h5", "49.h5"):
        (train_root / name).touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-eventhdr-smoke",
        ],
    )
    check_env.main()
~~~~~~

# tests/test_graph_vectorized.py

~~~~~~python
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
~~~~~~

# tests/test_inspect_all.py

~~~~~~python
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
~~~~~~

# tests/test_metrics_ssim.py

~~~~~~python
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
~~~~~~

# tests/test_p0_engine.py

~~~~~~python
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


def test_training_rejects_nonfinal_split_without_explicit_override(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    split = {
        "status": "provisional",
        "train_files": ["train.h5"],
        "val_files": ["val.h5"],
    }
    manifest.write_text(json.dumps(split), encoding="utf-8")
    config = {
        "dataset": {"split_manifest": str(manifest)},
        "train": {"allow_provisional_split": False},
    }
    with pytest.raises(ValueError, match="allow_provisional_split"):
        _enforce_training_split_status(config)
    config["train"]["allow_provisional_split"] = True
    _enforce_training_split_status(config)

    split["status"] = "final"
    manifest.write_text(json.dumps(split), encoding="utf-8")
    config["train"]["allow_provisional_split"] = False
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
~~~~~~

# tests/test_pipeline.py

~~~~~~python
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from asgcn_recon.cli import inspect_dataset
from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.data.common import stratified_subsample, uniform_cap_factor
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
    assert sample["metadata"]["dataset_sampling_factor"] == 3
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
    assert dataset[1]["metadata"]["dataset_sampling_factor"] == 1


def test_eventaid_next_frame_alignment(tmp_path):
    make_eventaid(tmp_path / "eventaid")
    dataset = EventAidRZipDataset(tmp_path / "eventaid", max_events=32)
    assert len(dataset) == 3
    assert dataset.samples[0]["frame_id"] == 1
    assert dataset.samples[0]["target_name"].endswith("000002_img.png")
    assert dataset[0]["metadata"]["dt_us"] == 10_000
    assert dataset[0]["metadata"]["raw_event_count"] == 80
    assert dataset[0]["metadata"]["cropped_event_count"] == 80
    assert dataset[0]["metadata"]["retained_event_count"] == 27
    assert dataset[0]["metadata"]["dataset_sampling_factor"] == 3


def test_max_events_uses_exact_integer_stride_sampling() -> None:
    events = np.arange(13 * 4, dtype=np.float32).reshape(13, 4)
    assert uniform_cap_factor(len(events), max_events=5) == 3
    retained = stratified_subsample(events, max_events=5)
    np.testing.assert_array_equal(retained, events[[0, 3, 6, 9, 12]])
    assert uniform_cap_factor(len(events), max_events=None) == 1
    np.testing.assert_array_equal(stratified_subsample(events, None), events)


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
        and sample["metadata"]["dataset_sampling_factor"] == 1
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
~~~~~~

# tests/test_temporal_metric.py

~~~~~~python
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
~~~~~~
