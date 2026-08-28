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
bash scripts/setup_server.sh
source .venv/bin/activate
```

CPU 전용 설치나 PyPI 기본 wheel을 사용할 때는 `TORCH_INDEX_URL`을 비워 둔다. GPU가 반드시
보여야 하는 compute node에서 설치를 검사하려면 다음과 같이 실행한다.

```bash
python scripts/check_environment.py --require-cuda
python -m pytest -q
python -m asgcn_recon.smoke --workspace /tmp/asgcn-smoke
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

EventAid-R는 ZIP을 풀지 않고 직접 읽는다. 로더만 확인할 때는 작은 장면 하나만 받는다.

```bash
bash scripts/download_eventaid_r.sh R-bear
python -m asgcn_recon.cli inspect --config configs/eventaid_r_eval.json --samples 2
```

전체 14개 장면이 필요할 때만 `bash scripts/download_eventaid_r.sh --all`을 사용한다.

## 4. 단일 GPU 서버

SSH 연결이 끊겨도 학습이 유지되도록 `tmux` 안에서 실행한다.

```bash
tmux new -s asgcn
bash scripts/run_train.sh configs/eventhdr_train.json
```

분리: `Ctrl-b`, `d`
재접속: `tmux attach -t asgcn`

중단된 학습은 optimizer, AMP scaler, RNG와 epoch가 들어 있는 `last.pt`에서 재개한다.

```bash
python -m asgcn_recon.cli train \
  --config configs/eventhdr_train.json \
  --resume runs/eventhdr_asgcn/last.pt
```

평가:

```bash
bash scripts/run_eval.sh \
  configs/eventhdr_eval.json \
  runs/eventhdr_asgcn/best.pt
```

EventAid-R 외부평가는 첫 번째 인자만 `configs/eventaid_r_eval.json`으로 바꾼다.
SNN 평가는 결과 덮어쓰기를 막기 위해 각각 `configs/eventhdr_snn_eval.json`,
`configs/eventaid_r_snn_eval.json`을 사용한다.

## 5. SLURM 클러스터

기본 예제는 GPU 1개, CPU 8개, 메모리 32GB를 요청한다. 클러스터 정책에 맞게 `#SBATCH`
값을 수정한다.

```bash
sbatch server/slurm_train.sbatch

sbatch --export=ALL,CONFIG_PATH=configs/eventaid_r_eval.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best.pt server/slurm_eval.sbatch
```

로그는 기본적으로 `slurm-<job-name>-<job-id>.out/.err`에 기록되고 Git에서는 제외된다.

## 6. Docker

서버에 Docker와 NVIDIA Container Toolkit이 구성된 경우:

```bash
docker build \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t asgcn-event-reconstruction .

docker compose run --rm experiment \
  inspect --config configs/eventhdr_train.json
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
