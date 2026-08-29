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
