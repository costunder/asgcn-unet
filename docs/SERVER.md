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
