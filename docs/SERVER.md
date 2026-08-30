# MobaXterm·Linux GPU 서버 실행 가이드

MobaXterm은 Windows PC에서 Linux 서버에 접속하는 SSH terminal/SFTP client다. 학습과 평가는
MobaXterm 자체가 아니라 접속한 GPU server 또는 scheduler compute node에서 실행한다. 아래 명령은
저장소 root 기준이며, 전체 EventHDR와 EventAid-R를 사용하는 본실험 경로만 설명한다.

## 1. private repository 인증, clone과 환경 설치

> 저장소는 private로 유지한다. 본실험용 clone/pull 전에는 sanitized history와 과거 CI run/artifact
> 정리 기록, 원격 `main`의 배포 commit SHA, 같은 SHA의 **로컬 실제-marker release gate**와
> **GitHub Actions 필수 gate** 통과를 확인한다. CI는 실제 marker를 받지 않아 로컬 검사를 대신하지 않는다.
> 문서나 최신 CI badge만으로 배포 성공을 판단하지 않으며, 확인 전에는 본실험을 시작하지 않는다.

MobaXterm에서 SSH session을 열고 서버 terminal에서 실행한다. 서버 로그인용 SSH 인증과 GitHub
private repository 인증은 서로 별개다. 공유 서버에서는 repository 범위의 읽기 전용 Deploy key를
사용한다. 전용 키가 없다면 생성하고 공개키만 출력한다.

```bash
prepare_asgcn_deploy_key() {
  local key="$HOME/.ssh/asgcn_unet_deploy"
  local derived_pub stored_pub

  mkdir -p "$HOME/.ssh" || return 1
  chmod 700 "$HOME/.ssh" || return 1
  if [[ -e "$key" || -e "$key.pub" ]]; then
    if [[ ! -f "$key" || ! -f "$key.pub" ]]; then
      echo "ERROR: incomplete deploy key pair; inspect it manually" >&2
      return 1
    fi
    derived_pub="$(ssh-keygen -y -f "$key")" || return 1
    stored_pub="$(awk 'NF >= 2 { print $1 " " $2; exit }' "$key.pub")"
    if [[ -z "$stored_pub" || "$derived_pub" != "$stored_pub" ]]; then
      echo "ERROR: deploy public key does not match the private key" >&2
      return 1
    fi
    echo "Using the verified existing deploy key pair."
  else
    ssh-keygen -t ed25519 \
      -C "asgcn-unet read-only deploy key" \
      -f "$key" || return 1
  fi

  chmod 600 "$key" || return 1
  chmod 644 "$key.pub" || return 1
  cat "$key.pub"
}

prepare_asgcn_deploy_key
unset -f prepare_asgcn_deploy_key
```

공개키를 [repository Deploy keys](https://github.com/costunder/asgcn-unet/settings/keys)에 추가하되
`Allow write access`는 체크하지 않는다. 같은 경로의 개인키가 이미 있다면 `ssh-keygen`으로 덮어쓰지
않는다. 등록 후 기본 clone 명령부터 실행하지 말고 해당 키로 인증되는지 먼저 확인한다. 최초 연결
prompt에서는 `yes`를 입력하기 전에 algorithm과 fingerprint를
[GitHub 공식 목록](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)과
대조한다.

`ssh -T`는 성공해도 종료 코드 1이 정상이므로 `set -e` one-shot block과 분리해 사람이 메시지만
확인한다.

```bash
ssh -o "IdentityFile=$HOME/.ssh/asgcn_unet_deploy" \
  -o IdentitiesOnly=yes \
  -T git@github.com
```

최종 자동 판정은 exact-repository `ls-remote`다.

```bash
git -c core.sshCommand="ssh -o 'IdentityFile=$HOME/.ssh/asgcn_unet_deploy' -o IdentitiesOnly=yes" \
  ls-remote git@github.com:costunder/asgcn-unet.git HEAD
```

`ls-remote`가 `HEAD`를 출력한 뒤 사용자가 선택한 directory에서 clone한다. 아래 기본값은 현재
directory 아래 `asgcn-unet`이고, `ASGCN_DIR`로 바꿀 수 있다. 이후 pull도 같은 키를 사용하도록
repository-local 설정을 저장한다.

```bash
ASGCN_DIR="${ASGCN_DIR:-$PWD/asgcn-unet}"
(
  set -e
  mkdir -p "$ASGCN_DIR"
  cd "$ASGCN_DIR"
  if [[ -n "$(find . -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: clone target is not empty" >&2
    exit 1
  fi
  git -c core.sshCommand="ssh -o 'IdentityFile=$HOME/.ssh/asgcn_unet_deploy' -o IdentitiesOnly=yes" \
    clone git@github.com:costunder/asgcn-unet.git .
  git config core.sshCommand \
    "ssh -o 'IdentityFile=$HOME/.ssh/asgcn_unet_deploy' -o IdentitiesOnly=yes"
)
```

clone block이 성공하면 `git -C "$ASGCN_DIR" rev-parse HEAD`를 CI에서 확인한 배포 SHA와 대조한다.
같은 SHA일 때만 다음 설치·검증 block을 실행한다. 이후 pull 뒤에도 다시 대조한다. 이 block도
subshell 안에서 fail-fast로 동작하므로 실패해도 MobaXterm 로그인 shell 자체를 종료하지 않는다.

```bash
(
  set -e
  cd "$ASGCN_DIR"
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
)
```

검증한 fingerprint에 동의하는 것은 `known_hosts` 등록일 뿐 GitHub 사용자 인증 성공을 의미하지 않는다.
비표준 이름의 전용 키에는 위 shell-safe `IdentityFile` option이 필요하다. `ssh -T`는 성공 메시지 뒤에도 종료 코드 1을
반환할 수 있으므로 `ls-remote`를 최종 판정으로 사용한다. clone 명령의 마지막 `.`은 별도의 하위
directory를 만들지 않고 현재 directory를 checkout root로 사용한다. 기존 파일이나 실패한 clone의
숨김 파일이 있으면 `ls -la`로 확인하고 덮어쓰거나 임의 삭제하지 않는다. 약 50.4GB의 최종 dataset
외에도 EventHDR upload archive, 가상환경과 결과 공간이 필요하므로 `df -h .`로 quota를 먼저 확인한다.
더 이상 server-side pull이 필요 없으면 GitHub에서 Deploy key를 revoke하고 서버 정책에 따라 전용
key pair를 폐기한다. Deploy key는 자동 만료되지 않는다.

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
allocation 안에서 수행한다. cluster가 module을 쓴다면 Python/CUDA module을 먼저 load한다. Slurm과
PBS wrapper들은 필요할 때 `CUDA_MODULE=cuda/<version>`도 받는다.

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
temporary file과 atomic replace를 사용한다. checked-in config와 `scripts/run.sh`는 논리 경로
`data/EventHDR`를 고정해 검증하므로 shared storage는 위 `--link` 방식으로 연결한다. 임의 destination과
별도 config는 새 실험으로 관리하고 기본 full runner와 섞지 않는다.

### EventAid-R 14개 ZIP

```bash
bash scripts/get_aid.sh --all
```

manifest의 14개 scene, 표시 합계 약 24.68 GB를 내려받는다. 각 파일이 ZIP container인지 검사하며
loader가 archive member를 직접 읽으므로 압축을 풀지 않는다. shared storage를 쓰면 repository의
`data/EventAid-R`가 검증된 shared directory를 가리키도록 symlink한다. 기본 runner는 이 논리 경로를
검사하므로 `EVENTAID_ROOT`만 다른 곳으로 바꾸면 안 된다.

### full-data와 decode 검사

GPU allocation 안에서 다음 검사를 한 번 통과시킨다.

```bash
source .venv/bin/activate
python scripts/check_env.py \
  --require-cuda --require-full-data --lock constraints/py312.txt

asgcn-unet inspect --config configs/train.json --samples 2 --validate-all
asgcn-unet inspect --config configs/aid.json --samples 2 --validate-all
```

`train.json` inspect는 manifest에 따라 EventHDR train 51개와 eval 19개 root를 모두 검사한다.
EventAid 명령은 manifest의 14개 ZIP에 있는 모든 선택 event block과 target을 decode한다.
`--validate-all`은 metadata만 세는 명령이 아니므로 dataset 크기에 따라 오래 걸린다. 실패한 file을
제외해 진행하지 말고 원본을 다시 전송·검증한다. 출력의 `event_timestamp_diagnostics`에는 각 block의
원 timestamp min/max, interval span ratio·offset·범위 이탈 합계가 포함된다. 공식 14 ZIP에서 공통
timestamp basis와 단위가 확인되기 전까지 이 값은 진단용이며 자동 rejection 조건은 아니다.

## 3. 직접 서버에서 전체 실행

실행 순서를 먼저 확인하려면 data/GPU 작업을 실제 수행하지 않는 schedule 출력을 본다.

```bash
DRY_RUN=1 bash scripts/run.sh all
```

GPU shell 또는 allocation 안에서 전체 protocol을 시작한다.

```bash
source .venv/bin/activate
mkdir -p logs
bash scripts/run.sh all 2>&1 | tee logs/run.log
```

SSH 연결 종료에 대비하려면 tmux를 사용한다.

```bash
mkdir -p logs
tmux new-session -s asgcn -c "$PWD" \
  "bash -lc 'source .venv/bin/activate && bash scripts/run.sh all 2>&1 | tee logs/run.log'"
```

`run.sh all`은 다음을 순서대로 수행한다.

1. `check_env.py --require-full-data --lock constraints/py312.txt`와 기본 CUDA 검사
2. EventHDR train/eval과 EventAid-R 전체 `inspect --validate-all`
3. EventHDR train 전체 graph topology scan과 edge 수 상위 표본 CUDA forward/backward profile
4. profile을 현재 config/data/source/runtime에 재검증한 뒤 ANN 40-epoch 학습 또는 resume
5. EventHDR train 모든 frame을 이용한 `best.pt`→`best_snn.pt` 보정
6. EventHDR/EventAid-R ANN과 `literal_eq15`/`standard_if` × `T=4,8,16,32` evaluate+benchmark

기본 benchmark는 warmup 10, 측정 100 frame이다. 필요하면 실행 전에
`BENCHMARK_WARMUP`, `BENCHMARK_STEPS`, `SIMULATION_STEPS_LIST`를 설정한다. 본실험 기본 행렬을 바꾼
결과는 별도 protocol로 기록한다.
보고용 quality evaluation은 config의 `eval.max_samples=null`을 유지해 전체 dataset을 사용한다.
부분 quality cap은 기본적으로 거부되며 합성-test non-reporting 우회를 사용해도 결과 표에 넣을 수 없다.
benchmark step 수는 이 quality cap과 별도의 compute-only 측정 계약이다.
두 dataset config의 `expected_file_count`와 EventHDR final split/EventAid-R fixed file manifest를 제거하면
보고용 evaluate와 benchmark가 시작되지 않는다. EventHDR는 현재 content·transform·manifest도 source
ANN validation과 같아야 한다.

profile 기본값은 `PROFILE_TOP_DENSITY=10`, `PROFILE_SAMPLES=3`,
`PROFILE_OUTPUT=runs/profile.json`이다. 전수 scan은 edge guard 초과 표본을 찾고, 실제 CUDA probe는
edge 수 상위 3개 표본에서 configured loss·optimizer까지 포함한 학습 step과 peak allocated/reserved
VRAM을 잰다. 이는 기록된 GPU와 선택 표본에 한정된 실측 gate이며 절대 VRAM 보증이 아니다.

기존 training/evaluation artifact는 묵시적으로 덮어쓰지 않는다. 기존 SNN checkpoint만 의도적으로
다시 만들 때는 `OVERWRITE_CALIBRATION=1`을 사용하며 새 checkpoint가 완성된 뒤 atomic replace된다.
evaluation directory가 이미 있으면 결과를 다른 보존 위치로 옮기거나 config의 `eval.output_dir`을
바꾼 뒤 실행한다.

보고용 calibration은 `CALIBRATION_SAMPLES=all`을 사용한다. 전체 training sample보다 작은 숫자는
기본 wrapper에서 sealed checkpoint를 만들지 못하고 실패한다. 부분 보정은 직접 CLI의 명시적
비보고용 우회에서만 허용되며 해당 checkpoint는 전체 평가 표에 사용할 수 없다.
보정 checkpoint는 model tensor의 persistent `calibration_attempts`, 32-byte
`calibration_commitment_digest`, `calibration_commitment_sealed`에도 실제 시도 횟수와
protocol/count/sampling 및 valid/minimum/dead-channel summary core commitment를 저장하므로 부분 보정
tensor에 전체 metadata만 덮어씌우거나 dead-channel 수만 바꾼 파일은 load 단계에서 거부된다. 각 graph
layer는 관측 raw maximum `calibration_activation_max`, 식 (6)에 실제 사용한 `normalization_scale`,
`dead_channel_mask`를 분리 저장한다. raw=0인 channel은 scale만 1이며 save→strict reload 뒤에도 dead
summary가 유지된다.
`--allow-unsealed-calibration`을 직접 지정한 실행은 전체 sample이어도 override 자체가 기록돼
`sealed=false`이며 보고용 표에 사용할 수 없다.

## 4. 중단 후 epoch-boundary resume

직접 실행은 다음과 같다.

```bash
RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  bash scripts/run.sh train 2>&1 | tee logs/train-resume.log
```

동일한 동작을 저수준 wrapper로 실행하려면 다음 명령을 쓴다.

```bash
RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  bash scripts/train.sh configs/train.json
```

`last.pt`는 각 완료 epoch 뒤에 저장되므로 종료된 epoch 내부 step은 되풀이된다. checkpoint는 같은
configured run directory 안에 있어야 하며, source tree/Git 상태, model·optimizer·scheduler·AMP,
validation/data full SHA-256과 PyTorch/CUDA/cuDNN·GPU 이름/compute capability·visible CUDA RNG state가
일치해야 한다. `runs/profile.json`도 현재 config·전체 train data·source·동일 CUDA runtime에 다시
결합되어야 한다. 다른 GPU나 source checkout으로 옮기는 것은 일반 weight load가 아니라 exact resume
요청이므로 거부될 수 있다. 상세 계약은 [EXPERIMENT.md](EXPERIMENT.md)의 resume 절을 따른다.

상위 runner는 `check`, `profile`, `train`, `calibrate`, `eval`, `all` stage를 제공한다. 학습 재개가 끝난 뒤에는
`bash scripts/run.sh calibrate`, 그다음 `bash scripts/run.sh eval`을 실행한다. 기존 training,
profile, calibration, evaluation artifact를 자동으로 건너뛰거나 덮어쓰지 않으므로, 부분 실패 복구 시
완료된 stage를 묵시적으로 재사용하지 않는다. CUDA profile 우회는 합성 비보고용 직접 CLI에서만
명시적으로 가능하고 scheduler wrapper 및 `all`에서는 차단된다.

## 5. Slurm: profile → train → calibrate → evaluation matrix

header의 partition/account/GPU type/walltime은 cluster 정책에 맞춰 수정한다. 기본 요청은 GPU 1개,
CPU 8개, RAM 32 GB이며 profile/calibration 12시간, train 48시간, evaluation 8시간이다. 저장소 root에서
제출하면 `SLURM_SUBMIT_DIR`을 project root로 사용한다. 다른 위치에서 제출할 때는
`--export=PROJECT_ROOT=/absolute/path/to/repo`를 추가한다. `--export`에는 아래처럼 job별 필수 변수만
나열하며 login shell의 token, proxy, credential까지 전달할 수 있는 `ALL`은 사용하지 않는다.

앞 절의 full-data/decode 검사를 완료한 뒤 다음 dependency chain을 제출한다.

```bash
unset SNN_DYNAMICS

profile_id=$(sbatch --parsable \
  --export=PROJECT_ROOT="$PWD" \
  server/profile.sbatch)

train_id=$(sbatch --parsable \
  --dependency="afterok:${profile_id}" \
  --export=PROJECT_ROOT="$PWD",VALIDATE_DATASET=0 \
  server/train.sbatch)

cal_id=$(sbatch --parsable \
  --dependency="afterok:${train_id}" \
  --export=PROJECT_ROOT="$PWD",VALIDATE_DATASET=0,CALIBRATION_SAMPLES=all \
  server/calibrate.sbatch)

for cfg in configs/hdr.json configs/aid.json; do
  sbatch --dependency="afterok:${cal_id}" \
    --export="PROJECT_ROOT=$PWD,VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH=${cfg},CHECKPOINT_PATH=runs/train/best.pt,INFERENCE_MODE=ann" \
    server/eval.sbatch
done

for cfg in configs/hdr.json configs/aid.json; do
  for dynamics in literal_eq15 standard_if; do
    for steps in 4 8 16 32; do
      sbatch --dependency="afterok:${cal_id}" \
        --export="PROJECT_ROOT=$PWD,VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH=${cfg},CHECKPOINT_PATH=runs/train/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS=${dynamics},SIMULATION_STEPS=${steps}" \
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
  --export=PROJECT_ROOT="$PWD",VALIDATE_DATASET=0,RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  server/train.sbatch)
```

wrapper log의 공개 기본값은 host/job 식별자를 생략하고 config/checkpoint를 basename으로 표시한다. 정확한
hostname, scheduler job ID, project/config/checkpoint 경로는 비공개 로컬 진단에서만
`INCLUDE_PRIVATE_HOST_PROVENANCE=1`을 해당 `--export` 목록에 추가해 확인한다. 이 opt-in log는 공개하거나
첨부하지 않는다.

## 6. PBS/Torque: profile → train → calibrate → evaluation matrix

`select`, `ngpus`, queue/project resource 이름은 site마다 다르므로 `server/*.pbs` header를 제출 전에
확인한다. 저장소 root에서 제출하면 `PBS_O_WORKDIR`을 project root로 사용한다. 외부에서 제출할 때는
`-v PROJECT_ROOT=/absolute/path/to/repo`를 추가한다.

```bash
unset SNN_DYNAMICS

profile_id=$(qsub server/profile.pbs)

train_id=$(qsub \
  -W depend="afterok:${profile_id}" \
  -v VALIDATE_DATASET=0 \
  server/train.pbs)

cal_id=$(qsub \
  -W depend="afterok:${train_id}" \
  -v VALIDATE_DATASET=0,CALIBRATION_SAMPLES=all \
  server/calibrate.pbs)

for cfg in configs/hdr.json configs/aid.json; do
  qsub -W depend="afterok:${cal_id}" \
    -v VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH="${cfg}",CHECKPOINT_PATH=runs/train/best.pt,INFERENCE_MODE=ann \
    server/eval.pbs
done

for cfg in configs/hdr.json configs/aid.json; do
  for dynamics in literal_eq15 standard_if; do
    for steps in 4 8 16 32; do
      qsub -W depend="afterok:${cal_id}" \
        -v VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH="${cfg}",CHECKPOINT_PATH=runs/train/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS="${dynamics}",SIMULATION_STEPS="${steps}" \
        server/eval.pbs
    done
  done
done
```

PBS에서 resume chain을 시작할 때는 다음처럼 `RESUME_CHECKPOINT`를 넘긴다.

```bash
train_id=$(qsub \
  -v VALIDATE_DATASET=0,RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  server/train.pbs)
```

PBS는 login environment 전체를 전달하는 `#PBS -V`를 사용하지 않는다. site CUDA module이 필요하면
`CUDA_MODULE`을 넘긴다. 별도 venv나 checkout을 쓸 때는 `PYTHON_BIN`, `PROJECT_ROOT`를 `-v`로
명시한다.

### scheduler log 공개 절차

wrapper가 log **내용**에서 host/job 식별자를 생략해도 scheduler가 만든 raw 파일명 자체에 job ID가
있으므로 공개 준비가 끝난 것이 아니다. Slurm의 `slurm-<job-name>-<job-id>.out/.err`와 PBS의
`<job-name>.o<job-id>`(site에 따라 server suffix 포함)는 job ID를 드러낸다. raw scheduler log는
접근이 제한된 실험 기록으로 보존하고, 공개할 때는 필요한 파일만 다음처럼 중립 파일명의 공개 후보
사본으로 만든다. raw 파일을 그대로 첨부하거나 raw 이름을 archive member 이름으로 보존하지 않는다.

```bash
raw_log=/path/to/restricted/raw-scheduler-log-with-id
public_log=logs/public/train.stdout.log
mkdir -p "$(dirname -- "$public_log")"
cp -- "$raw_log" "$public_log"

python scripts/scan_private_text.py "$public_log" \
  --root "$PWD" --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

stdout/stderr가 분리됐으면 둘 다 중립명으로 복사하고 각각 검사한다. 위 검사는 내용을 자동 삭제하는 도구가
아니므로 통과한 중립 사본만 공유한다. 실패한 사본은 공개하지 않고 원인을 제거한 뒤 다시 만든다.
`INCLUDE_PRIVATE_HOST_PROVENANCE=1`, `--include-private-host-provenance` 또는 이에 준하는 opt-in으로
hostname, job ID, 절대경로를 기록한 log는 이름을 바꾸거나 scan을 통과했더라도 비공개다. 공개용 log가
필요하면 private provenance opt-in 없이 job/진단을 다시 실행한다.

## 7. 산출물 확인과 운영상 오류

학습이 끝나면 다음 파일을 먼저 확인한다.

```bash
ls -lh runs/profile.json
ls -lh runs/train/{last.pt,best.pt,best_snn.pt,history.json,config.json}
find runs -name metrics.json -o -name benchmark.json | sort
```

평가 artifact는 다음 위치에 있다.

```text
runs/eval/hdr/ann/
runs/eval/hdr/snn_<dynamics>_T<steps>/
runs/eval/aid/ann/
runs/eval/aid/snn_<dynamics>_T<steps>/
```

`runs/profile.json`, `runs/status/{check,profile,train,calibrate,eval}.json`과 각 run의 `metrics.json`,
`frames.csv`, `predictions/`, `benchmark.json`을 config, Git commit, scheduler log, `check_env.py` 출력과
함께 보존한다. profile과 평가 artifact의 공개 protocol은 host 절대경로/hostname을 저장하지 않는다.
`check_env.py`와 `inspect`의 `--include-private-host-provenance` 출력은 로컬 진단 전용이며 공개
artifact나 외부 첨부에 포함하지 않는다.
내부 SHA-256은 checkpoint·metadata·tensor의 재현 identity와 우발적 손상 검사용이지 전자서명이나
출처 인증이 아니다. 제출용 file hash는 접근통제된 immutable 실험 원장 또는 signed manifest에도
별도로 보존한다.

원격 history 교체 전후에는 저장소 밖 로컬 denylist를 사용해 complete non-shallow clone의 current
tree와 모든 local-ref reachable blob을 로컬에서 함께 검사한다. 실제 marker는 로컬 환경변수
`PRIVATE_MARKERS_B64`로도 주입할 수 있지만 GitHub secret·변수·workflow·log·artifact로 전송하지
않는다. 로컬 release gate의 `--require-external-patterns`는 빈 denylist를 거부한다.

```bash
python scripts/scan_private_text.py \
  --all-tracked --all-history --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

코드·문서·검증 수치 수정과 history 정리를 끝내 최종 sanitized source commit을 확정한 뒤 clean
source에서 summary를 재생성하고 summary-only commit을 만든다. 생성 뒤 다른 tracked 파일 수정이나
source history rewrite가 필요하면 source commit 확정부터 반복한다. 같은 최종 SHA에서 위 로컬
실제-marker 검사와 다음 clean provenance gate를 통과해야 한다. CI는 실제 marker 없이 generic
current-tree/history 검사와 clean provenance를 수행한다. 원격 배포 후 같은 최종 SHA의 GitHub
Actions 필수 job 성공도 확인하되 로컬 실제-marker 검사와 혼동하지 않는다. 배포/CI 결과를 본문에
추가하기 위한 수정도 source 변경이므로, 결과는 해당 SHA의 Actions와 별도 로컬 배포 기록으로 남긴다.

```bash
python scripts/build_code_summary.py --check --require-clean-provenance
```

자주 중단되는 조건은 다음과 같다.

- `CUDA available: false`: login node가 아닌 GPU allocation인지, CUDA wheel과 driver가 맞는지 확인한다.
- `EventHDR ... exact official file set`: OneDrive 전송이 끝났는지, train 51/eval 19 외 H5가 섞이지
  않았는지 `get_hdr.sh --check`로 확인한다.
- `eventaid_r_zip must contain exactly 14`: `get_aid.sh --all`을 완료하고 ZIP을 압축 해제하지 않는다.
- `Fresh training run_dir is not empty`: 새 run이면 별도 `output.run_dir`, 중단 run이면 `last.pt` resume를
  사용한다.
- `passed CUDA preflight report not found` 또는 preflight mismatch: 같은 GPU allocation과 source/data로
  `bash scripts/run.sh profile`을 먼저 완료한다. 기존 report는 덮어쓰지 않으므로 실패 artifact를
  조사·보존한 뒤 새 output 경로로 다시 측정한다.
- resume protocol mismatch: source/GPU/software/data를 원래 run과 일치시킨다. 단순 checkpoint weight
  이식과 exact training resume를 혼동하지 않는다.
- calibrated checkpoint 오류: ANN `best.pt`를 `scripts/calibrate.sh`로 변환한 뒤 SNN 평가에
  `best_snn.pt`를 사용한다. partial calibration, manifest/transform/runtime/sample identity 누락 또는
  ANN training data/source 불일치는 보고용 seal에서 거부된다.
- evaluation output exists: 기존 artifact를 보존 위치로 옮기거나 별도 output directory를 쓴다.
- `max_graph_edges=2,000,000` 또는 OOM: edge를 조용히 잘라 진행하지 않는다. 별도 config에서
  `max_events`, `graph_radius`, model width를 변경하고 peak memory를 다시 측정해 다른 실험으로 기록한다.
- SSH 종료: foreground shell 대신 tmux, Slurm 또는 PBS job을 사용한다.
