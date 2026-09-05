# Linux GPU 서버 실행 가이드

MobaXterm 등의 SSH client로 접속한 Linux GPU 서버 또는 scheduler compute node에서 실행한다.
아래 명령은 저장소 root 기준이며, 전체 EventHDR와 EventAid-R를 사용하는 실험 경로를 설명한다.

기본 실험은 서버 B4/B8/B16 비교에서 선택된 **`EXPERIMENT=fast` (B16+Triton)** 이다.
run/train/eval/calibrate와 Slurm/PBS entrypoint는 모두 이 기본값을 공유한다. 모델·데이터·40 epoch와
T4/8/16/32 평가 행렬은 그대로 유지한다. 아래 명령과 결과 경로도 fast 기준이다.
`EXPERIMENT=single` (B1+Torch, `runs/train`)과 `EXPERIMENT=batch` (B4+Torch, `runs/batch`)는
기존 비교 실험 재현용으로 명시적으로 선택할 수 있다. 자세한 batch 측정은 [TRAIN.md](TRAIN.md)를 따른다.
실행 중인 학습 checkout을 갱신하거나 같은 GPU에서 학습을 겹쳐 실행하지 않는다.

셸 entrypoint는 `bash scripts/run.sh ...` 또는 scheduler로 실행한다. 실수로 source하면 안내만 출력하고
실행을 시작하지 않는다. 호출 셸의 옵션·작업 디렉터리·trap은 유지하며, 정상 실행 중 오류는 해당 자식
작업의 실패 상태로 반환한다. `scripts/runtime.sh`만 source용 함수 라이브러리다.
GPU 번호는 코드에서 선택하지 않는다. scheduler/관리자가 제공한 allocation과 `CUDA_VISIBLE_DEVICES`를
그대로 사용하며, 장치 번호는 재할당마다 달라질 수 있다. 출력의 process-local device는 할당된 visible
장치 안의 인덱스다.
CUDA 실행 전에는 유효한 명시 mask 또는 device-cgroup whitelist로 할당 근거를 확인한다.
mask 없이 MIG 장치 한 개가 보인다는 사실, job ID 또는 NVIDIA_VISIBLE_DEVICES=all만으로는
충분하지 않다. 근거가 없으면 GPU 초기화 전에 중단하고 현재 할당 정보를 확인한다.

## 1. 환경 설치

최초 설치는 [README의 설치 및 실행](../README.md#설치-및-실행)을 따른다.
이 문서는 환경 전환, 데이터 배치, scheduler 제출과 실행 복구를 다룬다.

- Conda `asgcn` 환경 생성은 최초 한 번만 한다. 재접속 시에는 다시 생성하지 않고
  `conda activate asgcn`만 실행한다.
- 코드를 둘 현재 폴더에서 README의 HTTPS clone → Conda 환경 → 설치 → 전체 데이터 → 본실험 순서를 따른다.
  기존 환경이나 `asgcn-unet` 폴더가 있으면 삭제하거나 생성·clone을 반복하지 않는다.

설치·다운로드·학습·보정·평가는 활성화한 non-base Conda 환경 하나를 사용한다. Git은 서버에 이미
설치되어 있어야 한다. 설치기는 `CONDA_PREFIX`의 Python에 직접 설치하며 별도 환경을 만들지 않는다.
`constraints/server.json`은 Python **3.12.14**, PyTorch **2.13.0+cu126**, CUDA runtime **12.6**을
지정한다. `constraints/server.txt`는 pip·setuptools·wheel 및 CUDA library·Triton을 포함한 전이
의존성의 버전과 배포 파일 SHA-256을 고정한다. 설치기는 `--require-hashes --only-binary=:all:`로
이 lock을 설치하며 최신 bootstrap package를 임의로 받지 않는다. `constraints/py312.txt`의 core/dev
버전도 함께 검사한다. 기본 PyTorch wheel은
[공식 CUDA 12.6 index](https://download.pytorch.org/whl/cu126/torch/)에서 받는다.
서버 profile의 Linux wheel은 glibc 2.28 이상을 요구한다. 설치기는 활성화한 Conda·Python·wheel과
package 호환성을 검사한다. `.env`는 만들 필요가 없고 기존 파일도 설치기가 읽거나 변경하지 않는다.
예전 `.env`의 Python·torch·index 설정으로 새 고정 profile을 바꾸지 않는다.

GPU가 안 보이는 login node에서도 설치할 수 있다. 실제 CUDA 검증은 GPU allocation 안에서 수행한다.
설치와 동시에 GPU 검증도 요구하려면 `REQUIRE_CUDA=1 bash scripts/setup.sh`를 사용한다.
진단이 필요할 때만 `nvidia-smi`, `df -h .`로 driver와 공간을 확인한다. site CUDA module이 필요한
경우 scheduler의 `CUDA_MODULE=cuda/<version>`을 지정한다. driver 호환성은 wheel과 별도로 필요하며,
설치 성공이나 `nvidia-smi` 출력만으로 GPU 학습 성공을 보장하지 않는다.

고정 profile이 실제 GPU에서 동작하지 않으면 중단하고 관리자에게 GPU 할당·driver/container를 확인한다.
`TORCH_VERSION`·`TORCH_INDEX_URL`을 임의로 바꾸거나 CUDA 검사를 끄지 않는다. 동일 package profile은
software 버전을 맞추는 기준이지 서로 다른 GPU·driver에서 bitwise 동일한 결과를 보장하지 않는다.
이는 확인된 Python·PyTorch·CUDA 기준에 맞춰 새로 고정한 profile이며 이전 서버 package 전체의
export는 아니다. Conda의 system library, OS, driver와 GPU는 이 pip hash lock의 대상이 아니다.

### 기존 `.venv` 설치에서 전환

실행 중인 학습·보정·평가를 먼저 정상 종료하고 checkpoint와 결과를 보존한다. package를 바꾸는 동안
기존 job을 계속 실행하지 않는다. 이미 Python 3.12.14인 `asgcn` 환경이 있다면 새 환경 생성은 필요 없다.
다른 Python 버전의 기존 환경을 자동 삭제하거나 덮어쓰지 말고 먼저 별도 전환 계획을 확인한다.
기존 저장소 root에서 실행한다.

```bash
if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
conda activate asgcn
git pull --ff-only &&
bash scripts/setup.sh
```

`deactivate`는 기존 virtualenv가 활성화되어 있을 때만 실행한다. pull이 충돌하면 기존 변경을
보존하고 확인한다. 이미 받은 `data/`는 그대로 사용하며
환경 전환 때문에 데이터를 다시 다운로드하지 않는다. 기존 `.env`와 `.venv`도 자동 삭제하지 않는다.
새 Conda 환경에서 아래 GPU 검증과 기존 데이터 검사를 통과한 뒤에만, 이전 환경을 더 이상 쓰는 job이
없는지 확인하고 필요하면 기존 `.venv` 폴더만 수동 정리한다. source/runtime이 바뀐 기존 checkpoint는
exact resume 검증에서 거부될 수 있으므로 결과를 덮어쓰거나 강제로 이어 붙이지 않는다.

```bash
python scripts/check_env.py --require-cuda --lock constraints/py312.txt \
  --runtime-profile constraints/server.json
```

아래 release gate와 전체 Ruff/pytest 회귀검사는 **유지관리자 배포 절차**다. 확인된 배포를 설치하는
실험 사용자가 이를 전부 다시 실행해야 한다는 뜻은 아니다. 데이터·GPU readiness 검사는 뒤 절차를 따른다.

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

### EventHDR: 서버 직접 다운로드 (기본)

MobaXterm으로 접속한 Linux 서버의 저장소 root에서 실행한다. PC 다운로드·SFTP 전송, 사용자
로그인·브라우저·쿠키는 필요 없다. Python 표준 라이브러리 HTTP로 공식 OneDrive 공유 폴더에 익명
접근하고, H5를 ZIP으로 묶지 않고 `data/EventHDR/{train,eval}`에 직접 저장한다.

```bash
bash scripts/get_hdr.sh --download
```

기본값은 train `1.h5`–`51.h5`와 eval `1.h5`–`19.h5` 전체다. 한 split만 받을 때는 다음처럼
지정한다. 전체 학습·평가 전에 두 split 모두 있어야 한다.

```bash
bash scripts/get_hdr.sh --download --split train
bash scripts/get_hdr.sh --download --split eval
```

중단되면 같은 명령으로 다시 실행한다. `.part` 파일에서 이어받고, 일시적 HTTP 실패를 재시도하며
만료된 다운로드 링크는 새로 조회한다. 완료 파일은 정확한 이름 집합·OneDrive metadata의 byte size와
SHA-256·HDF5 signature를 검증한다. 익명 접근 token과 임시 다운로드 URL은 메모리에서만 사용하며
파일이나 로그에 기록하지 않는다.

이 구현은 2026-08-30 실접속으로 확인한 **문서화되지 않은 OneDrive 익명 호환 endpoint**를 사용한다.
Microsoft가 보장하는 안정적 API 계약은 아니므로 서비스 변경·공유 해제·서버 네트워크 차단 시
실패할 수 있다. 해당 날짜의 실접속 검증은 train 51/eval 19개 metadata 조회와 각 split H5에 대한
`Range: bytes=0-7`의 HTTP 206·HDF5 signature, 새 익명 token의 공유 접근 갱신 확인이다.
전체 약 25.72GB 다운로드·전체 decode·GPU
본실험 완료를 주장하지 않는다. 다운로드 실패를 빈 dataset이나 CPU 실행으로 우회하지 않는다.

### EventHDR: 이미 받은 데이터 가져오기 (선택)

이미 서버에 ZIP, 풀린 H5 directory 또는 shared-storage 데이터가 있을 때만 아래 import 경로를
사용한다. 직접 다운로드의 선행 조건이 아니다. train/eval 별도 archive는 각각 split을 지정한다.

```bash
bash scripts/get_hdr.sh --archive data/_archives/train.zip --split train
bash scripts/get_hdr.sh --archive data/_archives/eval.zip --split eval
```

train/eval을 함께 담은 한 archive이면 `--split` 없이 한 번 실행한다.

```bash
bash scripts/get_hdr.sh --archive data/_archives/EventHDR.zip
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

GT 확장자는 장면에 따라 PNG 또는 JPG다(JPEG 확장자도 지원). `R-traffic`은
`event_upload/`, `gt_upload/`, `timestamps_upload.txt`, `parts.txt`를 사용하는 분할 배포다.
로더가 두 구조를 직접 처리하므로 압축 해제·확장자 변경·재압축은 하지 않는다.

이전 코드에서 `R-ball.zip: event and GT files are required`가 발생했다면 PNG만 인식하던
로더 문제다. 이미 설치한 Conda 환경과 기존 저장소에서 아래처럼 갱신한다. 데이터나 환경을 다시
다운로드·생성하지 않는다. 이 복구 명령은 학습 전 데이터 검사에서 중단된 경우에 해당한다.

```bash
git pull --ff-only &&
bash scripts/run.sh all
```

수정 로더는 발견한 event/GT 수와 지원 파일명을 오류에 표시한다. 최신 코드에서도 누락 오류가 나면
해당 ZIP 내부 구조를 확인하며, 파일을 제외하거나 전체 데이터 검사를 끄지 않는다.

### full-data와 decode 검사

GPU allocation 안에서 다음 검사를 한 번 통과시킨다.

```bash
conda activate asgcn
python scripts/check_env.py \
  --require-cuda --require-full-data --lock constraints/py312.txt \
  --runtime-profile constraints/server.json

asgcn-unet inspect --config configs/fast.json --samples 2 --validate-all
asgcn-unet inspect --config configs/aid-fast.json --samples 2 --validate-all
```

`train.json` inspect는 manifest에 따라 EventHDR train 51개와 eval 19개 root를 모두 검사한다.
EventAid 명령은 manifest의 14개 ZIP에 있는 모든 선택 event block과 target을 decode한다.
`--validate-all`은 metadata만 세는 명령이 아니므로 dataset 크기에 따라 오래 걸린다. 실패한 file을
제외해 진행하지 말고 원본 다운로드 또는 import 상태를 확인·검증한다. 출력의
`event_timestamp_diagnostics`에는 각 block의 원 timestamp min/max, interval span ratio·offset·범위
이탈 합계가 포함된다. 공식 14 ZIP에서 공통
timestamp basis와 단위가 확인되기 전까지 이 값은 진단용이며 자동 rejection 조건은 아니다.

## 3. 직접 서버에서 전체 실행

실행 순서를 먼저 확인하려면 data/GPU 작업을 실제 수행하지 않는 schedule 출력을 본다.

```bash
DRY_RUN=1 bash scripts/run.sh all
```

GPU shell 또는 allocation 안에서 전체 protocol을 시작한다.

```bash
conda activate asgcn
mkdir -p logs
set -o pipefail
bash scripts/run.sh all 2>&1 | tee logs/run.log
```

SSH 연결 종료에 대비하려면 먼저 tmux 세션을 열고, 그 안에서 위의 Conda 활성화와 실행 블록을 실행한다.

```bash
tmux new-session -s asgcn -c "$PWD"
```

`run.sh all`은 다음을 순서대로 수행한다.

1. `check_env.py --require-full-data --lock constraints/py312.txt --runtime-profile constraints/server.json`과 기본 CUDA 검사
2. EventHDR train/eval과 EventAid-R 전체 `inspect --validate-all`
3. EventHDR train 전체 CUDA graph topology scan, 최초·빈·희소 입력과 edge 수 상위 표본의 CUDA 학습 검사
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
`PROFILE_OUTPUT=runs/fast-profile.json`이다. 전수 scan은 edge guard 초과 표본을 찾고, 실제 CUDA probe는
edge 수 상위 3개 표본에서 configured loss·optimizer까지 포함한 학습 step과 peak allocated/reserved
VRAM을 잰다. 이는 기록된 GPU와 선택 표본에 한정된 실측 gate이며 절대 VRAM 보증이 아니다.
새 전수 scan은 events를 선택한 CUDA 장치로 옮겨 graph topology를 계산한다. CPU는 HDF5 읽기와
좌표·메타데이터 준비를 담당한다. `PROFILE_CPU_THREADS=4`가 profile의 CPU 보조 연산 기본값이고,
공통 wrapper는 torch/numpy import 전에 `OMP_NUM_THREADS`와 `MKL_NUM_THREADS`도 설정한다.
첫 프레임, 첫 zero-event 프레임, 최소 양수 node 표본은 중복을 제거하고 각각 초기 모델에서 추가로
검사한다. 밀집 표본 3개가 모든 입력의 수치 안정성을 보장한다고 해석하지 않는다.

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

## 4. 중단 후 profile 및 batch-boundary resume

### 사전검사 중단 후 이어가기

`runs/fast-profile.scan/index.json`과 작은 구간 파일들에 전수검사 기록이 저장된다. 128개 또는 30초 간격에
표본 경계에서 저장하며, 정상적인 interrupt/오류 시에도 완료된 표본을 저장한다. 강제 종료 시에는
마지막 원자적 commit 이후 구간을 다시 계산한다. data SHA-256·설정·topology 구현 계약이 같아야 한다.

```bash
PROFILE_RESUME=1 bash scripts/run.sh profile
```

GPU probe만 실패했어도 완료된 scan을 재사용할 수 있다. GPU probe 결과는 이어 붙이지 않고 새로
측정한다. 이미 통과한 최종 보고서는 이 명령으로 덮어쓰지 않는다. `all`을 다시 실행하지 않는다.

### AMP 첫 step 오류와 이전 profile 이관

FP16 기본 scale 65,536에서만 gradient가 비유한 값이 되고 scale 1/FP32는 유한했던 오류는 AMP
overflow 복구 전에 clip 검사가 종료시키던 문제였다. 공통 학습 step은 같은 프레임의 scale을 낮춰
최대 16번 재시도한다. 실패 시 가중치·optimizer를 업데이트하지 않고 BatchNorm buffer와 RNG를
되돌린다. recurrent/temporal state는 성공한 시도만 다음 프레임에 전달한다. loss NaN/Inf,
AMP가 꺼진 상태의 잘못된 gradient, 지속되는 overflow, 다른 CUDA/clip 오류는 숨기지 않는다.

기존 검사가 통과했고 아직 학습 checkpoint가 없는 경우, 종료된 작업에 대해 사용한다.

```bash
git pull --ff-only &&
export PROFILE_OUTPUT=runs/fast-profile2.json &&
PROFILE_REUSE_REPORT=runs/fast-profile.json bash scripts/run.sh profile &&
RESTART_TRAIN=1 bash scripts/run.sh train &&
bash scripts/run.sh calibrate &&
bash scripts/run.sh eval
```

Git 갱신 실패 시 아래 명령은 실행하지 않는다. 새 terminal에서는 `PROFILE_OUTPUT`도 다시 설정한다.
원본 `runs/fast-profile.json`은 보존된다. `PROFILE_REUSE_REPORT`는 완전한 기록·요약·data/config hash와
검토된 source 계약을 검증한 후 topology 통계만 새 보고서로 이관한다. legacy v1 보고서는 허용 목록의
clean commit/source hash 조합만 받고, 임의 수정된 보고서나 알 수 없는 구현은 재사용하지 않는다.
통계의 CPU/CUDA 출처를 보존하며 새로운 코드·GPU에서 수치·밀집 probe를 다시 실행한다.

metadata-only `runs/fast`은 `runs/fast.failed-*/train`으로 보존된다. `config.json`,
`preflight_gate.json`, `.data_hash_cache.json` 외 파일·하위 폴더 또는 checkpoint가 있으면 자동으로
옮기지 않는다. 기존 작업은 먼저 종료해야 한다. 이 옵션은 epoch 내부 학습을 복원하는 resume가 아니다.

### epoch 중간의 확인된 batch부터 학습 재개

직접 실행은 다음과 같다.

```bash
RESUME_CHECKPOINT="$PWD/runs/fast/last.pt" \
  bash scripts/run.sh train 2>&1 | tee logs/train-resume.log
```

동일한 동작을 저수준 wrapper로 실행하려면 다음 명령을 쓴다.

```bash
RESUME_CHECKPOINT="$PWD/runs/fast/last.pt" \
  bash scripts/train.sh configs/fast.json
```

두 wrapper 모두 `PROFILE_OUTPUT`을 따르며, 저수준 wrapper에서 `PREFLIGHT_REPORT`를 따로 지정하면
그 값이 우선한다. 이전 profile 이관을 사용했다면 `PROFILE_OUTPUT=runs/fast-profile2.json`을 유지한다.

현재 checkpoint는 기본 300초 간격, 각 epoch 끝, `Ctrl+C`/`SIGTERM` 또는 `MAX_HOURS` 요청 뒤의
**성공한 optimizer update 경계**에서 `last.pt`를 원자적으로 갱신한다. model·optimizer·scheduler·AMP
scaler·Python/NumPy/Torch/CUDA RNG뿐 아니라 현재 epoch의 다음 batch cursor, 처리 frame 수, 누적 loss,
AMP 재시도 수, 누적 학습 시간과 활성 sequence별 recurrent state·이전 prediction/target을 저장한다.
DataLoader가 미리 decode한 batch가 아니라 실제 optimizer update와 context commit이 끝나 확인된 batch만
cursor에 반영한다. 재개 시 저장된 schedule SHA-256과 cursor/frame 수/context identity를 dataset index와
대조한 뒤 다음 batch부터 읽으므로 완료 update를 중복하지 않는다.

시간을 나눠 실행하려면 scheduler walltime보다 짧은 예산을 지정한다. 예시는 fast run을 6시간 단위로
실행하는 경우다.

```bash
EXPERIMENT=fast MAX_HOURS=6 bash scripts/run.sh train
EXPERIMENT=fast RESUME_CHECKPOINT=runs/fast/last.pt MAX_HOURS=6 \
  bash scripts/run.sh train
```

정상 일시정지는 종료코드 75이며 상위 `run.sh`은 train 상태를 `PAUSED`로 기록하고 calibration/eval을
실행하지 않는다. 주기는 `CHECKPOINT_SECONDS=120`처럼 바꿀 수 있다. `SIGKILL`, 전원 차단 또는 scheduler
hard kill은 handler가 실행되지 않으므로 마지막 주기 checkpoint 뒤의 batch는 다시 계산될 수 있다.
checkpoint는 같은 configured run directory 안에 있어야 하며, source tree/Git 상태,
model·optimizer·scheduler·AMP, validation/data full SHA-256과 PyTorch/CUDA/cuDNN·GPU 이름/compute
capability·visible CUDA RNG state가 일치해야 한다. profile도 현재 config·전체 train data·source·동일
CUDA runtime에 다시 결합되어야 한다. 중단 중 `git pull`하거나 config/runtime을 바꾸지 않는다.

현재 새 학습은 batch 1에서 training protocol v7, sequence batch에서 v8을 기록한다. 기존 v5/v6
checkpoint의 reporting metadata는 calibration/evaluation에서 읽을 수 있지만, isolated epoch loader RNG와
mid-epoch schedule 계약이 없는 v5/v6 checkpoint를 v7/v8 실행에 exact training resume로 승격하지 않는다.
그 checkpoint는 작성 당시 source에서 epoch 경계 재개하거나, 현재 source에서는 새 output의 새 run을
시작한다. 다른 GPU나 source checkout으로 옮기는 것도 일반 weight load가 아니라 exact resume 요청이므로
거부될 수 있다. 상세 계약은 [EXPERIMENT.md](EXPERIMENT.md)의 resume 절을 따른다.

상위 runner는 `check`, `profile`, `train`, `calibrate`, `eval`, `all` stage를 제공한다. 학습 재개가 끝난 뒤에는
`bash scripts/run.sh calibrate`, 그다음 `bash scripts/run.sh eval`을 실행한다. 기존 training,
profile, calibration, evaluation artifact를 자동으로 건너뛰거나 덮어쓰지 않으므로, 부분 실패 복구 시
완료된 stage를 묵시적으로 재사용하지 않는다. CUDA profile 우회는 합성 비보고용 직접 CLI에서만
명시적으로 가능하고 scheduler wrapper 및 `all`에서는 차단된다.

한 dataset의 평가만 실패했으면 `eval-hdr` 또는 `eval-aid` stage를 사용한다. 같은 출력 root에
`EVAL_RESUME=1`을 지정하면 완료된 mode는 건너뛰고, 실패한 partial artifact를
`.incomplete-<UTC>-<PID>-<random>/<원래 이름>` 아래에 보존한 뒤 그 mode만 처음부터 다시 실행한다. frame-level resume은
지원하지 않으므로 중단된 mode 내부의 처리 완료 frame부터 이어 붙이지는 않는다. 이 복구는 같은
config/checkpoint/edge guard뿐 아니라 source/runtime/data/protocol까지 일치할 때만 사용한다.
실제 quality/benchmark writer와 복구 검사는 동일한 sibling `.<mode>.writer.lock`을 독점 획득한다.
완료 여부 검사와 partial 보존도 그 잠금 안에서 수행하므로 쓰는 중인 결과를 이동하지 않는다.
기존 잠금은 자동으로 stale 판정하거나 제거하지 않으며 원인 확인 전에는 재실행하지 않는다.

기존 `configs/aid-fast.json`의 8,192-event / radius 0.08 계약에는 실측 edge 상한 **7,475,202**를
`eval.max_graph_edges_override`에 평가 전용 기본 guard로 적용한다. model 설정과 checkpoint hash는 유지한다.
sample 37791의 ANN과 literal_eq15 T32 진단에서 peak allocated
678.06 MiB / reserved 922 MiB를 기록했고, 전체 51,512프레임의 ANN+SNN 9개 mode 평가가 완료됐다.
이 근거는 해당 데이터·전처리·graph 규칙에만 적용된다. 학습 guard와 다른 config에는 적용하지 않으며,
명시 CLI/env override가 config보다 우선한다. 셸은 별도의 숫자 기본값을 만들지 않으며 resume 판정·quality·benchmark는 같은 config 값을 해석한다.
입력/graph 규칙이나 batch·하드웨어가 달라지면 아래 scan/probe 및 실제 batch 메모리를 다시 측정한다.
사용자가 완료한 기존 9개 조건 결과와 checkpoint는 보존한다. 기존 scan/probe와 동일한 graph 규칙이면
이를 다시 수행하지 않는다. 새 배치 실행은 별도 output에서 검증하고, batching 변경만으로 재학습하지 않는다.

edge guard에서 중단됐다면 숫자를 추측해 전체 평가를 반복하지 않는다. 예를 들어 진행률
`32843/51512`에서 4,000,000-edge guard가 중단된 경우, 완료된 `[0,32843)` 구간의 상한과 나머지 구간의
정확한 topology scan을 결합한다. 이 명령은 edge 목록이나 모델 forward를 만들지 않으며 128 sample 또는
30초마다 `runs/fast/aid-topology-tail.scan/`에 원자적으로 기록한다.

```bash
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
python scripts/check_env.py --require-cuda --lock constraints/py312.txt \
  --runtime-profile constraints/server.json

# 선택 사항: 실패한 평가가 만든 검증 가능 hash cache를 첫 scan 전에 재사용한다.
cp -- runs/fast/eval-recovery-4m/aid/.data_hash_cache.json \
  runs/fast/aid-topology-tail.data_hash_cache.json

python -m asgcn_unet.cli scan-eval-topology \
  --config configs/aid-fast.json \
  --output runs/fast/aid-topology-tail.json \
  --start-index 32843 \
  --known-prefix-max-edges 4000000 \
  --cpu-threads 4
```

중단된 동일 scan은 다른 인자를 바꾸지 않고 위 명령 끝에 `--resume`을 붙인다. 출력의
`global_max_is_exact=true`를 확인하고 `global_max_actual_directed_edges`, `global_max_sample.dataset_index`와
`global_edge_guard_upper_bound`를 사용한다. `--start-index` 앞의 모든 frame이 지정 상한 이하였다라는 실제 성공
기록이 없으면 tail scan을 사용하지 말고 `--start-index 0`으로 전체를 조사한다. topology 최대값은 필요한
guard를 정할 뿐 그 edge 수의 모델 forward가 해당 GPU VRAM에서 안전함을 증명하지 않는다.

따라서 전체 평가 전에 최대-edge sample을 같은 할당 GPU에서 ANN과 가장 긴 SNN 설정으로 실제 실행한다.
아래 두 placeholder에는 scan JSON의 값을 넣고, 각 출력의 `graph_topology.actual_directed_edges`,
`gpu_memory.peak_allocated_mib`, `gpu_memory.peak_reserved_mib`, `runtime.cuda_visible_devices`를 확인한다.
probe가 OOM이면 guard만 더 올리지 않는다. tensor 참조·sparse/chunking·cache·precision·physical batch의
실측 메모리부터 점검하고 필요한 경우 더 큰 GPU allocation을 요청한다. 모델·그래프 규모를 줄이지 않는다.

```bash
DENSE_INDEX=<global_max_sample.dataset_index>
EDGE_GUARD=<global_edge_guard_upper_bound>

python scripts/probe_eval_sample.py \
  --config configs/aid-fast.json --checkpoint runs/fast/best.pt \
  --sample-index "$DENSE_INDEX" --max-graph-edges "$EDGE_GUARD" \
  --inference-mode ann --output runs/fast/aid-ann-dense-probe.json

python scripts/probe_eval_sample.py \
  --config configs/aid-fast.json --checkpoint runs/fast/best_snn.pt \
  --sample-index "$DENSE_INDEX" --max-graph-edges "$EDGE_GUARD" \
  --inference-mode snn --simulation-steps 32 --snn-dynamics literal_eq15 \
  --output runs/fast/aid-snn-t32-dense-probe.json
```

`CUDA_VISIBLE_DEVICES`가 비어 있고 PyTorch에 MIG 하나만 보인다는 사실만으로는 그 장치가 scheduler가
할당한 것인지 인증할 수 없다. scheduler가 제공한 allocation shell, job 환경 또는 MIG UUID로 identity를
확인한다. cluster가 visible-device token을 제공하면 그 값을 그대로 설정하며 물리 번호를 추측하지 않는다.
위 probe가 동일한 확인된 allocation에서 여유를 두고 통과한 뒤 새 평가 root로 EventAid-R 전체를 실행한다.

```bash
EDGE_GUARD=<global_edge_guard_upper_bound>
EXPERIMENT=fast \
EVAL_OUTPUT_ROOT="$PWD/runs/fast/eval-recovery-measured" \
EVAL_MAX_GRAPH_EDGES="$EDGE_GUARD" \
EVAL_RESUME=1 \
  bash scripts/run.sh eval-aid
```

`EVAL_MAX_GRAPH_EDGES`는 quality와 benchmark 양쪽의 inference-only guard이며 각 결과 protocol에
요청값과 effective 값을 기록한다. 위 숫자는 예시일 뿐이고 특정 MIG/GPU에서 안전하거나 모든 frame에
충분하다는 보장이 없다. 실측 없이 guard를 올리지 않으며 기존 output을 삭제하거나 같은 root로
덮어쓰지 않는다. `EVAL_RESUME=1`은 `metrics.json`과 `frames.csv`를 quality 완료 표식으로,
`benchmark.json`을 mode 완료 표식으로 사용한다. dataset별 상태 파일은 `eval-hdr.json`,
`eval-aid.json`이다.

## 5. Slurm: profile → train → calibrate → evaluation matrix

header의 partition/account/GPU type/walltime은 cluster 정책에 맞춰 수정한다. 기본 요청은 GPU 1개,
CPU 8개, RAM 32 GB이며 profile/calibration 12시간, train 48시간, evaluation 8시간이다. 저장소 root에서
제출하면 `SLURM_SUBMIT_DIR`을 project root로 사용한다. 다른 위치에서 제출할 때는
아래 `PROJECT_ROOT` 값을 해당 checkout으로 바꾼다. 활성화한 Conda의 `CONDA_PREFIX`는 모든 job에
전달하며, 같은 Conda Python을 `PYTHON_BIN`으로 명시할 수도 있다. `--export`에는 job별 필수 변수만
나열하며 login shell의 token, proxy, credential까지 전달할 수 있는 `ALL`은 사용하지 않는다.

앞 절의 full-data/decode 검사를 완료한 뒤 다음 dependency chain을 제출한다.

```bash
conda activate asgcn
unset SNN_DYNAMICS

profile_id=$(sbatch --parsable \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX" \
  server/profile.sbatch)

train_id=$(sbatch --parsable \
  --dependency="afterok:${profile_id}" \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0 \
  server/train.sbatch)

cal_id=$(sbatch --parsable \
  --dependency="afterok:${train_id}" \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0,CALIBRATION_SAMPLES=all \
  server/calibrate.sbatch)

for cfg in configs/hdr-fast.json configs/aid-fast.json; do
  sbatch --dependency="afterok:${cal_id}" \
    --export="PROJECT_ROOT=$PWD,CONDA_PREFIX=$CONDA_PREFIX,VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH=${cfg},CHECKPOINT_PATH=runs/fast/best.pt,INFERENCE_MODE=ann" \
    server/eval.sbatch
done

for cfg in configs/hdr-fast.json configs/aid-fast.json; do
  for dynamics in literal_eq15 standard_if; do
    for steps in 4 8 16 32; do
      sbatch --dependency="afterok:${cal_id}" \
        --export="PROJECT_ROOT=$PWD,CONDA_PREFIX=$CONDA_PREFIX,VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH=${cfg},CHECKPOINT_PATH=runs/fast/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS=${dynamics},SIMULATION_STEPS=${steps}" \
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
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0,RESUME_CHECKPOINT="$PWD/runs/fast/last.pt" \
  server/train.sbatch)
```

wrapper log의 공개 기본값은 host/job 식별자를 생략하고 config/checkpoint를 basename으로 표시한다. 정확한
hostname, scheduler job ID, project/config/checkpoint 경로는 비공개 로컬 진단에서만
`INCLUDE_PRIVATE_HOST_PROVENANCE=1`을 해당 `--export` 목록에 추가해 확인한다. 이 opt-in log는 공개하거나
첨부하지 않는다.

## 6. PBS/Torque: profile → train → calibrate → evaluation matrix

`select`, `ngpus`, queue/project resource 이름은 site마다 다르므로 `server/*.pbs` header를 제출 전에
확인한다. 저장소 root에서 제출하면 `PBS_O_WORKDIR`을 project root로 사용한다. 외부에서 제출할 때는
`-v`의 `PROJECT_ROOT` 값을 해당 checkout으로 지정한다. 모든 job에 같은 Conda 환경을 전달한다.

```bash
conda activate asgcn
unset SNN_DYNAMICS

profile_id=$(qsub -v CONDA_PREFIX="$CONDA_PREFIX" server/profile.pbs)

train_id=$(qsub \
  -W depend="afterok:${profile_id}" \
  -v CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0 \
  server/train.pbs)

cal_id=$(qsub \
  -W depend="afterok:${train_id}" \
  -v CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0,CALIBRATION_SAMPLES=all \
  server/calibrate.pbs)

for cfg in configs/hdr-fast.json configs/aid-fast.json; do
  qsub -W depend="afterok:${cal_id}" \
    -v CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH="${cfg}",CHECKPOINT_PATH=runs/fast/best.pt,INFERENCE_MODE=ann \
    server/eval.pbs
done

for cfg in configs/hdr-fast.json configs/aid-fast.json; do
  for dynamics in literal_eq15 standard_if; do
    for steps in 4 8 16 32; do
      qsub -W depend="afterok:${cal_id}" \
        -v CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0,RUN_BENCHMARK=1,CONFIG_PATH="${cfg}",CHECKPOINT_PATH=runs/fast/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS="${dynamics}",SIMULATION_STEPS="${steps}" \
        server/eval.pbs
    done
  done
done
```

PBS에서 resume chain을 시작할 때는 다음처럼 `RESUME_CHECKPOINT`를 넘긴다.

```bash
train_id=$(qsub \
  -v CONDA_PREFIX="$CONDA_PREFIX",VALIDATE_DATASET=0,RESUME_CHECKPOINT="$PWD/runs/fast/last.pt" \
  server/train.pbs)
```

PBS는 login environment 전체를 전달하는 `#PBS -V`를 사용하지 않는다. site CUDA module이 필요하면
`CUDA_MODULE`을 넘긴다. `PYTHON_BIN`을 명시한다면 같은 Conda 환경의 Python을 지정한다.
다른 checkout을 쓸 때는 `PROJECT_ROOT`도 `-v`로 명시한다.

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
ls -lh "${PROFILE_OUTPUT:-runs/fast-profile.json}"
ls -lh runs/fast/{last.pt,best.pt,best_snn.pt,history.json,config.json}
find runs -name metrics.json -o -name benchmark.json | sort
```

평가 artifact는 다음 위치에 있다.

```text
runs/eval/hdr/ann/
runs/eval/hdr/snn_<dynamics>_T<steps>/
runs/eval/aid/ann/
runs/eval/aid/snn_<dynamics>_T<steps>/
```

`runs/fast-profile.json`, `runs/fast-status/{check,profile,train,calibrate,eval}.json`과 각 run의 `metrics.json`,
`frames.csv`, `predictions/`, `benchmark.json`을 config, Git commit, scheduler log, `check_env.py` 출력과
함께 보존한다. profile과 평가 artifact의 공개 protocol은 host 절대경로/hostname을 저장하지 않는다.
`check_env.py`와 `inspect`의 `--include-private-host-provenance` 출력은 로컬 진단 전용이며 공개
artifact나 외부 첨부에 포함하지 않는다.
내부 SHA-256은 checkpoint·metadata·tensor의 재현 identity와 우발적 손상 검사용이지 전자서명이나
출처 인증이 아니다. 제출용 file hash는 접근통제된 immutable 실험 원장 또는 signed manifest에도
별도로 보존한다.

배포 전에는 저장소 밖 로컬 denylist를 사용해 complete non-shallow clone의 current
tree와 모든 local-ref reachable blob을 로컬에서 함께 검사한다. 실제 marker는 로컬 환경변수
`PRIVATE_MARKERS_B64`로도 주입할 수 있지만 GitHub secret·변수·workflow·log·artifact로 전송하지
않는다. 로컬 release gate의 `--require-external-patterns`는 빈 denylist를 거부한다.

```bash
python scripts/scan_private_text.py \
  --all-tracked --all-history --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

코드·문서·검증 기록을 검토하고 source commit을 확정한 뒤 clean
source에서 summary를 재생성하고 summary-only commit을 만든다. 생성 뒤 다른 tracked 파일 수정이나
source history rewrite가 필요하면 source commit 확정부터 반복한다. 같은 최종 SHA에서 위 로컬
실제-marker 검사와 다음 clean provenance gate를 통과해야 한다. CI는 실제 marker 없이 generic
current-tree/history 검사와 clean provenance를 수행한다. 원격 배포 후 같은 최종 SHA의 GitHub
Actions 필수 job 성공도 확인하되 로컬 실제-marker 검사와 혼동하지 않는다. 배포/CI 결과를 본문에
추가하기 위한 수정도 source 변경이므로, 결과는 해당 SHA의 Actions와 별도 로컬 배포 기록으로 남긴다.

```bash
python scripts/build_code_summary.py --check --require-clean-provenance
```

### MIG에서 전체 데이터 검사 후 profile만 실패한 경우

`check`가 끝난 뒤 새 `profile` 프로세스에서 `profile failed: Invalid device id`가 발생한 경우다.
PyTorch 2.13의 cuDNN version 조회는 내부에서 장치 수를 얻고 각 장치의 capability를 읽는다.
CUDA 초기화 전 MIG의 NVML 장치 수로 반복 범위가 정해지면, 첫 capability 조회에서 초기화된 실제
runtime 장치 수와 달라 존재하지 않는 index를 조회할 수 있다. [PyTorch 2.13 공식 구현](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/backends/cudnn/__init__.py#L83-L90)

수정된 profile은 `torch.cuda.init()`을 cuDNN version·장치 정보 조회보다 먼저 호출한다.
앞서 성공한 `check_env.py`는 별도 프로세스이므로 그 초기화 상태가 profile로 전달되지 않는다.
checkpoint RNG 저장·복원도 모든 CUDA 장치를 열거하기 전에 초기화한다. Conda 의존성, 모델,
데이터, GPU 할당과 `CUDA_VISIBLE_DEVICES`는 변경하지 않는다.

전체 decode가 완료됐고 이 오류가 topology scan 전에 발생해 profile/train 산출물이 없는 경우,
기존 저장소와 활성화된 Conda 환경에서 다음 단계만 실행한다. 데이터 재다운로드·재설치와
`all`의 전체 decode 반복은 필요 없다.

```bash
git pull --ff-only &&
bash scripts/run.sh profile &&
bash scripts/run.sh train &&
bash scripts/run.sh calibrate &&
bash scripts/run.sh eval
```

이는 완료된 데이터 검사를 되풀이하지 않는 재개 절차이며 CUDA profile을 생략하는 우회가 아니다.
기존 보고서와 `runs/fast-profile.scan/`은 자동으로 덮어쓰지 않는다. 같은 실패·중단 스캔은 앞의
`PROFILE_RESUME=1` 절차로 이어가고, 별도 새 검사라면 `PROFILE_OUTPUT`에 새 파일명을 지정해
원본 보고서와 journal을 모두 보존한다. JSON만 옮기면 기존 journal 때문에 새 검사가 거부된다.
이미 학습 checkpoint가 생긴 다른 실행에는 위 fresh-train
명령을 그대로 쓰지 말고 [batch-boundary resume](#4-중단-후-profile-및-batch-boundary-resume)를 따른다.

자주 중단되는 조건은 다음과 같다.

- 이전 `check_env.py`의 GPU 이름 조회에서 `AssertionError: Invalid device id`: MIG에서는 CUDA
  초기화 전 NVML 장치 수와 초기화 후 CUDA runtime이 열거하는 장치 수가 다를 수 있다. 검사는 이제
  `torch.cuda.init()` 뒤 장치 수를 읽고 실제 사용 가능한 장치만 조회한다. 저장소에서
  `git pull --ff-only`로 갱신한다. pull이 충돌하면 기존 변경을 보존하고 먼저 확인한다.
  전체 decode를 시작하기 전 이 오류로 중단됐고 profile/train 산출물이 없는 경우에는
  `bash scripts/run.sh all`을 다시 실행한다. 전체 decode 이후 profile 실패는 바로 위 재개 절차를
  따른다. 현재 Conda 환경과 데이터는 유지하며, scheduler의 `CUDA_VISIBLE_DEVICES`를 임의로
  덮어쓰거나 CUDA 검사를 끄지 않는다. 이 수정의 로컬 회귀검사는 CPU 모의 장치 기반이며 실제 MIG
  학습 완료를 의미하지 않는다.
- `CUDA device probe failed`: 실제 CUDA 초기화·장치 조회 실패로 중단한 것이다. GPU 할당과
  driver/PyTorch CUDA 호환성을 확인한다. 공개 진단에는 예외 종류만 출력하며, 원문 예외가 필요하면
  `check_env.py --include-private-host-provenance`를 비공개 진단에서만 사용한다.
- `CUDA available: false`: login node가 아닌 GPU allocation인지, CUDA wheel과 driver가 맞는지 확인한다.
- EventHDR 다운로드 중단: `bash scripts/get_hdr.sh --download`를 다시 실행해 `.part`에서 이어받는다.
  반복 실패하면 서버의 outbound HTTPS와 공식 공유 상태를 확인한다. 익명 endpoint나 임시 링크를
  임의 변경하거나 token·서명 URL을 공개 로그에 붙이지 않는다.
- `EventHDR ... exact official file set`: 직접 다운로드 또는 선택한 import가 끝났는지,
  train 51/eval 19 외 H5가 섞이지 않았는지 `get_hdr.sh --check`로 확인한다.
- `eventaid_r_zip must contain exactly 14`: `get_aid.sh --all`을 완료하고 ZIP을 압축 해제하지 않는다.
- `images/image... is missing 'event_idx'`: 일부 공식 H5는 해당 속성이 없다. 현재 로더는 이미지
  timestamp에서 누락된 인덱스를 읽기 전용으로 복원한다. 기존 데이터는 삭제하거나 다시 받지 않는다.
  이전 버전을 실행했다면 코드를 갱신하고 선택한 Conda 환경에서 `bash scripts/run.sh check`로
  전체 decode를 다시 확인한다. 복원 정책과 검증 범위는 [실험 프로토콜](EXPERIMENT.md#eventhdr-이벤트-인덱스)을
  참고한다. `timestamp` 누락·비단조 event 시각·유효하지 않은 저장 인덱스는 우회하지 않는다.
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
