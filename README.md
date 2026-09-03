# ASGCN-U-Net: Event-to-Frame 복원 실험

[![CI](https://github.com/costunder/asgcn-unet/actions/workflows/ci.yml/badge.svg)](https://github.com/costunder/asgcn-unet/actions/workflows/ci.yml)

EventHDR 전체 공개 배포본으로 학습하고 EventHDR 공식 eval과 EventAid-R 전체에서 평가하는
event-to-frame 연구 코드다. ASGCN graph encoder와 recurrent U-Net decoder를 결합하고,
ANN 및 ANN→SNN 변환 모델의 복원 품질·지연·발화율을 동일한 데이터와 평가 조건에서 비교한다.

## 설치 및 실행

Linux x86_64, glibc 2.28 이상, Git·Conda·curl이 필요하다. 학습·평가는 NVIDIA GPU에서 실행한다.
아래 명령은 서버 터미널 기준이다. scheduler를 사용하는 환경에서는 학습 전에 GPU를 할당받는다.

### 1. 저장소 받기

```bash
git clone https://github.com/costunder/asgcn-unet.git &&
cd asgcn-unet
```

이후 명령은 저장소 root에서 실행한다. 기존 checkout이 있으면 중복 clone 없이 해당 디렉터리를 사용한다.

### 2. 환경 설치

Python 3.12.14인 `asgcn` Conda 환경이 이미 있으면 환경 생성 명령은 생략한다.
다른 버전의 기존 환경은 덮어쓰지 않는다.

```bash
conda create -n asgcn --override-channels -c conda-forge python=3.12.14 pip
conda activate asgcn
bash scripts/setup.sh
```

활성화한 Conda 환경 하나에 설치하고 다운로드·학습·평가도 같은 환경을 쓴다. `base`에는 설치하지 않는다.
`constraints/server.json`이 Python **3.12.14**, PyTorch **2.13.0+cu126**, CUDA runtime **12.6**을
고정한다. `constraints/server.txt`가 pip·설치 도구를 포함한 전이 의존성의 버전과 배포 파일 SHA-256을
고정하며, `constraints/py312.txt`의 core/dev 버전도 함께 검사한다. `.env`는 필요 없으며 설치기는
기존 `.env`도 읽지 않는다. 이전 설치에서 전환한다면 [서버 환경 안내](docs/SERVER.md#1-환경-설치)를 따른다.
학습 실행 시 CUDA·전체 데이터·profile 검사를 수행하며, 검사 실패 시 학습을 시작하지 않는다.

### 3. 두 데이터셋 준비

서버에서 EventAid-R 14개 ZIP과 EventHDR train 51개·eval 19개의 H5를 직접 다운로드한다.

```bash
bash scripts/get_aid.sh --all &&
bash scripts/get_hdr.sh --download &&
python scripts/check_env.py --require-full-data --lock constraints/py312.txt \
  --runtime-profile constraints/server.json
```

중단되면 같은 다운로드 명령을 다시 실행한다. EventHDR은 Python 표준 라이브러리 HTTP로
[공식 OneDrive 폴더](https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N)에 익명 접근하며,
사용자 로그인·브라우저·쿠키 없이 `data/EventHDR/{train,eval}`에 저장한다. `.part` 이어받기,
재시도·만료 링크 갱신과 정확한 파일 집합·크기·SHA-256·HDF5 signature 검사를 수행한다.

이 경로는 2026-08-30 확인한 **문서화되지 않은 OneDrive 익명 호환 endpoint**를 사용하므로
Microsoft의 안정적 API 계약을 보장하지 않는다. 실접속 검증 범위는 70개 파일 metadata와 train/eval
각 H5의 첫 8-byte signature, 새 익명 token의 공유 접근 갱신이다. 전체 약 25.72GB 다운로드나 GPU
학습을 완료했다는 뜻은 아니다.
split별 다운로드와 이미 가진 ZIP/H5/shared storage의 선택적 가져오기는
[서버 데이터 안내](docs/SERVER.md#2-전체-데이터-배치)를 따른다.
최종 데이터는 약 50.4GB이며 가상환경·학습 결과, 선택적으로 보관하는 ZIP 원본 공간은 별도로 필요하다.

EventAid-R은 장면별 PNG/JPEG GT를 ZIP에서 직접 읽는다. `R-traffic`의 `_upload` 폴더와
`parts.txt`도 지원하며, 네 구간을 넘어 이벤트·GT를 연결하지 않는다. `inspect`는 장면별 영상 형식과
구간 정보를 출력한다. 기존 다운로드 파일을 변환하거나 다시 압축할 필요는 없다.

### 4. 전체 학습·보정·평가

GPU가 할당된 터미널에서 실행한다. 기본 설정은 EventHDR train 전체를 사용하는 **40 epoch 학습**이다.

```bash
conda activate asgcn
mkdir -p logs
set -o pipefail
bash scripts/run.sh all 2>&1 | tee logs/run.log
```

자동 순서: 전체 데이터 검사 → CUDA 전수 그래프 검사·수치/밀집 표본 검증 → ANN 학습 → SNN 보정 → 두 데이터셋 전체
ANN/SNN 평가. 결과는 `runs/`, 실행 로그는 `logs/run.log`에 저장된다.

연결이 끊겨도 계속 실행하려면 위 블록을 **`tmux new-session -s asgcn`으로 연 세션 안에서**
실행한다(`tmux` 설치 필요). 분리는 `Ctrl-b`, `d`, 재접속은 `tmux attach -t asgcn`이다.
SLURM/PBS 서버는 [scheduler 안내](#slurmpbs-scheduler)를 따른다.
중단 후에는 처음부터 `all`을 반복하지 말고 [재개 절차](#중단-후-재개와-결과-보호)를 따른다.
전체 데이터 검사 뒤 MIG `profile failed: Invalid device id`로 멈췄다면 [profile부터 재개](docs/SERVER.md#mig에서-전체-데이터-검사-후-profile만-실패한-경우)한다.

서버 재접속 후에는 기존 저장소로 이동하고 `conda activate asgcn`만 실행한다.
clone·환경 생성·설치를 매번 반복하지 않는다.

## 측정 기반 GPU 학습 설정

`configs/train.json`의 B1·`configs/batch.json`의 B4 기준선은 보존한다. 실제 서버에서 CUDA 관련
검사 **123개**가 통과했고, 같은 실제 EventHDR 측정 프레임 512개를 쓰는 18회 비교에서 B16+Triton은
`36.38/36.51 frame/s`, B16+Torch는 `9.07/9.02 frame/s`였다. 이 결과를 반영한 별도
`configs/fast.json`은 **B16 + Triton**, 전체 프레임, 40 epochs와 기존 모델 크기를 유지하고
`runs/fast`에만 기록한다. 이는 해당 512-frame window의 약 4.03배 처리량 결과이며 전체 epoch 시간,
모든 입력의 OOM 안전성 또는 40-epoch 수렴을 입증하지 않는다.

새 설정은 BN 통계와 optimizer 갱신 주기가 다른 별도 protocol이므로 B1/B4 checkpoint에서 이어 붙이지
않는다. 학습 전에 B16 실제 full-batch와 밀집/첫/빈/희소 입력의 CUDA gate를 새로 만든다.

```bash
conda activate asgcn
EXPERIMENT=fast bash scripts/run.sh profile
EXPERIMENT=fast MAX_HOURS=6 bash scripts/run.sh train
```

두 번째 명령은 최대 6시간 뒤 안전한 batch 경계에 `runs/fast/last.pt`를 저장하고 종료코드 75로
일시정지한다. 다음 작업 시간에는 같은 checkout·Conda 환경·data·profile을 유지한 채 이어간다.

```bash
conda activate asgcn
EXPERIMENT=fast RESUME_CHECKPOINT=runs/fast/last.pt MAX_HOURS=6 \
  bash scripts/run.sh train
```

완료 후에는 같은 `EXPERIMENT=fast`로 `calibrate`, `eval`을 실행한다. 호환되는 이전 profile이 있으면
topology 통계만 명시적으로 재사용할 수 있지만 GPU probe는 항상 새로 수행한다. 상세 측정 범위와
실행·재개 계약은 [성능 안내](docs/PERF.md#동일-실데이터-gpu-비교)와
[학습 안내](docs/TRAIN.md)를 따른다.

## 실험 범위

기본 실험 범위는 다음과 같다.

- EventHDR `train/1.h5`–`51.h5` 전체로 40 epoch ANN 학습
- EventHDR `eval/1.h5`–`19.h5` 전체로 마지막 epoch에 한 번 내부 평가
- 학습 전 EventHDR train 전체 graph topology를 조사하고 최고 밀도 표본에서 CUDA 학습 step 측정
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

### 연산 최적화

graph candidate 확장·edge compaction의 중복을 줄이고, Spline 집계에는 edge-message 전체를
역전파까지 보관하지 않는 custom autograd를 적용했다. SNN의 고정 첫 계층 전류는 forward당 한 번만
계산한다. graph·모델·sampling·학습 범위는 유지하며 기존 실행 명령도 같다. 출력·gradient 검증과
CPU 측정, 아직 측정하지 않은 GPU 성능의 구분은 [성능 기록](docs/PERF.md)에 정리했다.
새 source에는 새 CUDA 학습 검증이 필요하다. 검증된 이전 전수 통계만 명시적으로 이관할 수 있으며,
이전 GPU 측정값이나 학습 checkpoint의 exact-resume 검사를 우회하지 않는다.

## 데이터와 실험 역할

| 데이터 | 공개 파일 | 용량 | 이 저장소의 역할 |
|---|---:|---:|---|
| EventHDR train | H5 51개 | EventHDR 합계 약 25.72GB | ANN 학습 및 ANN→SNN 보정 |
| EventHDR eval | H5 19개 | 위 합계에 포함 | 마지막 epoch 내부 평가와 최종 평가 |
| EventAid-R | ZIP 14개 | 약 24.68GB | 학습·보정에 쓰지 않는 외부 평가 |

두 데이터의 합계는 약 **50.4GB**로 100GB 미만이다. 가상환경, checkpoint, prediction, 로그와
선택적 EventHDR import용 ZIP을 동시에 보관하면 추가 공간이 필요하다.

EventHDR 공식 배포는 train 51 H5와 eval 19 H5를 서로 다른 root로 제공한다. 공개 자료에는 이
70개 H5와 실제 촬영 physical scene 사이의 완전한 대응표가 없다. 따라서
`manifests/eventhdr_split.json`은 공식 train/eval root를 그대로 고정하며 H5 sequence file을 recurrent
state와 macro metric의 group 단위로 사용한다. 확인되지 않은 physical-scene 대응을 만들어내지
않는다.

`configs/train.json`의 `validate_every: null`은 EventHDR eval을 **40번째(마지막) epoch에 단 한 번만**
평가한다는 뜻이다. eval loss는 gradient에 들어가지 않고 여러 epoch 중 checkpoint를 고르는 데도
사용되지 않는다. 호환성을 위해 파일명은 `best.pt`지만, 이 설정에서는 마지막 epoch에서 한 번
계산한 finite macro SSIM을 가진 모델이다. EventAid-R은 그 이후 외부 일반화 평가에만 사용한다.

두 loader는 config에 명시된 `target_normalization.mode=integer_dtype_max`로 정수 target을
`[0,1]`에 놓은 뒤 동일한 `log1p(5000*x)/log1p(5000)` tone mapping을 적용한다. float target은
`known_scale` 또는 `already_normalized`를 명시해야 하며 NaN/Inf와 범위 위반은 즉시 거부한다.
frame별 percentile 보정은 `percentile_debug_only`와 `debug_only=true`를 함께 지정한 비보고용
진단에서만 열린다. EventAid-R은 event block `i`를 다음 GT `i+1`과 짝짓는 `target_offset: 1`을
사용한다. offset은 bool·실수가 아닌 정확한 정수만 받는다. 이는 이 저장소의 정렬 가정이지 ASGCN 논문
값은 아니다. EventHDR `images/image<index>`는 numeric suffix가 유일해야 하며 문자열순이 아닌 숫자순으로
읽는다. 저장된 `event_idx`가 없는 공식 H5는 timestamp에서 참조 packager의 predecessor 규칙으로
경계를 복원하고 그 출처를 기록하며, 원본 파일은 변경하지 않는다.
[EventHDR 인덱스 정책](docs/EXPERIMENT.md#eventhdr-이벤트-인덱스)을 참고한다.
EventAid-R full inspect는 event text의 원 timestamp min/max와 `timestamps.txt` interval의 span
ratio·offset·범위 이탈 수를 기록하되, 공식 14 ZIP의 단위가 실측으로 확인되기 전에는 이를 임의로 hard
fail 조건으로 만들지 않는다.

## 전체 실행에서 하는 일

`scripts/run.sh all`은 다음 순서를 fail-fast로 실행한다.

1. `check`: CUDA·locked dependency·전체 파일 coverage를 확인하고 두 데이터의 모든 sample을 decode
2. `profile`: EventHDR train graph를 CUDA에서 전수 조사하고 첫 프레임·빈/희소 입력 및 edge 수 상위
   표본 3개에서 CUDA forward/backward, optimizer step, peak allocated/reserved VRAM과 step time 측정
3. `train`: 검증된 `runs/profile.json`을 현재 config·data·source·CUDA runtime에 다시 결합한 뒤
   EventHDR train 전체 40 epoch ANN 학습
4. `calibrate`: EventHDR train의 모든 calibration sample로 `best_snn.pt` 생성
5. `eval`: EventHDR eval과 EventAid-R 전체에 대해 ANN 1회 및
   `literal_eq15`/`standard_if` × `T=4,8,16,32` 평가·benchmark

기본 `PROFILE_TOP_DENSITY=10`은 edge 수 상위 10개 interval의 topology를 상세 기록하고,
`PROFILE_SAMPLES=3`은 그중 3개를 실제 학습 step으로 측정한다. 전체 scan 개수를 3개로 줄이지 않는다.
첫 프레임·첫 zero-event 프레임·최소 양수 node 표본도 중복을 제거해 각각 초기 모델에서 검사한다.
새 scan은 graph 계산을 CUDA에서 수행하고 CPU는 HDF5 읽기·메타데이터 처리에 사용한다.
`PROFILE_CPU_THREADS=4`가 CPU 보조 연산의 기본값이며 GT 픽셀의 전수 검증은 앞선 `check`가 담당한다.
이 profile은 기록된 GPU에서 선택된
최고 밀도 표본이 통과했다는 실측 gate이지 전체 40 epoch의 모든 미래 step에 대한 절대 VRAM 보증은
아니다. profile artifact에도 `absolute_vram_guarantee=false`가 고정된다.

전체 품질 평가는 모든 sample을 사용한다. 기본 `BENCHMARK_STEPS=100`은 GPU compute latency를 재는
별도 timing 반복 수이며 품질 평가를 100 sample로 줄이는 설정이 아니다. `save_predictions: 20`도
PNG 저장 수만 제한하고 metric 계산 범위는 제한하지 않는다.
보고 가능한 `metrics.json`은 `eval.max_samples=null`인 quality evaluation에서만 생성된다. 값이
설정된 부분 quality 실행은 기본적으로 시작 전에 거부하며, 합성 테스트용 명시적 non-reporting
우회로 실행해도 `report_eligible=false`와 그 이유가 기록된다. 표본 수를 갖는 compute-only
benchmark는 이 quality 계약과 별도로 sampling provenance를 기록한다.

실행될 명령표만 확인하려면 데이터와 GPU 없이 다음을 사용할 수 있다. 실제 실행을 대신하지 않는다.

```bash
DRY_RUN=1 bash scripts/run.sh all
```

## 중단 후 재개와 결과 보호

### 사전검사가 중단된 경우

새 버전은 `runs/profile.scan/`에 128개 또는 30초 간격(표본 처리 경계)으로 검사 기록을 원자적으로
저장한다. 중단한 같은 scan은 다음처럼 재개한다. 이미 통과한 최종 보고서는 덮어쓰지 않는다.

```bash
PROFILE_RESUME=1 bash scripts/run.sh profile
```

`preflight-topology` 100%는 통계 검사 완료이며, 이어지는 GPU 검증까지 성공한 뒤 최종
`runs/profile.json`의 `status=passed`를 확인해야 학습으로 이어갈 수 있다.

### 이전 버전의 첫-step AMP 오류에서 복구

기존 `profile.json`은 통과했지만 첫 학습 step의 FP16 gradient overflow로 종료된 경우,
수정된 코드로 갱신한 뒤 다음 순서로 실행한다. 기존 데이터·보고서는 삭제하지 않는다.

```bash
git pull --ff-only &&
export PROFILE_OUTPUT=runs/profile2.json &&
PROFILE_REUSE_REPORT=runs/profile.json bash scripts/run.sh profile &&
RESTART_TRAIN=1 bash scripts/run.sh train &&
bash scripts/run.sh calibrate &&
bash scripts/run.sh eval
```

첫 명령의 갱신이 실패하면 다음 명령은 실행하지 않는다. 다른 터미널로 이어갈 때에도
`export PROFILE_OUTPUT=runs/profile2.json`을 설정한다. 이관은 출처가 확인된 전수 topology 통계만
재사용하고 첫/빈/희소·밀집 표본의 GPU 검증은 새로 수행한다. 이전 CPU 통계는 CPU 출처로 남으며,
이전 GPU 수치가 새 측정으로 둔갑하지 않는다. 출처·데이터·설정이 맞지 않으면 이관을 거부한다.

`RESTART_TRAIN=1`은 **이전 작업을 종료한 상태에서**, epoch checkpoint 없이 metadata만 남은
`runs/train`을 `runs/train.failed-*/train`에 보존하고 새 학습을 시작한다. checkpoint·history·알 수 없는
파일이 있으면 거부한다. 완료된 epoch가 없는 실행의 재시작이며, 중간 optimizer step의 복원이 아니다.
AMP overflow는 scale을 낮춰 같은 프레임을 최대 16번 재시도하고 실패한 시도의 BatchNorm·난수 상태를
복원한다. 프레임을 버리지 않으며, 비유한 loss나 지속되는 오류는 그대로 중단한다.

### 학습 중간 checkpoint와 재개

학습은 기본 300초 간격과 매 epoch 종료 시, **성공한 optimizer update 뒤의 안전한 batch 경계**에서
`last.pt`를 원자적으로 갱신한다. checkpoint에는 model, optimizer, scheduler, AMP scaler, RNG,
현재 epoch의 다음 batch cursor·누적 metric과 시퀀스별 recurrent/temporal context가 들어간다.
따라서 epoch 중간 checkpoint에서도 이미 반영한 batch를 다시 update하지 않고 이어간다.
`Ctrl+C`, `SIGTERM` 또는 `MAX_HOURS`는 다음 안전 경계에서 저장한 뒤 종료코드 75로 끝난다.
`run.sh all`도 이때 train 상태를 `PAUSED`로 기록하고 calibration/eval로 넘어가지 않는다.

기본 B1 run을 재개하는 명령은 다음과 같다.

```bash
RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  bash scripts/run.sh train 2>&1 | tee -a logs/train.log
```

`run.sh`은 `check`, `profile`, `train`, `calibrate`, `eval`, `all` stage를 각각 실행할 수 있다. `train`은
기본적으로 통과한 `runs/profile.json`을 요구하고, 이를 현재 config·전체 train data digest·source·GPU
runtime에 다시 결합한다. 학습 재개가 끝나면 `bash scripts/run.sh calibrate`, 이어서
`bash scripts/run.sh eval`을 실행한다. 기존 artifact를 발견해도 자동으로 건너뛰거나 덮어쓰지 않으므로,
부분 실행을 복구할 때 완료된 stage를 임의로 다시 실행하지 않는다. 합성 fixture용
`--allow-unverified-preflight` 우회는 checkpoint에 비보고용으로 영구 기록되고 `all` 및 scheduler
wrapper에서는 허용되지 않는다.

dataset 하나의 평가만 복구할 때는 `eval-hdr` 또는 `eval-aid`를 사용한다. 예를 들어 EventAid-R가
configured edge guard에서 중단됐고, 별도 topology 및 VRAM 측정으로 4,000,000 edges가 안전하다고
확인한 경우 다음처럼 **새** 출력 root에서 그 dataset의 전체 ANN+SNN 행렬을 처음부터 실행한다.

```bash
EXPERIMENT=fast \
EVAL_OUTPUT_ROOT="$PWD/runs/fast/eval-recovery-4m" \
EVAL_MAX_GRAPH_EDGES=4000000 \
  bash scripts/run.sh eval-aid
```

`EVAL_MAX_GRAPH_EDGES`는 quality와 benchmark에 같은 inference-only guard override를 전달하며 결과
protocol에 기록된다. 이 예시 값은 9.5 GiB를 포함한 임의의 GPU에서 안전하거나 충분하다는 보장이
아니므로 실측 없이 사용하지 않는다. quality evaluation은 frame-level resume을 지원하지 않는다.
실패한 기존 directory나 완료된 다른 dataset 결과를 삭제·덮어쓰지 말고 매 복구 시 고유한
`EVAL_OUTPUT_ROOT`를 선택한다. 상태는 각각 `eval-hdr.json`과 `eval-aid.json`에 기록된다.

resume 시 model, optimizer, scheduler, AMP scaler, RNG, history뿐 아니라 config, 상대 data identity,
현재 epoch cursor/context, 전체 data SHA-256, source tree hash와 GPU protocol을 교차검증한다.
같은 중단·재개 cycle의 source를 바꾸거나 `git pull`하지 않는다. `SIGKILL`, 전원 차단 또는 scheduler의
hard kill은 handler가 실행되지 않으므로 마지막 300초 주기 checkpoint 이후의 성공 batch는 다시 처리될
수 있다. scheduler walltime보다 짧게 `MAX_HOURS`를 잡아 정상 일시정지 시간을 남긴다.
주기를 바꾸려면 예를 들어 `CHECKPOINT_SECONDS=120`을 지정한다. `validate_every: null`인 run은
계획한 terminal epoch도 protocol에 봉인하므로 마지막 평가를 마친 뒤 epochs만 늘려 같은 run을
재개할 수 없다. 연장 학습은 새 output directory의 새 protocol로 시작한다. 계약이 일치하지 않으면
조용히 다른 실험을 이어 붙이지 않고 중단한다.
EventHDR 보고용 ANN은 문법적으로 유효한 반복 `best_validation_macro_ssim` checkpoint도 허용하지 않고,
계획 epoch에서 완료된 `single_final_epoch` terminal validation을 반드시 요구한다.

calibration은 clean `ann_inference` checkpoint만 받고, 학습 당시 EventHDR train 전체 byte digest,
transform, final split manifest와 source tree가 현재 실행과 일치해야 한다. 또한 manifest가 가리키는
모든 training sample을 dataset index `0..N-1` 순서로 정확히 한 번씩 사용해야 `sealed=true`가 된다.
일부 sample만 지정하면 기본 실행은 변환 전에 실패한다. 테스트용 `--allow-unsealed-calibration`으로만
부분 보정을 만들 수 있고, 결과에는 불일치 이유와 `report_eligible=false`가 영구 기록된다.
이 override를 명시한 사실 자체가 taint이므로 sample이 우연히 전체여도 `sealed=false`다.
validation protocol v7은 train/validation index의 sample 수·group별 수·ordered identity SHA-256도
봉인하며, 보고용 ANN은 `max_train_samples=max_val_samples=null`인 전체 학습·검증만 허용한다.
SNN model state에는 실제 시도 수 `calibration_attempts`, 32-byte
`calibration_commitment_digest`, `calibration_commitment_sealed`를 persistent buffer로 저장한다.
protocol·선택 수·sampling과 valid/minimum/dead-channel summary core의 commitment를 metadata summary,
layer별 valid count와 `calibration_activation_max`(관측 raw maximum),
`normalization_scale`(식 (6)의 effective scale), `dead_channel_mask`까지 대조한다. dead channel은 raw
maximum과 mask에는 그대로 남고 effective scale만 1이므로 저장 후 strict reload에서도 summary가
변하지 않는다. 따라서
부분 보정 tensor에 전체 보정 metadata만 이식하거나 선택 metadata만 사후 재라벨링한 checkpoint는
load 단계에서 거부된다.
기존 calibration 또는 평가 결과도 자동 덮어쓰지 않는다. 의도적으로 calibration만 다시 만들 때는
`OVERWRITE_CALIBRATION=1`을 명시할 수 있지만, 이미 완료된 평가 artifact는 보존하거나 config의
`eval.output_dir`를 새 경로로 바꿔 별도 run으로 실행해야 한다.

주요 출력은 다음과 같다.

```text
runs/train/
├── config.json
├── history.json
├── last.pt                       # 학습 재개용 전체 상태
├── best.pt                       # ANN inference용 clean checkpoint
└── best_snn.pt                   # 보정된 SNN graph encoder checkpoint

runs/profile.json                 # 전체 topology + 최고 밀도 CUDA 학습-step 사전검증

runs/status/
├── check.json
├── profile.json
├── train.json
├── calibrate.json
└── eval.json                     # RUNNING/COMPLETED/FAILED stage 상태

runs/eval/hdr/ann/
runs/eval/hdr/snn_literal_eq15_T{4,8,16,32}/
runs/eval/hdr/snn_standard_if_T{4,8,16,32}/
runs/eval/aid/ann/
runs/eval/aid/snn_literal_eq15_T{4,8,16,32}/
runs/eval/aid/snn_standard_if_T{4,8,16,32}/
```

각 평가 폴더에는 `metrics.json`, `frames.csv`, `predictions/`, `benchmark.json`이 생긴다. 보고용 ANN은
검증된 CUDA preflight를 통과한 clean `ann_inference` lineage를, 보고용 SNN은 그 ANN에서 생성한
`calibration_protocol.sealed=true` lineage를 요구한다. config/model/checkpoint file·tensor,
현재 eval data content·transform·manifest·coverage·sampling, source/runtime/precision 계약과 SHA-256을
`metrics.json`과 `benchmark.json`에 기록한다. 평가 시에도 calibration의 transform·final manifest·전체
sample identity/sampling·runtime을 ANN train index commitment와 비교하고, source ANN의 epoch·selection·
terminal validation을 포함한 reporting contract도 다시 검증한다. 합성 테스트에서만 쓰는 명시적 unsealed-checkpoint
우회는 결과의 `report_eligible=false`와 이유를 영구 기록하며 public wrapper에는 노출하지 않는다.
EventHDR lineage는 planned/completed/checkpoint epoch가 같고 validation selection이
`single_final_epoch_macro_ssim`인 경우만 보고 가능하다.
source ANN의 training dataset type도 EventHDR로 고정하고 training protocol의 seed·optimizer·scheduler·
loss·data order·AMP·runtime을 public training config와 교차검증한다. 평가 dataset은 명시적
`expected_file_count`와 final/fixed manifest의 정확한 파일 집합을 가져야 하며, EventHDR quality/benchmark는
현재 file bytes·transform·manifest가 source ANN validation 계약과 같아야 한다. protocol SHA-256에는
실제 inference mode, SNN T와 effective dynamics도 포함된다.
quality evaluation은 추가로 `eval.max_samples=null`을 요구하므로 부분 `metrics.json`이 보고 가능 상태로
표시되지 않는다. benchmark의 고정 warmup/측정 step은 품질 표본 제한과 다른 compute protocol이다.
JSON/CSV와
checkpoint는 temporary file을 완성한 뒤 원자적으로 교체하며 JSON은 NaN/Infinity를 허용하지 않는다.
SHA-256 seal은 재현 identity와 우발적·부분적 metadata 변조 탐지용이지 전자서명이 아니다. 악의적 주체가
checkpoint와 모든 내부 hash를 함께 다시 작성하는 상황까지 인증하려면 산출물 hash를 별도 immutable
실험 원장이나 서명 저장소에 보존해야 한다.
평가 직후 target, prediction, metric, graph diagnostic, latency와 `dt_us`의 비정상 수치도 거부한다.
품질 지표는 PSNR(완전 일치 시 120 dB 상한), Gaussian SSIM, RMSE, temporal L1과
micro/H5-or-ZIP-group macro 집계를 포함한다. `configs/hdr.json`과 `configs/aid.json`은 품질 비교의
기본 precision을 FP32/TF32 off로 명시하며, `amp_fp16`과 `bf16`을 선택한 별도 run은 요청값·실제
autocast dtype·parameter dtype·TF32 적용 여부를 결과에 기록한다. benchmark는 data I/O와
host-to-device 이동을 제외한 model compute latency, FPS, percentile, graph 처리량, spike rate와 GPU
peak memory를 기록한다. 이 compute-only memory 범위에는 GPU event input과 model 실행은 포함하지만
품질 계산에만 쓰는 ground-truth target은 CPU에 유지해 제외한다. PyTorch GPU latency를 FPGA/ASIC
latency나 에너지로 해석하면 안 된다.

## SLURM/PBS scheduler

클러스터 batch job은 저장소 root에서 제출한다. 아래는 실측 선택인 `EXPERIMENT=fast`를 6시간
학습 구간으로 실행하는 SLURM profile→학습→전체 calibration 의존성이다.

```bash
conda activate asgcn
profile_id=$(sbatch --parsable \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",EXPERIMENT=fast server/profile.sbatch)
train_id=$(sbatch --parsable --dependency=afterok:${profile_id} \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",EXPERIMENT=fast,MAX_HOURS=6,CHECKPOINT_SECONDS=300 server/train.sbatch)
cal_id=$(sbatch --parsable --dependency=afterok:${train_id} \
  --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",EXPERIMENT=fast server/calibrate.sbatch)
```

ANN 두 평가와 SNN 전체 행렬을 dependency로 제출한다.

```bash
for config in configs/hdr-fast.json configs/aid-fast.json; do
  sbatch --dependency=afterok:${train_id} \
    --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",EXPERIMENT=fast,CONFIG_PATH="$config",CHECKPOINT_PATH=runs/fast/best.pt,INFERENCE_MODE=ann \
    server/eval.sbatch
done

for config in configs/hdr-fast.json configs/aid-fast.json; do
  for dynamics in literal_eq15 standard_if; do
    for timestep in 4 8 16 32; do
      sbatch --dependency=afterok:${cal_id} \
        --export=PROJECT_ROOT="$PWD",CONDA_PREFIX="$CONDA_PREFIX",EXPERIMENT=fast,CONFIG_PATH="$config",CHECKPOINT_PATH=runs/fast/best_snn.pt,INFERENCE_MODE=snn,SNN_DYNAMICS="$dynamics",SIMULATION_STEPS="$timestep" \
        server/eval.sbatch
    done
  done
done
```

SLURM `--export`에는 각 job이 실제로 쓰는 변수만 명시한다. login shell 전체를 전달하면 token, proxy,
credential 같은 무관한 환경변수도 compute node와 job 환경에 복제될 수 있으므로 `ALL`은 사용하지 않는다.
학습 resume job에는 `EXPERIMENT=fast`, `MAX_HOURS=6`과
`RESUME_CHECKPOINT="$PWD/runs/fast/last.pt"`를 `--export`에 추가한다. 기존 B4 결과를 계속할 때만
`EXPERIMENT=batch`와 `runs/batch` 경로를 사용한다.
각 job은 전달받은 `CONDA_PREFIX`의 Python을 사용한다. 같은 환경의 Python을 `PYTHON_BIN`으로
명시할 수도 있다. PBS/Torque에도 `-v CONDA_PREFIX="$CONDA_PREFIX"`를 전달한다.
PBS/Torque용 동등 wrapper는 `server/profile.pbs`, `server/train.pbs`, `server/calibrate.pbs`,
`server/eval.pbs`이며
`qsub -W depend=afterok:<job-id>`로 같은 의존성을 건다. `#SBATCH`/`#PBS`의 GPU, memory, walltime,
queue, account는 학교 scheduler 정책에 맞게 조정해야 한다. 자세한 서버 운용은
[docs/SERVER.md](docs/SERVER.md), 실험 정의는 [docs/EXPERIMENT.md](docs/EXPERIMENT.md)에 있다.
wrapper log는 기본적으로 hostname/job ID를 생략하고 config/checkpoint는 basename만 남긴다. 로컬 진단에서
정확한 host와 경로가 꼭 필요할 때만 `INCLUDE_PRIVATE_HOST_PROVENANCE=1`을 제출 변수에 추가하고, 그 log는
공개하거나 첨부하지 않는다. 다만 Slurm의 `slurm-...-<job-id>.out/.err`와 PBS의
`<job-name>.o<job-id>` 같은 raw scheduler log는 **파일명 자체에 job ID가 있으므로 그대로 공개하지 않는다.**
공개 전에는 기본 provenance로 생성한 raw log를 `logs/public/train.stdout.log` 같은 중립 파일명으로 복사하고,
저장소 밖 로컬 denylist를 사용해 그 공개 후보의 내용도 로컬에서 검사한다. scan 실패 파일과
`INCLUDE_PRIVATE_HOST_PROVENANCE=1`로 만든 opt-in log는 중립명으로 바꿔도 비공개다.

```bash
python scripts/scan_private_text.py logs/public/train.stdout.log \
  --root "$PWD" --require-external-patterns \
  --extra-patterns /path/outside/repository/private_markers.txt
```

## 현재 검증 상태와 한계

- CI는 Linux Conda 고정 profile의 실제 설치·버전 일치·전체 회귀검사와 Ubuntu/Windows의
  Python 3.10·3.11·3.12 호환성을 검사한다. 검증 결과는 사용한 commit의 Actions에서 확인한다.
- `code_summary.md`는 source commit과 파일별 SHA-256을 기록하는 전체 text snapshot이다.
  CI는 snapshot 일치와 source provenance를 확인한다. 개인정보 검사는 현재 tracked text와 Git history를
  대상으로 하며, 실제 식별자는 저장소 밖 로컬 검사에만 사용한다.
  상세 검증 기록과 유지관리 절차는 [인계서](hand_off.md#14-테스트-상태와-검증-범위)에 있다.
- 환경 검사와 profile은 CUDA를 먼저 초기화한 뒤 cuDNN·장치 정보를 읽고, checkpoint RNG도 초기화 후
  장치를 열거한다. 실제 초기화 실패나 CUDA 불가는 우회하지 않는다.
- 코드의 unit/integration test와 Linux 의존성 검사는 구성되어 있지만, EventHDR+EventAid-R 전체
  실데이터를 사용한 `runs/profile.json`, CUDA 40-epoch 학습·전체 행렬 실행, A6000/A100 peak
  memory·runtime·latency artifact는 이 로컬 검증에서 생성하지 않았다. 따라서 README는 실행 절차
  설명이지 측정 완료 보고서가 아니다.
- 공개 자료에 EventHDR H5↔physical-scene 완전 대응표가 없어 공식 train/eval root 이상의
  scene-disjoint 주장을 하지 않는다.
- 원 논문의 동적 asynchronous K-hop update, pooling/classifier, energy model은 포함하지 않는다.
- 반도체 RTL/FPGA/ASIC, event compression/transport, 실제 전력·에너지 측정은 후속 과제 범위다.
- B16+Triton의 실측은 조건당 측정 512프레임 범위이며 전체 epoch/수렴 결과는 아직 없다. 재개 단위는
  기본 300초마다 저장된 성공 batch 경계다. 전체 실행 시간과 저장 공간은 서버 GPU, filesystem,
  dataset decode 속도에 따라 달라진다.

코드 전체 스냅샷은 [code_summary.md](code_summary.md), 인수인계와 연구상 주의점은
[hand_off.md](hand_off.md)를 참조한다.
