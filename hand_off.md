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
