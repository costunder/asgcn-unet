# ASGCN Event-to-Frame 프로젝트 인수인계

## 0. 문서 기준

- 저장소: private `costunder/asgcn-event-reconstruction`
- 브랜치: `main`
- 갱신일: 2026-08-29, Asia/Seoul
- 구현 기준: 이 문서와 `code_summary.md`가 포함된 동일 commit
- 정확한 commit 확인: clone 후 `git rev-parse HEAD`

`hand_off.md`는 설계, 데이터 계약, 실행 절차, 검증 상태와 남은 한계를 설명한다.
`code_summary.md`는 generated 문서 두 개를 제외한 프로젝트 text file을 다음 형식으로 합친 원문
스냅샷이다.

```text
# 파일경로
코드내용

# 파일경로
코드내용
```

두 파일을 다른 ChatGPT에 함께 전달하면 설명과 실제 코드를 직접 대조할 수 있다. H5/ZIP,
checkpoint, `.venv`, 로그, prediction은 포함하지 않는다. 사용자가 요청해 원문 HWP와 production
dummy/synthetic 실행 코드는 삭제했으며 Git에 포함하지 않았다. `tests/fixtures.py`의 작은 생성기는
CI에서 loader contract를 검증하기 위한 test-only fixture다.

## 1. 목표와 정확한 현재 수준

EventHDR 실제 이벤트에서 luminance frame을 복원하는 모델을 학습하고, EventHDR 공식 eval과
EventAid-R에서 품질·지연·외부 일반화를 평가하는 소프트웨어 연구 기반이다.

정확한 명칭:

> ASGCN-inspired edge-conditioned graph encoder + residual U-Net/ConvGRU event-to-frame
> reconstruction prototype, with a calibrated graph-activation rate proxy

구현된 범위:

- H5/ZIP 원본 직접 로딩
- event-to-target 정렬과 strict input validation
- bounded causal event graph
- graph message passing, feature rasterization, U-Net, ConvGRU
- ANN 학습, exact epoch resume, macro-SSIM model selection
- BN folding과 activation threshold calibration
- ANN/rate-proxy 품질·지연·메모리 평가
- MobaXterm/SSH, Bash, SLURM, PBS/Torque, Docker 실행 골격

구현하지 않은 범위:

- 원 ASGCN 논문의 정확한 B-spline/SplineConv
- membrane/leak/reset을 timestep별로 전개하는 IF/LIF SNN
- surrogate-gradient SNN 학습
- event compression 또는 network transport protocol
- RTL, FPGA, ASIC, NPU kernel, DMA, on-chip buffer
- 전력·에너지·면적·공정 결과

따라서 과제 전체의 반도체 기반 통합 시스템이 완성된 것은 아니다. 지금 코드는 알고리즘/서버
실험의 재현 가능한 출발점이다.

## 2. 데이터 역할

| 단계 | 데이터 | 용도 |
|---|---|---|
| ANN 학습 | EventHDR train | weight 최적화 |
| 모델 선택 | EventHDR holdout | 물리 scene split 확정 후 macro SSIM과 `best.pt` |
| rate 보정 | EventHDR train | BN folding·threshold |
| 내부 최종시험 | EventHDR 공식 eval | 학습 후 고정 평가 |
| 외부 최종시험 | EventAid-R | 학습·보정 없이 일반화 평가 |

EventAid-R 결과로 hyperparameter나 threshold를 다시 선택하면 더 이상 잠금 외부시험이 아니다.

현재 `manifests/eventhdr_split.json`은 다음 파일 번호 임시 분할이다.

- train: `1.h5`–`47.h5`
- validation: `48.h5`–`51.h5`
- status: `provisional`

물리 scene 대응표가 없으므로 동일 장면 누수 여부는 아직 확정하지 못했다. `train()`은 manifest
status가 `final`이 아니면 중단한다. 오직 `configs/hdr_smoke.json`만
`train.allow_provisional_split=true`로 비보고용 1 epoch를 허용한다.

## 3. 전체 pipeline

```text
EventHDR H5 / EventAid-R ZIP
  -> dataset-specific frame/event alignment
  -> strict schema, boundary, timestamp, coordinate, polarity checks
  -> sample {events[N,4], target[C,H,W], sensor_size, metadata}
  -> bounded vectorized causal graph
  -> ANN 또는 calibrated rate-proxy graph encoder
  -> cell-wise mean feature rasterization
  -> residual U-Net + analog ConvGRU
  -> sigmoid luminance frame [1,C,H,W]
  -> reconstruction loss 또는 quality/temporal/latency/memory metrics
```

핵심 진입점:

- CLI: `src/asgcn_recon/cli.py`
- orchestration: `src/asgcn_recon/engine.py`
- sample model forward: `src/asgcn_recon/model.py::ASGCNReconstructor.forward_sample`
- graph: `src/asgcn_recon/graph.py`
- metrics: `src/asgcn_recon/metrics.py`

## 4. 저장소 구조

```text
configs/                 train/eval/smoke JSON
constraints/             Python 3.12 exact dependency constraints
docs/                    server와 experiment 문서
manifests/               EventHDR split, EventAid-R URL/표시 크기
scripts/                 setup/check/download/train/eval wrapper
server/                  SLURM sbatch와 PBS template
src/asgcn_recon/         production package
tests/                   CI-only fixtures와 regression tests
README.md                MobaXterm end-to-end 절차
code_summary.md          project text snapshot
hand_off.md              이 문서
```

`data/`, `runs/`, `logs/`, H5/ZIP/PT/HWP, venv와 cache는 Git ignore다.

## 5. 공통 sample 계약과 target domain

| key | 형식 | 의미 |
|---|---|---|
| `events` | float32 `[N,4]` | `[x,y,t,p]` |
| `target` | float32 `[C,H,W]` | `[0,1]` luminance target |
| `sample_id` | string | `<scene>/<frame>` |
| `sensor_size` | `(H,W)` | crop 이후 출력 크기 |
| `metadata` | dict | scene, source, timestamp, `dt_us` 등 |

공통 전처리:

- polarity `0/1`을 `-1/+1`로 통일
- event 수가 `max_events`보다 크면 전체 시간 구간에서 등간격 subsample
- crop 밖 event 제거 후 좌표 원점 이동
- integer image를 dtype 최대값으로 `[0,1]` 정규화
- RGB는 BT.709 luminance
- EventHDR와 EventAid-R 모두 `log1p(5000*x)/log1p(5000)` 적용

마지막 조치는 모델 output의 수치 domain을 통일한다. 서로 다른 카메라의 radiometric response가
같다는 뜻은 아니며, 외부 일반화 해석에 sensor/domain 차이는 남는다.

## 6. EventHDR loader와 검증

기대 구조:

```text
events/xs, events/ys, events/ts, events/ps
images/imageXXXXXXXXX
  attrs.event_idx
  attrs.timestamp
```

정렬 규칙은 이전 selected image의 `event_idx`부터 현재 image의 `event_idx` 직전까지의 event를
현재 target과 연결하는 방식이다. `frame_stride>1`에서는 건너뛴 interval을 다음 target에 합친다.
중첩 H5의 식별자는 EventHDR root 기준 상대 POSIX 경로다. manifest key, metric scene,
`sample_id`가 같은 규칙을 사용하므로 같은 basename이나 `.h5`/`.hdf5` stem도 충돌하지 않는다.

index 생성 시 모든 H5에서 검사:

- `events`, `images` group 존재
- xs/ys/ts/ps가 1차원 numeric이고 길이가 같음
- image array와 `event_idx`, `timestamp` 존재
- event index가 정수이며 `[0,event_count]`
- image event index와 timestamp가 비감소

sample 로드 시 해당 interval에서 검사:

- xs/ys/ts/ps 길이
- timestamp 유한·비감소
- 좌표 유한·target 해상도 범위
- polarity 유한·`-1/1` 또는 `0/1`

전체 event array를 dataset 초기화 때 RAM으로 읽지는 않는다. 모든 interval의 값까지 확인하려면
`inspect --validate-all`을 사용한다. HDF5 handle은 worker process별로 분리한다.

보유한 실제 `26.h5`에서 확인한 구조:

- 500 image sample
- 1,118,211 events
- image 240×320
- xs/ys/ts/ps 길이 일치
- image event index/timestamp 비감소
- 전체 500 sample strict load 통과

이 한 파일의 통과는 공식 70개 H5 전체 검증을 대신하지 않는다.

## 7. EventAid-R loader와 검증

`R-*.zip`을 압축 해제하지 않고 읽는다.

```text
event/NNNNNN.txt        timestamp x y polarity
gt/NNNNNN_img.png
timestamps.txt
shape.txt
```

정렬은 `event_i -> gt_(i+1)`이며 기본 `target_offset=1`이다. `timestamps.txt`에서 `i-1`, `i`를
각각 interval의 t0/t1로 사용한다.

검사:

- event token 수가 4의 배수
- timestamp 유한·비감소
- coordinate 유한·PNG 범위
- polarity 유한·허용 값
- scene-level timestamps 엄격 증가
- `shape.txt`와 PNG 크기 일치

보유한 실제 `R-bear.zip`은 65 paired sample 전체 strict load를 통과했다. 전체 14장면 검증은
서버에서 `--validate-all`로 수행해야 한다.

manifest 표시 합계는 약 24.68GB다. downloader는 ZIP container 유효성을 확인하지만 공식
checksum이 없어 cryptographic integrity는 보장하지 않는다.

## 8. Causal graph

node 전처리:

```text
x_n = x/(W-1)
y_n = y/(H-1)
t_n = (t-t_first)/(t_last-t_first)
node_features = [2*x_n-1, 2*y_n-1, 2*t_n-1, polarity]
positions = [x_n,y_n,t_n]
```

각 event는 index상 과거의 최대 `causal_candidates`개만 후보로 연결한다.

```text
src=i, dst=i+offset
keep if ||delta_xy|| <= spatial_radius and delta_t <= temporal_radius
```

후보 offset 32개를 Python loop로 CUDA에 순차 제출하지 않고 `[offset,event]` bounded tensor로
vectorize했다. `max_events=8192`, candidates=32에서 후보 tensor는 O(NK)이며 O(N²) graph를 만들지
않는다. 모든 non-empty node에는 self edge가 있다. empty interval은 0 node/0 edge로 유지한다.

## 9. Encoder, raster, U-Net

`SplineMessageLayer` 이름과 달리 실제 B-spline basis는 아니다.

```text
gate_e = sigmoid(MLP(edge_attr_e))
message_e = Linear(node_src) * gate_e
aggregate_dst = incoming message mean
z = SelfLinear(node) + aggregate
h = ReLU(BatchNorm(z))
```

graph feature는 `ceil(H/downsample) × ceil(W/downsample)` grid cell별 평균으로 rasterize한다.
decoder는 residual two-level U-Net이며 bottleneck에 optional ConvGRU가 있다. 출력은 sigmoid 후 원
sensor size로 resize한다.

scene, sequence index 또는 shape가 불연속이면 recurrent state를 초기화한다. frame마다 state를
detach하므로 temporal context는 forward에 쓰지만 BPTT 길이는 1이다. 실질적으로 batch size 1이고
A6000의 VRAM을 충분히 활용하는 구조는 아니다.

## 10. ANN과 rate proxy

ANN 학습은 ReLU graph encoder를 사용한다. 현재 `snn` 경로는 graph layer preactivation을 다음과
같이 양자화한다.

```text
r = clamp(ReLU(z)/threshold, 0, 1)
spike_count = floor(r*T)
rate_output = spike_count*threshold/T
```

없는 요소:

- timestep loop와 timestep별 graph propagation
- membrane, leak, reset, refractory state
- spike tensor 전달
- surrogate-gradient 학습

input projection, edge MLP, aggregation, raster, U-Net, ConvGRU는 analog다. 따라서 `T` 변화는 rate
양자화 해상도이지 neuromorphic 실행 시간이나 에너지 모델이 아니다.

보정 절차:

1. ANN checkpoint load와 eval mode
2. encoder input/layer BatchNorm folding
3. layer/channel threshold reset
4. EventHDR train의 여러 파일과 시간대에서 activation maxima 누적
5. 평가 전용 clean `best_snn.pt` 저장

SNN evaluate/benchmark guard:

- `simulation_steps >= 1`
- checkpoint `batch_norm_folded == true`
- `snn_calibration_samples >= 1`

보정 checkpoint에는 optimizer, scaler, RNG, history, training config가 없으며 resume용이 아니다.

## 11. 균형 sampling과 checkpoint 선택

validation의 `_balanced_contiguous_indices()` 동작:

1. EventHDR `path`, EventAid-R `scene`으로 group
2. sample budget을 group별 round-robin 할당
3. group별 deterministic contiguous window 선택
4. group 수보다 budget이 작으면 validation을 중단
5. global index 정렬로 scene 내부 시간 순서 유지

sampling 구분:

- recurrent validation: 위 연속 window 채점 전 같은 group predecessor를 기본 최대 64개 unscored
  replay; non-recurrent validation은 context replay 없음
- calibration: group-balanced time-spread `linspace`
- recurrent benchmark: group-balanced contiguous window + group당 기본 최대 32개 unmeasured predecessor
- non-recurrent benchmark: group-balanced time-spread

`hdr_train`의 `max_val_samples=500`, `hdr_smoke`의 `max_val_samples=32`는 scored frame 수다. context
frame은 이 budget 밖에서 forward되며 smoke는 group당 최대 8개를 사용한다.

validation 결과는 `micro`, `macro`, `per_scene`, `frames`를 저장한다. `best.pt` 선택 score는
`macro.ssim`이다. checkpoint에는 선택한 frame identity, group 전체 길이, source 상대경로·크기와
train/validation 원본 전체 SHA-256을 포함한 validation protocol을 한 번 남긴다. absolute root와
mtime은 protocol에서 비교하지 않아 상대 파일 identity와 byte가 같은 데이터를 다른 mount로 옮겨도
resume할 수 있다. 같은 경로의 resume은 `.data_hash_cache.json`에서 size/mtime/ctime이 모두 같은
파일의 기존 full hash를 재사용한다. 원본을 교체·복원했거나 전수 확인하려면
`train.rehash_data=true`로 cache를 무시한다.

## 12. 학습과 resume

기본 `configs/hdr_train.json`:

| 항목 | 값 |
|---|---:|
| epochs | 40 |
| batch | 1 |
| workers | 4 |
| crop | 256×256 |
| max events | 8192 |
| hidden/layers | 64/6 |
| candidates | 32 |
| decoder channels | 48 |
| AdamW LR | 2e-4 |
| AMP | CUDA에서 true |
| scored validation samples | 최대 500 |
| validation context | group당 최대 64 frame |

loss:

```text
1.0 Charbonnier
+ 0.2 (1-SSIM)
+ 0.1 spatial gradient L1
+ 0.2 temporal difference L1
```

`last.pt`에는 model, optimizer, scaler, epoch, best macro SSIM, history, config, validation protocol,
Python/NumPy/Torch/CUDA RNG가 들어간다. atomic replace를 사용한다. resume은 epoch 경계에서 exact
optimizer/scaler/RNG를 복원하며 inference-only checkpoint는 거부한다. 현재 config·data fingerprint와
historical `best.pt`의 protocol/model/score가 모두 일치해야 한다. mid-epoch data position과 recurrent
state는 복원하지 않는다.

`best.pt`는 `ann_inference` checkpoint로 model과 평가/선택 metadata만 저장하며 optimizer, scaler,
RNG, history, training config는 넣지 않는다. `evaluate()`와 `benchmark()`는 모든 checkpoint를 먼저
CPU에 읽고 model tensor만 GPU로 옮긴다.

첫 validation 전에 preemption되어 best score가 아직 `-inf`인 경우에는 `last.pt`만으로 resume할 수
있다. 유한 best score가 생긴 뒤에는 같은 run의 `best.pt`가 반드시 있어야 한다. epoch history에는
sampling count만 저장해 frame identity 목록을 매 epoch 반복하지 않는다.
fresh run은 기존 `last.pt`, `best.pt`, `history.json`, `config.json`이 있는 run directory를 덮어쓰지
않고 중단한다.

각 epoch record에는 CUDA peak allocated/reserved MiB가 들어간다. CPU에서는 null이다.

## 13. 평가와 metric

frame quality:

- PSNR, data range 1
- RMSE
- Gaussian SSIM: 기본 11×11, sigma 1.5, valid window
- 작은 image는 fitting odd window
- optional LPIPS
- 같은 scene·shape의 연속 sequence frame만 사용하는 no-flow `temporal_l1`

`temporal_l1`은 `L1((pred_t-pred_t-1),(target_t-target_t-1))`이다. 첫 frame과 sequence gap 뒤 첫
frame은 CSV null이고 집계에서 제외된다. Metric accumulator는 metric별 유효 frame 수를 저장한다.

SSIM은 이전 box average/zero-padding 구현에서 Gaussian valid window로 바뀌었다. 그래도 논문
공식 수치와 비교할 때 border/crop/color/tone/package가 같은지는 별도 확인해야 한다.

평가 산출물:

```text
eval.output_dir/
├── metrics.json
├── frames.csv
└── predictions/*_pred.png, *_gt.png
```

`evaluate()` latency는 sample load와 device move 뒤부터 metric 전까지이며 graph+model forward를
포함한다. cold first frame도 포함한다. `benchmark()`는 I/O/device move를 timer 밖에 두고 CUDA
Event 또는 CPU perf counter로 warmup 이후를 측정한다. recurrent window 앞 최대 32 predecessor는
timer 밖에서 forward해 state를 예열한다. unmeasured frame은 frame별 CUDA synchronize도 하지 않는다.
`warmup`은 device/kernel warmup이며 측정 state로 재사용하지 않는다.

지연 결과:

- mean, p50, p90, p95, p99, max, FPS
- events/s, mean node/edge
- RTF, deadline miss ratio
- rate-proxy firing rate
- peak allocated/reserved GPU memory
- sampling identity/fingerprint, state reset 수·비율과 `io_excluded=true`
- device warmup frame 수와 recurrent context frame 수

## 14. Config 역할

| config | 역할 | output |
|---|---|---|
| `configs/hdr_smoke.json` | provisional split real-data 1 epoch | `runs/smoke` |
| `configs/hdr_train.json` | final split ANN train/calibration | `runs/eventhdr_asgcn` |
| `configs/hdr_ann.json` | EventHDR ANN eval | `runs/eventhdr_official_eval_ann` |
| `configs/hdr_snn.json` | EventHDR rate eval | `runs/eventhdr_official_eval_snn` |
| `configs/aid_ann.json` | EventAid-R ANN eval | `runs/eventaid_r_external_ann` |
| `configs/aid_snn.json` | EventAid-R rate eval | `runs/eventaid_r_external_snn` |

EventAid configs는 이제 EventHDR training과 같은 log/mu target domain을 사용한다.

`eval.batch_size`는 현재 loader가 1로 고정되어 있어 미사용이다. train
`output.save_predictions`도 현재 학습 loop에서는 미사용이다.

## 15. 환경 재현성

`constraints/py312.txt`는 현재 검증한 exact public versions를 고정한다.

- Python 3.12 profile
- torch 2.13.0 public version
- NumPy, h5py, Pillow, tqdm
- torch/dev transitive packages
- pytest, Ruff

CUDA local build(`+cuXXX`)은 서버 driver에 따라 공식 index에서 선택한다. `check_env.py`는 Python,
torch build string, torch CUDA runtime, cuDNN, GPU 이름/메모리, constraint public version 일치,
data/runs 별 free space를 출력한다. 이는 hash 기반 cross-platform lock은 아니다. CUDA runtime
packages와 LPIPS/torchvision을 포함한 환경은 서버별로 별도 freeze가 필요하다.

`setup.sh`는 command environment를 `.env`보다 우선하며, 기존 venv의 Python도 재검사한다.
`requirements.txt`와 Dockerfile은 py312 constraints를 사용한다. Linux CI의 Python 3.12 lint job도
같은 constraints를 설치하고 `check_env --lock`을 실행한다. Python 3.10/3.11 호환성 matrix는
범위 의존성으로 별도 검사한다.

## 16. 서버 실행 경로

README가 복사 가능한 전체 순서의 기준 문서다.

- 직접 서버: tmux에서 `scripts/train.sh`, `scripts/eval.sh`
- interactive allocation: `ssai_agpu -g=1` 후 동일 Bash wrapper
- SLURM: `server/train.sbatch`, `server/eval.sbatch`
- PBS: `server/train.pbs`, `server/eval.pbs`
- Docker: Python 3.12 + constraints, data read-only/run writable volume

PBS script는 `PROJECT_ROOT` 또는 제출 당시 root를 검증하고 잘못된 디렉터리면 중단한다. PBS
resource key `select/ngpus`, queue와 account는 클러스터마다 달라 template header 수정이 필요하다.
`.pbs`는 Git에서 LF checkout을 강제한다.

## 17. 테스트와 확인 상태

현재 로컬 확인:

- Python 3.12.13
- torch 2.13.0+cpu
- `pytest`: 81 tests passed
- Ruff: 통과
- Python compileall: 통과
- JSON parse와 `git diff --check`: 통과
- `pip check`: 통과
- `check_env --lock constraints/py312.txt`: constraint public versions 일치
- 실제 `26.h5`: 500/500 strict sample load
- 실제 `R-bear.zip`: 65/65 strict sample load

테스트가 다루는 주요 항목:

- H5/ZIP mapping, stride, next-frame target
- missing/malformed group, attr, array, timestamp, coordinate, polarity
- multiprocess handle 분리
- empty event, causal edge, vectorized graph reference equivalence
- model forward/backward, BN folding, checkpoint reload
- Gaussian SSIM reference와 small image gradient
- variable metric accumulator와 temporal metric
- file-balanced contiguous/time-spread sampling, sequence/shape reset, macro selection
- provisional split guard
- SNN step/calibration guard와 clean checkpoint
- training resume state와 validation/best checkpoint protocol 일치 검사
- CPU checkpoint load, path-independent full-file digest, pre-validation resume
- evaluate/benchmark end-to-end contract
- inspect `--validate-all`

GitHub CI는 Ubuntu Ruff/locked install/shell syntax와 Ubuntu·Windows Python 3.10–3.12 pytest matrix를
구성한다. 이 handoff 갱신 후 push된 workflow의 실제 성공 여부는 GitHub Actions run으로 별도
확인해야 한다.

아직 검증하지 못한 항목:

- 공식 EventHDR 51+19개와 EventAid-R 14장면 전체
- 물리 scene-disjoint split
- CUDA AMP, CUDA Event, A100/A6000 peak memory와 OOM
- 40 epoch 수렴, 품질, 실제 walltime
- LPIPS/torchvision locked CUDA install과 offline weights
- PBS/SLURM resource directive의 학교 scheduler 승인
- Docker GPU runtime 실실행

## 18. 외부 입력이 있어야 해결되는 blocker

### P0

1. EventHDR 파일→물리 scene 대응표를 확보한다.
2. 대응표로 train/val group을 다시 쓰고 manifest를 `final`로 바꾼다.
3. 서버 driver에 맞는 torch 2.13.0 official CUDA index를 확정한다.
4. A100 10GB allocation에서 `hdr_smoke.json`과 peak memory를 기록한다.
5. A6000에서 step time을 측정해 40 epoch walltime을 산정한다.
6. 전체 데이터를 `inspect --validate-all`로 통과시킨다.

코드가 자동으로 추측해서는 안 되는 것은 1–3이다. 실제 GPU가 없는 로컬 환경에서 4–5 결과를
만들어내면 안 된다.

### 후속 연구

1. 정확한 B-spline ASGCN baseline
2. real IF/LIF timestep graph propagation
3. multi-scene batching 또는 더 큰 BPTT
4. LPIPS와 dataset 공식 temporal metric reproduction
5. event density/count channel ablation
6. compression/transport/memory traffic model
7. FPGA/ASIC mapping과 latency/power/energy measurement

## 19. 운영·보안 주의

- `.env`는 Bash가 source하므로 신뢰하는 설정만 넣고 Git에 올리지 않는다.
- token, SSH private key, storage password를 config, scheduler script, log에 넣지 않는다.
- `sbatch --export=ALL` 전 credential environment를 점검한다.
- PBS는 필요한 변수만 `qsub -v`로 전달한다.
- 데이터 license를 확인하고 H5/ZIP을 private GitHub에도 올리지 않는다.
- EventAid-R은 ZIP 직접 사용으로 약 25GB 추출본 중복을 피한다.
- OOM이면 무작정 A6000 VRAM을 채우지 말고 `max_events`, candidates, decoder width의 scientific
  trade-off를 기록한다.
- benchmark는 I/O 제외 compute latency이므로 전체 시스템 전송 latency와 구분한다.
- rate-proxy GPU latency를 SNN hardware energy/latency로 주장하지 않는다.

## 20. 다른 ChatGPT 교차 검증 요청문

```text
hand_off.md는 설계와 검증 상태이고 code_summary.md는 같은 작업 상태의 프로젝트 text 원문이다.
설명을 믿지 말고 반드시 실제 코드와 config를 대조하라.

1. EventHDR/EventAid-R frame-event alignment와 strict validation이 공식 format에 맞는지 본다.
2. validation/benchmark의 연속 window와 calibration time-spread가 각 용도에 맞고 state gap이 없는지 본다.
3. macro SSIM checkpoint selection, resume, temporal metric, latency timer를 검증한다.
4. ANN checkpoint 또는 0-step/0-sample 상태가 SNN path로 들어갈 수 있는지 공격적으로 테스트한다.
5. EventHDR와 EventAid target transform이 같은지, 그래도 남는 radiometric domain gap을 구분한다.
6. Gaussian SSIM을 독립 구현과 비교하고 논문 공식 metric과 같다고 과장한 문구가 없는지 본다.
7. Python/torch/CUDA constraints와 MobaXterm, PBS, SLURM, Docker 명령의 portability를 본다.
8. ASGCN-inspired encoder와 rate proxy를 실제 ASGCN/SNN/반도체 구현으로 잘못 주장한 곳을 찾는다.
9. 테스트 통과와 공식 전체 데이터·GPU 실측을 명확히 구분한다.

결과는 P0/P1/P2로 나누고 파일, 함수, 재현 절차와 최소 수정안을 제시하라.
```

## 21. 삭제·단축 이력

삭제:

- 원문 HWP
- production random-weight smoke/dummy entrypoint
- generated smoke data와 build/cache artifact

짧게 유지한 주요 이름:

| 파일 | 역할 |
|---|---|
| `scripts/setup.sh` | server install |
| `scripts/check_env.py` | environment/data readiness |
| `scripts/get_aid.sh` | EventAid-R download |
| `scripts/train.sh` | train wrapper |
| `scripts/eval.sh` | evaluate+benchmark wrapper |
| `configs/hdr_train.json` | full train |
| `configs/hdr_smoke.json` | real-data smoke |
| `configs/hdr_ann.json`, `hdr_snn.json` | EventHDR eval |
| `configs/aid_ann.json`, `aid_snn.json` | EventAid-R eval |
| `server/train.sbatch`, `eval.sbatch` | SLURM |
| `server/train.pbs`, `eval.pbs` | PBS/Torque |
