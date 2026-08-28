# ASGCN Event-to-Frame 프로젝트 인수인계

## 0. 문서 기준과 읽는 순서

- 구현 기준 커밋: `8c935001b0c71ce0ed7966ef0c6c1622a3fadb65`
- 기준 브랜치: `main`
- 비공개 저장소: `costunder/asgcn-event-reconstruction`
- 작성일: 2026-08-29 (Asia/Seoul)
- 함께 전달할 파일: `code_summary.md`

`hand_off.md`는 구조와 의도, 실행 방법, 현재 한계를 설명한다. `code_summary.md`는 기준
커밋에서 Git이 추적하던 텍스트 파일 49개의 원문 전체를 `# 상대경로` 형식으로 묶은
스냅샷이다. 다른 ChatGPT에 교차 검증을 맡길 때 두 파일을 함께 전달해야 설명과 실제 코드가
일치하는지 대조할 수 있다.

실제 H5/ZIP 데이터, `.venv`, checkpoint, 로그, 실행 결과는 두 문서에 포함되지 않는다.
과제 원문 HWP는 사용자의 요청으로 작업 폴더에서 삭제됐고 Git에 올라간 적도 없으므로, 현재
저장소만으로는 HWP 문구와 구현을 직접 대조할 수 없다.

## 1. 프로젝트 목표와 현재 수준

목표는 EventHDR의 실제 이벤트 스트림으로 HDR luminance frame을 복원하는 모델을 학습하고,
EventHDR 공식 eval과 EventAid-R에서 품질·지연시간·외부 일반화를 측정하는 것이다.

현재 구현의 정확한 명칭은 다음과 같다.

> ASGCN-style edge-conditioned graph encoder와 residual U-Net/ConvGRU decoder를 결합한
> event-to-frame 소프트웨어 연구 프로토타입

데이터 역할은 분리한다.

| 단계 | 데이터 | 허용 용도 |
|---|---|---|
| ANN 학습 | EventHDR train | weight 최적화와 crop |
| 모델 선택 | EventHDR train holdout | validation SSIM과 `best.pt` 선택 |
| SNN 보정 | EventHDR train | BN folding과 activation threshold 계산 |
| 내부 최종시험 | EventHDR 공식 eval | 학습 완료 후 1회 품질·지연 평가 |
| 외부 잠금시험 | EventAid-R | 학습·BN·threshold 선택 없이 일반화 평가 |

중요하게도 현재 저장소는 다음을 아직 구현하지 않는다.

- 원 논문의 정확한 B-spline/SplineConv ASGCN 재현
- timestep별 membrane을 갖는 완전한 IF/LIF SNN
- 이벤트 압축·전송 protocol 또는 network stack
- RTL, FPGA, ASIC, NPU kernel, DMA, on-chip buffer
- 전력·에너지·면적·공정 기반 반도체 결과

따라서 현 단계는 과제 전체의 **알고리즘/소프트웨어 실험 기반**이지, 저지연·고효율 반도체
통합 처리 시스템이 완성된 상태는 아니다.

## 2. 전체 데이터·모델 파이프라인

```text
EventHDR H5 / EventAid-R ZIP
  -> dataset별 frame-event 정렬
  -> sample {events[N,4], target[C,H,W], sensor_size, metadata}
  -> 좌표·시간·polarity 정규화
  -> bounded causal event graph
  -> ANN 또는 post-training rate-SNN graph encoder
  -> graph feature를 2D grid cell별 평균 rasterization
  -> residual U-Net + analog ConvGRU
  -> sigmoid luminance frame [1,C,H,W]
  -> 학습 loss 또는 PSNR/SSIM/RMSE/latency 평가
```

핵심 진입점은 `src/asgcn_recon/model.py`의
`ASGCNReconstructor.forward_sample()`이다. CLI는
`src/asgcn_recon/cli.py`, 학습·평가 orchestration은 `src/asgcn_recon/engine.py`가 담당한다.

## 3. 저장소 구조와 파일 책임

```text
asgcn-event-reconstruction/
├── configs/                    # 학습/평가 JSON
├── docs/                       # 서버·실험 문서
├── manifests/                  # EventHDR split, EventAid-R URL/크기
├── scripts/                    # 설치, 환경 진단, 다운로드, train/eval wrapper
├── server/                     # Slurm job
├── src/asgcn_recon/
│   ├── cli.py                  # inspect/train/evaluate/benchmark/calibrate CLI
│   ├── data/                   # H5·ZIP loader와 sample contract
│   ├── graph.py                # causal graph와 ASGCN-style encoder
│   ├── model.py                # rasterization, U-Net, ConvGRU
│   ├── engine.py               # train/validation/eval/benchmark/calibration
│   ├── losses.py               # reconstruction loss
│   ├── metrics.py              # PSNR/SSIM/RMSE와 집계
│   └── utils.py                # 경로, seed, 저장 함수
├── tests/                      # 설치 패키지와 분리된 fixture·테스트
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── README.md
```

`data/`, `runs/`, `logs/`, H5/ZIP/PT/HWP는 `.gitignore` 대상이다. 테스트 fixture는
`tests/`에만 있고 setuptools는 `src/`만 package로 찾으며 Docker image도 `tests/`를 복사하지
않는다.

## 4. 공통 sample 계약

두 loader는 최종적으로 다음 dictionary를 반환한다.

| 필드 | 형식 | 의미 |
|---|---|---|
| `events` | `float32 [N,4]` | 열 순서 `[x,y,t,p]` |
| `target` | `float32 [C,H,W]` | `[0,1]` 복원 정답 |
| `sample_id` | 문자열 | `<scene>/<frame>` 식별자 |
| `sensor_size` | `(H,W)` | crop 이후 출력 크기 |
| `metadata` | dictionary | dataset, scene, source, timestamp, `dt_us` 등 |

공통 처리 규칙은 `src/asgcn_recon/data/common.py`에 있다.

- polarity가 `0/1`이면 `-1/+1`로 바꾼다.
- 이벤트 수가 `max_events`보다 많으면 시간 범위를 유지하도록 등간격 index로 subsample한다.
- crop 밖 이벤트를 제거하고 남은 좌표를 crop 원점 기준으로 이동한다.
- 정수 영상은 dtype 최대값으로 나눈다.
- 범위 밖 float 영상은 frame별 0.1/99.9 percentile로 정규화한다.
- RGB를 1채널로 만들 때 BT.709 계수 `0.2126/0.7152/0.0722`를 사용한다.
- EventHDR 기본 target에는 `log1p(mu*x)/log1p(mu)`, `mu=5000` tone mapping을 적용한다.

## 5. EventHDR loader

구현 파일은 `src/asgcn_recon/data/eventhdr.py`다.

기대 HDF5 구조:

```text
events/xs
events/ys
events/ts
events/ps
images/imageXXXXXXXXX
  attrs.event_idx
  attrs.timestamp
```

각 target image의 `event_idx`를 끝점으로 하고 이전에 선택한 image의 끝점부터 현재 끝점까지를
입력 이벤트 interval로 사용한다. `frame_stride>1`이면 건너뛴 interval을 다음 선택 frame에
합쳐 이벤트를 버리지 않는다. 파일 handle은 process ID별로 다시 열어 DataLoader fork에서
HDF5 descriptor가 공유되지 않도록 한다.

H5 파일과 image key는 문자열 정렬이다. 숫자 파일명의 자연수 정렬은 아니므로 `1.h5`,
`10.h5`, `2.h5` 순서가 될 수 있다. 다만 파일이 바뀌면 `metadata.scene`도 바뀌어 recurrent
state를 초기화하므로 현재 파일 간 state가 섞이지는 않는다. Manifest filtering은 basename
기준이므로 서로 다른 하위 폴더에 같은 basename이 중복되는 배치는 피해야 한다.

현재 split manifest `manifests/eventhdr_split.json`은 다음 임시 규칙이다.

- train: `1.h5`부터 `47.h5`
- validation: `48.h5`부터 `51.h5`
- 공식 eval: 별도 `data/EventHDR/eval/*.h5`

이 split은 물리 장면 대응표를 반영하지 않은 파일 단위 임시 분할이다. 동일 장면이 train과
validation 양쪽에 들어가면 leakage가 발생할 수 있으므로 결과 확정 전에 group split으로
교체해야 한다.

또한 random crop RNG는 매 `__getitem__` 호출마다 `seed + path CRC`로 다시 만들어진다. 같은
파일은 frame·epoch가 바뀌어도 crop 위치가 고정되어 temporal consistency는 유지되지만,
epoch별 crop augmentation 다양성은 없다.

## 6. EventAid-R loader

구현 파일은 `src/asgcn_recon/data/eventaid_r.py`다. `R-*.zip`을 압축 해제하지 않고 직접 읽는다.

ZIP 내부 기대 구조:

```text
event/NNNNNN.txt        # timestamp x y polarity
gt/NNNNNN_img.png
timestamps.txt
shape.txt
```

핵심 정렬은 `event_i -> gt_(i+1)`이다. 즉 기본 `target_offset=1`이며 동일 번호 PNG를 target으로
쓰면 off-by-one 오류다. `timestamps.txt`의 `i-1`, `i` 값을 각각 `t0_us`, `t1_us`로 사용해
`dt_us`를 계산한다. `shape.txt`가 있으면 실제 PNG 크기와 일치하는지 검사한다.

`manifests/eventaid_r.json`은 14개 장면, 표시 합계 `24.68024GB`, 공개 다운로드 URL과 표시
크기를 담는다. checksum은 없으므로 Bash downloader가 확인하는 것은 ZIP container 유효성뿐
이며 파일 내용의 cryptographic integrity까지 보장하지는 않는다.

이벤트 txt token 수가 4의 배수가 아니면 `_read_events()`는 마지막 불완전 token을 조용히
버린다. timestamp와 좌표 범위도 loader에서 정렬·검사하지 않으므로 official data contract
검증을 별도로 추가해야 한다.

## 7. 이벤트 graph 생성

`src/asgcn_recon/graph.py`의 `prepare_event_nodes()`는 다음을 만든다.

```text
x_n = x/(W-1)
y_n = y/(H-1)
t_n = (t-t_first)/(t_last-t_first)
p   = {-1,+1}

node_features = [2*x_n-1, 2*y_n-1, 2*t_n-1, p]  # [N,4]
positions     = [x_n, y_n, t_n]                   # [N,3]
```

`build_causal_graph()`는 모든 `N^2` 쌍을 만들지 않는다. 이벤트 index 순서상 과거에 있는 최대
`causal_candidates`개만 검사한다. 기본값은 32다.

```text
src = i
dst = i + offset
delta = position(dst) - position(src)
edge 유지 조건:
  norm(delta_xy) <= spatial_radius
  delta_t <= temporal_radius
```

각 non-empty node에는 self-edge도 붙는다. `edge_attr`는 `[dx,dy,dt,||delta||]`다. 시간과 공간
radius는 실제 microsecond/pixel 단위가 아니라 sample별 정규화 좌표 기준이다. 또한 input
timestamp가 정렬됐다는 가정이 있지만 단조성 검사는 하지 않는다.

빈 이벤트 interval에는 가짜 중앙 node를 넣지 않는다. `[0,4]` node feature와 `[0,3]`
position, 0개 edge를 유지하고 이후 zero raster를 만든다. decoder bias와 기존 ConvGRU state는
계속 작동하므로 빈 interval의 출력이 반드시 0이거나 state가 그대로 유지되는 정책은 아니다.

## 8. ASGCN-style encoder

`SplineMessageLayer`라는 이름을 사용하지만 실제 B-spline basis나 PyG `SplineConv`는 아니다.
구현은 edge-conditioned MLP gate다.

```text
gate_e      = sigmoid(MLP(edge_attr_e))
message_e   = Linear(node_src) * gate_e
aggregate_j = incoming message의 평균
z_j         = SelfLinear(node_j) + aggregate_j
h_j         = ReLU(BatchNorm(z_j))
```

초기 입력은 `Linear(4,hidden_dim) -> BatchNorm1d -> ReLU`이고 위 message layer를 설정한 횟수만큼
통과한다. 따라서 논문이나 발표에서는 **원 ASGCN과 동일 구현**이 아니라
**ASGCN-style edge-conditioned encoder**라고 표현해야 한다.

## 9. ANN과 현재 SNN 경로

학습은 항상 ANN 경로다. surrogate-gradient SNN 학습은 없다.

현재 SNN은 각 graph layer의 ANN preactivation을 closed-form rate로 양자화한다.

```text
r = clamp(ReLU(z)/threshold, 0, 1)
spike_count = floor(r * simulation_steps)
rate_output = spike_count * threshold / simulation_steps
```

다음 요소는 여전히 analog PyTorch 연산이다.

- input projection의 Linear/BN/ReLU
- edge MLP와 message aggregation
- rasterization
- U-Net과 ConvGRU decoder 전체

실제 spike tensor, membrane potential, leak, reset, refractory state가 없고 graph layer를
`simulation_steps`번 반복하지 않는다. 그러므로 현재 SNN latency가 실제 timestep 수에 비례하는
neuromorphic 비용을 나타낸다고 해석하면 안 된다. 정확한 표현은 **post-training graph activation
rate approximation을 사용하는 hybrid ANN-SNN**이다.

### BN folding과 threshold calibration

`asgcn-recon calibrate`는 EventHDR train만 사용한다.

1. ANN checkpoint를 `eval()`로 로드한다.
2. encoder input BN과 각 message layer BN을 Linear weight/bias에 folding한다.
3. threshold를 `1e-6`으로 초기화한다.
4. calibration sample에서 layer/channel별 ReLU 최대값을 누적한다.
5. folded weight와 threshold를 `best_snn.pt`에 저장한다.

fold 여부는 persistent buffer로 state dict에 저장된다. 그러나 `evaluate --inference-mode snn`은
checkpoint가 실제로 calibrated됐는지 강제 검사하지 않는다. ANN checkpoint를 잘못 넘겨도
threshold 1.0과 미fold BN으로 실행될 수 있으므로 `batch_norm_folded`와
`snn_calibration_samples` 검증을 추가하는 것이 좋다.

engine의 calibration guard는 `dataset.type == eventhdr`만 확인한다. 임의 config가 EventHDR
공식 eval root를 가리켜도 막지 못하므로 checked-in `configs/hdr_train.json`을 사용해야 한다.
또한 calibrated checkpoint에는 기존 optimizer/scaler도 남아 있어 이를 training resume으로
잘못 쓰는 것을 차단하지 않는다. `best_snn.pt`는 평가 전용으로 취급해야 한다.

`benchmark()`는 SNN에서 `simulation_steps>=1`을 검사하지만 `evaluate()`는 동일한 명시 검사가
없다.

## 10. Rasterization과 decoder

graph feature는 `ceil(H/downsample) x ceil(W/downsample)` grid로 배치되고 같은 cell의 feature를
평균한다. 기본 raster shape은 `[1,64,ceil(H/4),ceil(W/4)]`이다. 별도 event-count channel이
없으므로 cell에 몇 개 이벤트가 있었는지는 명시적으로 보존되지 않는다.

decoder는 다음 analog 구조다.

```text
stem Conv3x3
  -> ResidualBlock(B)
  -> stride-2 Conv + ResidualBlock(2B)
  -> stride-2 Conv + 2*ResidualBlock(4B)
  -> optional ConvGRU(4B)
  -> bilinear upsample + skip + ResidualBlock(2B)
  -> bilinear upsample + skip + ResidualBlock(B)
  -> Conv3x3 + sigmoid
  -> 원 sensor_size로 bilinear resize
```

ResidualBlock은 Conv-GroupNorm-SiLU-Conv-GroupNorm에 residual을 더한다. ConvGRU는 bottleneck에서
동작하며 scene이 바뀌거나 spatial shape이 달라지면 zero state로 초기화된다.

train/validation/evaluate/benchmark는 매 frame 후 recurrent state를 detach한다. 즉 이전 frame
정보는 forward에 쓰지만 sequence 전체 BPTT는 하지 않는 길이 1 truncated-BPTT다.

## 11. 학습 절차

기본 `configs/hdr_train.json`:

| 항목 | 값 |
|---|---:|
| epoch | 40 |
| batch size | 1 |
| train workers | 4 |
| max events | 8192 |
| crop | 256x256 |
| hidden / graph layers | 64 / 6 |
| learning rate | `2e-4` |
| weight decay | `1e-6` |
| grad clip | `1.0` |
| AMP | CUDA에서 활성화 |

학습 loader는 `shuffle=False`이며 scene이 바뀌면 recurrent state, 이전 prediction과 target을
초기화한다. 현재 engine loop는 batch 길이가 1이 아니면 recurrent 옵션과 관계없이 오류를
내므로 실질적으로 batch size 1 전용이다.

기본 loss:

```text
L = 1.0 * Charbonnier
  + 0.2 * (1 - SSIM)
  + 0.1 * gradient L1
  + 0.2 * temporal-difference L1

L_temporal = L1((pred_t-pred_(t-1)) - (gt_t-gt_(t-1)))
```

이전 prediction/target과 recurrent state를 detach하므로 temporal gradient는 현재 frame에만
흐른다. optimizer는 AdamW다. validation SSIM이 개선되면 `best.pt`, 매 epoch 끝에는 `last.pt`를
atomic replace 방식으로 저장한다.

`last.pt`에는 model, optimizer, AMP scaler, epoch, best SSIM, history, Python/NumPy/Torch/CUDA
RNG가 들어 있다. resume은 이 상태를 복원하며 optimizer가 없는 inference checkpoint는 정확한
resume 용도로 거부한다.

`configs/hdr_train.json`의 `output.save_predictions`는 현재 train engine에서 읽지 않는 미사용
필드다.

기본 validation은 파일/scene 균형 sampling이 아니라 정렬된 validation dataset의 앞쪽 최대
500 sample만 사용한다. 모델 선택 결과를 확정하기 전에 전체 validation 또는 scene-balanced
평가와 비교해야 한다. Resume은 epoch 경계만 지원하며 mid-epoch sample 위치, recurrent state,
이전 prediction/target은 저장하지 않는다.

## 12. 평가·benchmark와 산출물

`evaluate()`는 scene 경계에서 state를 초기화하고 frame별로 다음을 계산한다.

- PSNR
- SSIM
- RMSE
- config에서 `eval.lpips=true`일 때만 LPIPS
- model forward latency
- `rtf = latency_ms / frame_interval_ms`
- event/node/edge 수

SSIM은 Gaussian-window 표준 package가 아니라 `avg_pool2d` box window 기반 자체 구현이다. 논문
또는 공식 benchmark의 SSIM과 값이 다를 수 있다. 품질은 전체 frame 평균 micro, scene 평균
macro와 per-scene으로 나뉜다.

`evaluate()`의 latency timer는 dataset I/O와 host-to-device 이동 뒤에 시작해 graph 생성과 model
forward를 포함한다. 별도 warmup이 없어 첫 frame cold-start도 집계된다. metric 계산과 PNG
저장은 timer 밖이다.

평가 결과:

```text
<eval.output_dir>/
├── metrics.json
├── frames.csv
└── predictions/
    ├── *_pred.png
    └── *_gt.png
```

`benchmark()`는 dataset read와 host-to-device 이동을 timer 밖에 두고 model compute만 측정한다.
CUDA에서는 `torch.cuda.Event`, CPU에서는 `perf_counter`를 쓴다. warmup 후 mean, p50/p90/p95/p99,
max, FPS, events/s, mean node/edge, firing rate, RTF, deadline miss ratio, GPU peak memory를
반환한다. benchmark JSON은 파일이 아니라 stdout으로 출력되므로 `tee`로 보존해야 한다.

평가 config의 `eval.batch_size`는 현재 `evaluate()`가 loader batch를 1로 고정하므로 사실상
사용되지 않는다. checked-in 값도 1이라 결과에는 영향이 없지만 config schema 정리 대상이다.

## 13. CLI와 config 역할

설치 후 entrypoint는 `asgcn-recon`이며 다음 subcommand가 있다.

| 명령 | 역할 |
|---|---|
| `inspect` | dataset 구조·sample·manifest train/val 검사 |
| `train` | EventHDR ANN 학습 또는 resume |
| `evaluate` | checkpoint 품질·frame latency 평가 |
| `benchmark` | I/O 제외 반복 model latency |
| `calibrate` | EventHDR train 기반 BN folding·SNN threshold 보정 |

config 경로는 shell의 현재 폴더가 아니라 `pyproject.toml`이 있는 checkout root에 고정해 해석한다.
외부 config에 checkout root가 없으면 해당 config 파일 폴더를 기준으로 삼는다.

| config | 용도 | output |
|---|---|---|
| `configs/hdr_train.json` | EventHDR 학습/검증/calibration | `runs/eventhdr_asgcn` |
| `configs/hdr_ann.json` | EventHDR 공식 ANN eval | `runs/eventhdr_official_eval_ann` |
| `configs/hdr_snn.json` | EventHDR 공식 SNN eval | `runs/eventhdr_official_eval_snn` |
| `configs/aid_ann.json` | EventAid-R 외부 ANN eval | `runs/eventaid_r_external_ann` |
| `configs/aid_snn.json` | EventAid-R 외부 SNN eval | `runs/eventaid_r_external_snn` |

## 14. MobaXterm/Linux 서버 실행

MobaXterm은 SSH/SFTP client이며 아래 명령은 접속한 Linux 서버에서 실행한다. private repo는
서버 SSH key를 GitHub에 등록하는 방식을 권장한다.

```bash
git clone git@github.com:costunder/asgcn-event-reconstruction.git
cd asgcn-event-reconstruction

read -r -p "PyTorch wheel index URL (Enter=PyPI default): " TORCH_INDEX_URL
export TORCH_INDEX_URL
export PROJECT_EXTRAS="dev,eval"
export REQUIRE_CUDA=0

bash scripts/setup.sh
source .venv/bin/activate
python scripts/check_env.py
python -m pytest -q
```

실제 GPU node/allocation에서는 다음을 통과해야 한다.

```bash
python scripts/check_env.py --require-cuda
```

`setup.sh`는 Python 3.10 이상 확인, `.venv`, 선택한 Torch wheel, editable project, 데이터 폴더와
`runs/`를 준비한다. PyTorch wheel index와 version은 고정값이 아니므로 서버의 NVIDIA driver에
맞춰 공식 PyTorch selector에서 선택해야 한다.

## 15. 데이터 준비와 용량

| 데이터 | 예상 용량 |
|---|---:|
| EventHDR | 약 25.72GB |
| EventAid-R 14장면 | 24.68024GB |
| 합계 | 약 50.40GB |

가상환경·checkpoint·prediction·로그를 고려해 최소 70GB 여유 공간을 권장한다. ZIP을 직접 읽고
voxel/graph cache를 만들지 않으므로 100GB 안에서 운용하는 구성을 목표로 한다.

```text
data/
├── EventHDR/
│   ├── train/1.h5 ... 51.h5
│   └── eval/*.h5               # 공식 eval 19개 기대
└── EventAid-R/R-*.zip
```

EventHDR는 공식 배포처에서 직접 받아 MobaXterm SFTP 또는 shared storage symlink로 배치한다.
EventAid-R Bash downloader는 인자가 없으면 작은 `R-bear`만 받는다.

```bash
bash scripts/get_aid.sh R-bear
bash scripts/get_aid.sh --all
```

Bash downloader는 resume/retry/ZIP 검사를 제공한다. 반면 `scripts/get_aid.ps1`은 `-Scenes`를
생략하면 현재 전체 14장면을 선택하며, 기존 파일 skip·ZIP 검증도 Bash판보다 약하다. Windows에서
작은 샘플만 받을 때는 반드시 `-Scenes R-bear`를 명시한다.

데이터 검사:

```bash
asgcn-recon inspect --config configs/hdr_train.json --samples 2
asgcn-recon inspect --config configs/hdr_ann.json --samples 2
asgcn-recon inspect --config configs/aid_ann.json --samples 2
```

## 16. 표준 실험 명령

### ANN 학습

```bash
tmux new-session -s asgcn -c "$PWD" \
  "bash -lc 'source .venv/bin/activate && bash scripts/train.sh configs/hdr_train.json'"
```

재개:

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/train.sh configs/hdr_train.json
```

### ANN -> SNN calibration

```bash
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500
```

### 평가와 benchmark

```bash
INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/hdr_ann.json runs/eventhdr_asgcn/best.pt

INFERENCE_MODE=snn SIMULATION_STEPS=16 RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/hdr_snn.json runs/eventhdr_asgcn/best_snn.pt

INFERENCE_MODE=ann RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/aid_ann.json runs/eventhdr_asgcn/best.pt

INFERENCE_MODE=snn SIMULATION_STEPS=16 RUN_BENCHMARK=1 \
  bash scripts/eval.sh configs/aid_snn.json runs/eventhdr_asgcn/best_snn.pt
```

`eval.sh`는 CUDA·dataset·checkpoint를 먼저 검사한 뒤 evaluate와 benchmark를 연속 실행한다.
기본 benchmark는 warmup 10, 측정 100회다.

### Slurm

```bash
sbatch server/train.sbatch

sbatch --export=ALL,CONFIG_PATH=configs/aid_snn.json,\
CHECKPOINT_PATH=runs/eventhdr_asgcn/best_snn.pt,INFERENCE_MODE=snn,\
SIMULATION_STEPS=16 server/eval.sbatch
```

기본 요청은 GPU 1개, CPU 8개, RAM 32GB다. train 최대 2일, eval 최대 8시간이며 partition,
account, GPU 종류와 CUDA module은 클러스터 정책에 맞춰 수정한다.

### Docker

Docker image는 Python 3.12 slim과 eval extras를 사용하고 data/checkpoint/tests를 포함하지 않는다.
Compose는 data를 read-only, runs를 writable volume으로 mount한다.

```bash
docker build --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t asgcn-event-reconstruction .

docker compose run --rm experiment inspect --config configs/hdr_train.json
```

## 17. 검증 완료 상태

구현 기준 커밋에서 확인한 내용:

- Ruff 통과
- 로컬 pytest: `15 passed`
- test-only EventHDR H5/EventAid-R ZIP fixture 기반 forward/backward/evaluate/benchmark 통과
- 실제 로컬 `R-bear.zip`: 65 paired samples, target `705x1265` 로딩 확인
- 실제 로컬 EventHDR `26.h5`: 500 samples, target `240x320` 로딩 확인
- wheel에 `tests/`, 삭제된 `smoke.py`, random-weight verify가 들어가지 않음을 확인
- GitHub Actions CI #4 성공
  - Ubuntu Python 3.12 Ruff와 Bash/Slurm syntax
  - Ubuntu/Windows, Python 3.10/3.11/3.12 총 6개 pytest 조합

실제 데이터 두 파일은 Git ignored local sample이며 GitHub clone에 포함되지 않는다. CI 역시 실제
데이터, CUDA GPU, full-size 학습을 검사하지 않는다.

테스트 15개가 보장하는 범위:

- EventHDR loader, stride interval aggregation
- EventAid-R next-frame target 정렬
- causal edge 방향
- 0-node empty-event 처리
- model forward/backward
- BN folding, reload, rate-SNN path
- CPU autocast dtype
- checkout 기준 config 경로
- manifest 누락 오류와 train/val inspect
- optimizer/scaler/RNG를 포함한 resume
- benchmark 입력 검증
- HDF5/ZIP multiprocess 안전성
- fixture 기반 checkpoint/evaluate/benchmark end-to-end contract

현재 테스트가 보장하지 않는 범위:

- 공식 데이터 전체 schema와 40 epoch 수렴/품질
- CUDA AMP, CUDA Event, peak memory, 실제 GPU OOM
- LPIPS 설치·weight download와 offline 실행
- `calibrate()` 전체 CLI/checkpoint 저장 end-to-end
- epoch resume 전후 weight의 bitwise 동일성, mid-epoch resume
- CLI executable subprocess와 실제 Bash/PowerShell 동작
- `persistent_workers=true` lifecycle; multiprocess test는 false로 실행
- metric accumulator, percentile, RTF/deadline 계산의 독립 단위 테스트
- malformed H5/ZIP, corrupt CRC, 중복 basename 같은 오류 경로

## 18. 알려진 한계와 우선순위

### P0: 결과를 논문·과제 성과로 사용하기 전에 해결

1. EventHDR를 물리 scene 단위 train/validation split으로 다시 구성한다.
2. EventHDR `event_idx -> image` 정렬을 공식 schema와 전체 파일에서 재검증한다.
3. EventAid-R `event_i -> gt_(i+1)`를 14개 장면에서 수동 표본 검증한다.
4. 원본 timestamp 단조 증가와 event 좌표 범위를 검사하는 validation을 추가한다.
5. SNN 평가에서 calibrated checkpoint flag와 `simulation_steps>=1`을 강제한다.
6. 공식/논문 metric 코드와 자체 box-window SSIM 값을 대조한다.
7. full dataset, 실제 CUDA GPU에서 학습·평가·메모리·지연 재현 기록을 만든다.
8. validation 첫 500개 선택과 전체/scene-balanced 결과 차이를 확인한다.

### P1: 연구 주장과 과제 범위를 맞추기 위해 필요

1. 정확한 B-spline ASGCN baseline과 현재 MLP-gated encoder를 분리 비교한다.
2. 실제 timestep IF/LIF 또는 명확한 ANN quantization baseline을 구현한다.
3. spike-timestep 수에 따른 실제 operation/latency/energy 비용을 측정한다.
4. EventHDR log domain과 EventAid-R linear domain 차이를 통제하거나 해석한다.
5. event count/density channel, max-events, graph radius, raster size ablation을 수행한다.
6. 장기 temporal 학습이 필요하면 truncated-BPTT 길이를 늘린다.
7. 반도체 과제용 event compression, memory traffic, accelerator, FPGA/ASIC 검증을 별도 개발한다.

### P2: engineering 정리

1. PowerShell downloader를 Bash판과 동일한 safe default·resume·ZIP/hash 검증으로 맞춘다.
2. `eval.batch_size`, train `output.save_predictions` 같은 현재 미사용 config 필드를 제거하거나 구현한다.
3. dependency lock 또는 검증된 Torch/CUDA environment file을 추가한다.
4. EventAid-R manifest에 공식 checksum이 확보되면 추가한다.
5. 장기 process에서 dataset handle을 명시적으로 닫는 lifecycle을 정리한다.
6. random crop이 epoch별로 바뀌면서 scene 내 frame에는 일관되도록 sampler state를 설계한다.
7. malformed EventAid txt를 조용히 자르지 말고 token 수·timestamp·좌표를 명시 검증한다.
8. calibration checkpoint의 평가 전용 여부와 ANN/SNN resume 경로를 분리한다.

## 19. 데이터·보안·운영 주의

- `.env`는 `setup.sh`가 shell code로 `source`하므로 신뢰하는 값만 넣고 권한을 제한한다.
- token, SSH private key, storage password를 config, `.env`, Slurm script, 로그에 기록하지 않는다.
- `sbatch --export=ALL` 전에 불필요한 credential 환경변수를 제거한다.
- 데이터 라이선스와 재배포 조건을 확인하고 H5/ZIP을 GitHub에 올리지 않는다.
- EventAid-R은 ZIP 상태로 사용한다. 압축 해제하면 100GB 목표를 침해할 수 있다.
- 공식 eval이나 EventAid-R 결과를 보고 hyperparameter/threshold를 다시 선택하면 잠금시험이 아니다.
- GPU OOM 시 `max_events`, `causal_candidates`, `decoder_channels` 순으로 줄인다.
- 장시간 실행은 interactive SSH shell 대신 `tmux`나 Slurm을 사용한다.
- 의존성은 version range이며 완전한 lockfile이 아니므로 설치 시점이 달라지면 환경이 달라질 수 있다.

## 20. 다른 ChatGPT에 줄 교차 검증 요청문

아래 문장을 `hand_off.md`, `code_summary.md`와 함께 전달하면 된다.

```text
hand_off.md는 프로젝트 설명이고 code_summary.md는 기준 커밋의 Git-tracked 파일 원문 전체다.
설명을 그대로 믿지 말고 반드시 code_summary.md의 실제 함수·config·script와 대조하라.

다음을 수행해라.
1. hand_off의 모든 핵심 주장과 실제 코드가 일치하는지 claim-to-code audit을 해라.
2. EventHDR/EventAid-R frame-event 정렬, leakage, timestamp, tone-map 문제를 검토해라.
3. ASGCN이 원 논문과 얼마나 다른지, 현재 SNN을 SNN이라 부를 수 있는지 구분해라.
4. train/resume/state detach/loss/metric/latency 측정에서 실험 결과를 왜곡할 버그를 찾아라.
5. MobaXterm, Bash, PowerShell, Slurm, Docker 실행 경로와 파일 참조가 모두 유효한지 확인해라.
6. 테스트가 놓친 실제 데이터·GPU·full training 실패 가능성을 찾아라.
7. 저지연·고효율 반도체 통합 처리 시스템이라는 과제 목표 대비 빠진 구현을 적어라.

결과는 P0(결과 무효 가능), P1(연구 주장 제한), P2(개선)로 나누고,
각 항목마다 파일 경로, 함수/설정 이름, 근거, 재현 방법, 권장 수정안을 제시해라.
추측과 코드로 확인한 사실을 명확히 구분해라.
```

## 21. 삭제·단축 이력

production에서 제거된 경로:

- `src/asgcn_recon/smoke.py`
- `scripts/verify_real_samples.py`
- `scripts/smoke_test.ps1`
- `data/smoke/`와 build/cache 산출물
- 작업 폴더의 원문 HWP

주요 단축 이름:

| 현재 | 이전 |
|---|---|
| `scripts/setup.sh` | `scripts/setup_server.sh` |
| `scripts/check_env.py` | `scripts/check_environment.py` |
| `scripts/get_aid.sh` | `scripts/download_eventaid_r.sh` |
| `scripts/train.sh` | `scripts/run_train.sh` |
| `scripts/eval.sh` | `scripts/run_eval.sh` |
| `configs/hdr_train.json` | `configs/eventhdr_train.json` |
| `configs/hdr_ann.json` | `configs/eventhdr_eval.json` |
| `configs/hdr_snn.json` | `configs/eventhdr_snn_eval.json` |
| `configs/aid_ann.json` | `configs/eventaid_r_eval.json` |
| `configs/aid_snn.json` | `configs/eventaid_r_snn_eval.json` |
| `server/train.sbatch` | `server/slurm_train.sbatch` |
| `server/eval.sbatch` | `server/slurm_eval.sbatch` |
