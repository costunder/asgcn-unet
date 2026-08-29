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
