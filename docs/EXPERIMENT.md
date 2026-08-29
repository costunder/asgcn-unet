# 실험 프로토콜

## 1. 고정할 연구 질문

1. EventHDR의 실제 이벤트에서 luminance frame을 얼마나 잘 복원하는가?
2. ASGCN paper-core의 graph depth와 radius가 복원 품질·비용에 어떤 영향을 주는가?
3. EventAid-R의 다른 장면·운동에서 품질이 얼마나 유지되는가?
4. ANN, 논문 식 (15)를 문자 그대로 실행한 IF, 표준 IF 대조군의 품질·연산 지연 차이는 무엇인가?

현재 `snn` 모드는 graph encoder의 IF membrane을 timestep마다 전개하지만 U-Net·ConvGRU decoder는
analog이고 GPU에서 기능을 모사한다. 따라서 저전력·에너지 우위를 검증하는 질문에는 답할 수 없으며,
해당 주장은 연산량 모델 또는 hardware 측정 후에만 추가한다. 논문 재현 범위는
[ASGCN 구현 범위](ASGCN.md)를 따른다.

기본 `literal_eq15`의 `+h(t-1)` self-feedback은 표준 rate-conversion IF와 동등하지 않다. 따라서
ANN↔SNN 오차는 `snn_dynamics`와 T를 함께 표기한다. `standard_if`는 모호성 분석용 대조군이지
ASGCN 저자 공식 구현이라고 간주하지 않는다.

## 2. 데이터 역할과 누수 방지

| 단계 | 데이터 | 허용되는 사용 |
|---|---|---|
| 학습 | EventHDR train | weight 최적화·crop |
| 검증 | EventHDR holdout | 물리 scene split 확정 후 macro SSIM checkpoint 선택 |
| 보정 | EventHDR train | BN folding·feature-wise threshold·parameter normalization |
| 내부 최종시험 | EventHDR 공식 eval | 학습 종료 후 1회 |
| 외부 최종시험 | EventAid-R | 학습·보정·threshold 선택 금지 |

`manifests/eventhdr_split.json`은 현재 물리 scene 대응표가 없는 legacy file-list
`provisional` 상태다. 최종 manifest는 아래처럼 동일 물리 장면의 파일을 `scene_groups`로 묶고,
겹치지 않는 scene ID를 `train_scenes`와 `val_scenes`에 넣어야 한다.

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

예시는 schema 설명일 뿐 실제 mapping이 아니다. `status`만 `final`로 바꾸거나 legacy 목록만 남기면
검증에서 거부된다. `configs/hdr_smoke.json`만 비보고용 provisional legacy schema를 허용한다.
smoke는 별도 `manifests/eventhdr_smoke.json`의 train `1.h5`, `2.h5`와 validation `48.h5`,
`49.h5`만 사용하고 content fingerprint도 이 네 파일만 hash한다.
final manifest는 `data/EventHDR/train` 아래의 모든 H5를 정확히 한 scene에 소유시키고 모든 scene을
train 또는 validation에 배정해야 한다. 누락·미선언 파일과 미배정·중복 scene/file은 거부한다.

validation sample limit은 채점 frame 수에만 적용된다. `hdr_train`은 최대 500개, `hdr_smoke`는
최대 32개를 채점하며 아래 recurrent context frame은 이 수에 포함하지 않는다.

1. group(final은 physical scene, provisional은 H5 file)별 quota를 round-robin으로 배정한다.
2. 각 group의 dataset index에서 deterministic contiguous window를 선택한다.
3. recurrent 모델은 window 앞의 같은 group predecessor를 `validation_context_frames` 한도에서
   metric 없이 replay해 streaming ConvGRU state를 예열한다. 기본값은 본학습 64, smoke 8이며
   `null`이면 전체 prefix다. non-recurrent 모델은 context를 replay하지 않는다.
4. sample limit이 group 수보다 작으면 일부 group을 버리지 않고 오류로 중단한다.
5. checkpoint 선택에는 scene별 SSIM 평균의 평균인 macro SSIM을 쓴다.

calibration은 recurrent state를 쓰지 않으므로 각 group(final은 physical scene, provisional은 H5
file)의 전체 index 범위를 `linspace`로 덮는다.
benchmark는 recurrent 모델이면 group별 연속 window와 최대 `eval.recurrent_context_frames`개의
unmeasured predecessor(현재 eval config 기본 32), 비순환 모델이면 time-spread sample을 사용한다.
`--warmup`은 recurrent context가 아니라 device/kernel warmup이다.
장면, sequence index, sensor shape가 끊기는 경계에서는 state를 초기화하며 benchmark 결과에 reset 수와
비율을 기록한다.

random crop의 RNG는 scene·source file/member를 합친 안정적인 sequence identity로 결정한다. 따라서
worker 수나 resume 여부와 무관하고, 같은 연속 sequence의 모든 frame은 동일한 sensor ROI를 사용해
ConvGRU pixel state와 temporal loss를 정렬한다. 이는 epoch마다 바뀌는 random augmentation이 아니다.

event 전처리는 spatial crop → adaptive `max_events` cap → model의 고정 paper sampling factor `R`
순서다. adaptive cap은 crop 뒤 event 수 `N`에 대해 `ceil(N/max_events)` 정수 stride를 사용하며,
서버 메모리를 위한 재구성 가정이지 논문의 공식 `R`이 아니다. sample metadata는 cap 전후를
`raw_event_count`, `cropped_event_count`, `retained_event_count`, `dataset_sampling_factor`로 분리한다.

## 3. 학습 optimizer protocol

논문에 명시된 Adam, gradient centralization, 초기 learning rate `1e-3`, L2 weight decay `5e-3`를
`optimizer=adam_gc`, `learning_rate=0.001`, `weight_decay=0.005`로 고정한다. 논문은 milestone에서
learning rate를 낮춘다고만 쓰고 정확한 epoch와 gamma를 공개하지 않는다. 따라서 40-epoch 복원
적응에서는 `lr_milestones=[20,30]`, `lr_gamma=0.1`을 명시적 가정으로 사용한다. 1-epoch smoke도
동일한 protocol을 기록하지만 milestone에 도달하지 않는다.
gradient centralization의 axis도 구현 가정이다. spline `[K,Cin,Cout]`와 root `[Cin,Cout]`는 마지막
output axis 외 차원에서, Conv/Linear는 첫 output axis 외 차원에서 gradient 평균을 제거한다.

## 4. output domain

EventHDR와 EventAid-R 모두 다음 target 변환을 쓴다.

```text
integer image -> dtype range로 [0,1] 정규화
RGB이면 BT.709 luminance
y = log1p(5000*x) / log1p(5000)
```

이는 output 수치 domain만 통일한다. 센서 response와 exposure가 동일하다는 보장은 없으므로 두
dataset의 절대 PSNR/SSIM을 동일 분포처럼 해석하지 않는다.
EventAid-R은 event block `i`를 GT `i+1`과 짝짓는 `target_offset=1`을 명시적 protocol 가정으로
사용한다. 이 정렬을 바꾼 결과는 같은 외부시험으로 합치지 않는다.

## 5. 실행 순서

```bash
# 전체 EventAid-R 전 작은 ZIP 하나로 loader만 비보고 점검
bash scripts/get_aid.sh R-bear
asgcn-recon inspect --config configs/aid_smoke.json --samples 2 --validate-all

# smoke용 EventHDR train과 GPU만 먼저 확인
python scripts/check_env.py --require-cuda --require-eventhdr-smoke \
  --lock constraints/py312.txt

# smoke manifest의 네 H5만 decode/hash하는 1 epoch 점검
asgcn-recon inspect --config configs/hdr_smoke.json --samples 2 --validate-all
asgcn-recon train --config configs/hdr_smoke.json

# 본학습·고정 내부/외부 평가 전 전체 파일 수 확인
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt

# 모든 event block 검증
asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all

# scene split final 이후 ANN 본학습
asgcn-recon train --config configs/hdr_train.json

# EventHDR train만으로 IF-SNN 보정·변환
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500

# 고정 내부시험
asgcn-recon evaluate \
  --config configs/hdr_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics literal_eq15

# 같은 calibrated checkpoint를 쓰는 비공식 standard-IF 대조군
asgcn-recon evaluate \
  --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics standard_if

# 고정 외부시험
asgcn-recon evaluate \
  --config configs/aid_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/aid_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16 \
  --snn-dynamics literal_eq15
```

SNN 명령은 ANN checkpoint, 모든 graph layer에서 유효한 non-empty calibration observation이 0개인
checkpoint, BN 미fold·parameter 미정규화 상태, 비정수 또는 `simulation_steps < 1`을 거부한다.
checkpoint metadata는 persistent graph-layer flag 및 변환 뒤 unit threshold와도 교차검증한다.
모든 ANN/SNN/training checkpoint는 model tensor byte의 SHA-256을 저장하고 load 전에 검증한다.
반대로 Eq. (6)이
적용된 SNN checkpoint를 ANN 모드에 넣는 것도 거부하므로 ANN 평가는 변환 전 `best.pt`를 쓴다.
`scripts/eval.sh`에서는 동일한 override를 `SNN_DYNAMICS=literal_eq15` 또는
`SNN_DYNAMICS=standard_if`로 전달한다.

최종 EventHDR eval config는 H5 정확히 19개를, EventAid-R final config는 manifest와 이름이 같은
ZIP 정확히 14개를 강제한다. 일부 EventAid-R만 허용하는 `aid_smoke.json` 결과는 보고용 외부시험이
아니다.

## 6. 품질 지표

- PSNR: `[0,1]` data range
- SSIM: 11×11, σ=1.5 Gaussian valid window; 작은 영상은 fitting odd window
- RMSE
- `temporal_l1`: 같은 scene·sensor shape에서 sequence index가 정확히 1 증가하는 frame 사이에만
  `L1((pred_t-pred_t-1), (gt_t-gt_t-1))`
- LPIPS: `eval.lpips=true`일 때만 선택적으로 실행

결과는 micro, group macro, per-group으로 계산한다. final holdout의 group은 physical scene,
provisional/EventHDR 공식 eval은 H5 파일, EventAid-R은 `R-*.zip` scene이다. JSON의 기존
`macro`/`per_scene` 이름은 호환성을 위해 유지한다. 첫 frame과 장면·index·shape 불연속 뒤 첫
frame은 `temporal_l1` 집계에서 제외되고 CSV에는 null이 들어간다.

기존 논문과 SSIM을 비교할 때는 해당 논문의 구현, crop, border, color space, tone mapping까지
동일하게 맞춰 별도 검증한다. 현재 Gaussian 구현을 사용했다는 이유만으로 공식 수치와 완전히
동일하다고 가정하지 않는다.

## 7. 지연·메모리 지표

- evaluate: graph 생성과 model forward latency, 첫 frame cold start 포함
- benchmark: dataset read와 host-to-device 이동 제외, warmup 이후 CUDA Event 측정
- mean, p50, p90, p95, p99, max, FPS
- `raw_events_per_second`: spatial crop/cap 전 source interval event 수 / model compute time
- `retained_events_per_second`: crop과 adaptive cap 뒤 event 수 / model compute time
- `graph_nodes_per_second`: model의 추가 factor `R` 뒤 graph node 수 / model compute time
- `mean_raw_events`, `mean_retained_events`, `retention_ratio`, 평균 edge
- isolate 비율과 max degree; `max_graph_edges=2,000,000` 초과는 graph를 자르지 않고 실패
- SNN layer별 `총 spike / 총 neuron-step` 발화율과 전체 가중 발화율
- timestamp 기반 RTF, deadline miss ratio
- peak allocated/reserved GPU memory

세 rate 모두 benchmark가 제외한 dataset read/I/O throughput이 아니라, 같은 model compute 시간으로
정규화한 workload rate다. 기존 `events_per_second`는 deprecated 하위 호환 alias이며 항상
`retained_events_per_second`와 같다. metadata가 없는 custom dataset은 tensor에 실제 남아 있는 event
수를 raw와 retained 양쪽의 안전한 fallback으로 쓴다.

`snn` 경로는 T번 IF membrane timestep을 실제로 실행하고 fixed edge의 B-spline basis/index는
sample당 한 번 계산해 layer와 timestep에서 재사용한다. 따라서 `T=4/8/16/32`는 이 PyTorch
구현에서 dynamics별 timestep 수에 따른 품질·GPU latency ablation이다. decoder는 analog이므로 이를
neuromorphic accelerator의 latency나 에너지로 환산하지 않는다.

artifact는 `<eval.output_dir>/ann/` 또는
`<eval.output_dir>/snn_<dynamics>_T<steps>/`에 `metrics.json`, `frames.csv`, `predictions/`,
`benchmark.json`으로 나뉜다. 같은 mode/dynamics/T artifact가 이미 있으면 덮어쓰지 않고 실패한다.
`benchmark.json`은 benchmark가, 나머지는 evaluate가 기록한다.
prediction PNG는 평가 순번, 안전한 slug, 전체 sample ID hash를 결합해 파일명 충돌과 OS 금지 문자를
차단한다.

## 8. 최소 비교표

1. `graph_layers=3` vs 6
2. ANN vs `literal_eq15` vs `standard_if`, 각각 `T=4/8/16/32`
3. `max_events=4096/8192/16384`
4. 정규화된 `x,y,t`에서 `graph_radius=0.04/0.08/0.12`
5. ConvGRU on/off
6. EventHDR 내부 macro/per-scene vs EventAid-R 외부 macro/per-scene

모든 비교는 split, seed, tone mapping, crop, 해상도, checkpoint selection rule을 고정한다. 비교마다
config 원문, Git commit, `check_env.py` 출력, GPU 이름, CUDA/PyTorch, peak memory와 wall-clock을 함께
보존한다.

exact resume protocol은 선택 frame identity, group 길이, transform, manifest와 선택된
train/validation 원본의 SHA-256을 저장한다. smoke에서는 네 H5만, 본학습에서는 최종 manifest의
모든 H5를 hash한다. 절대경로와 mtime은 protocol에서 비교하지 않아 상대 파일 identity와 byte가
같은 복사본은 다른 mount에서도 재개할 수 있다. 같은 경로의 resume은 run 폴더 sidecar에서
size/mtime/ctime이 모두 같은 파일의 기존 full hash를 재사용한다.
원본을 교체·복원했거나 전수 확인하려면 `rehash_data=true`로 cache를 무시한다.

## 9. 중단 조건

- manifest가 provisional이면 본학습 금지; `status`만 바꾸고 final scene schema를 생략해도 금지
- 전체 dataset validation 실패 시 해당 파일 제외가 아니라 원본 재다운로드/검증
- A100 10GB smoke OOM이면 full training 전에 기본 graph/model 설정을 재검토
- NaN loss/metric, 비단조 timestamp, 범위 밖 좌표 발생 시 결과 폐기
- EventAid-R 결과를 본 뒤 hyperparameter나 threshold를 바꾸면 기존 결과를 잠금시험으로 표기 금지
- A6000/A100 latency를 FPGA/ASIC latency 또는 에너지로 환산해 주장 금지
