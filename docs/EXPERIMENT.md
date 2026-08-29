# 본실험 프로토콜

이 문서는 `configs/hdr_train.json`, `configs/hdr_{ann,snn}.json`,
`configs/aid_{ann,snn}.json`과 현재 실행 코드를 기준으로 한다. 전체 기본 실행은
`bash scripts/full.sh`이며, 일부 파일이나 일부 frame만 사용한 결과는 본실험 결과로 합치지 않는다.
ASGCN 수식과 이 저장소의 과제용 확장 범위는 [ASGCN.md](ASGCN.md)를 함께 참고한다.

## 1. 연구 질문과 데이터 역할

1. EventHDR 실제 event로 luminance frame을 얼마나 잘 복원하는가?
2. 동일 ANN checkpoint를 보정한 뒤 `literal_eq15`와 `standard_if`가
   `T=4,8,16,32`에서 보이는 품질·지연 차이는 무엇인가?
3. 학습과 보정에 쓰지 않은 EventAid-R 14개 scene에서 성능이 얼마나 유지되는가?

| 단계 | 고정 데이터 | 코드상 역할 |
|---|---|---|
| 학습 | EventHDR 공식 train root의 `1.h5`–`51.h5` | 40 epoch weight 최적화 |
| 내부 평가 | EventHDR 공식 eval root의 `1.h5`–`19.h5` | 마지막 epoch에서만 ANN 평가하고 `best.pt` 생성 |
| ANN→SNN 보정 | EventHDR train 51개 H5의 모든 선택 frame | BN folding, activation maximum, 식 (6) 정규화 |
| 내부 비교 | 동일 EventHDR eval 19개 H5 | ANN과 두 IF dynamics × 네 T 비교 |
| 외부 비교 | EventAid-R manifest의 14개 ZIP | 고정 checkpoint의 외부 일반화 평가 |

`manifests/eventhdr_split.json`은 `status=final`,
`split_schema=official_separate_roots_v1`이다. train과 eval은 별도 root이고 숫자 basename이 겹치는
공식 배포 구조다. H5 하나를 recurrent state와 metric의 sequence group으로 사용할 뿐, 공개 자료에
물리 scene 대응표가 없으므로 두 root가 물리 scene-disjoint이거나 통계적으로 독립이라고 주장하지
않는다. JSON의 `macro`/`per_scene`도 EventHDR에서는 물리 scene이 아니라 H5 sequence-file 단위다.

`train.validate_every=null`이므로 1–39 epoch에는 내부 평가를 하지 않고 40번째 마지막 epoch에서만
EventHDR eval 전체를 평가한다. 따라서 현재 `best.pt`는 여러 epoch를 비교해 고른 checkpoint가 아니라
마지막 epoch checkpoint다. 파일명 `best.pt`와 `best_metric=macro_ssim`은 공통 checkpoint 계약을 위한
것이다. 이후 같은 eval root에서 생성하는 ANN/SNN 표는 내부 비교이며 독립 잠금시험으로 표현하지
않는다. EventAid-R 결과를 본 뒤 설정·보정·threshold를 바꾸면 그 결과 역시 외부시험으로 해석할 수
없다.

## 2. 고정 전처리

두 dataset의 target은 다음 순서로 `[0,1]` luminance domain에 놓는다.

```text
integer image / dtype maximum
  -> RGB이면 BT.709 luminance
  -> y = log1p(5000*x) / log1p(5000)
```

EventAid-R은 event block `i`와 GT `i+1`을 짝짓는 `target_offset=1`을 사용한다. 이는 이 과제를
위한 정렬 가정이며 다른 offset 결과와 섞지 않는다. 두 dataset의 sensor response와 exposure가 같다는
보장은 없으므로 절대 PSNR/SSIM을 동일 분포의 수치처럼 직접 비교하지 않는다.

event 전처리는 다음 순서다.

1. 설정된 sensor crop을 적용한다. 본실험 config는 `crop_size=null`이라 전체 sensor를 쓴다.
2. crop 뒤 event 수가 `max_events=8192`를 넘으면, 양 끝 timestamp를 포함하는 결정적 `linspace`
   index로 **정확히 8,192개**를 시간 전역에서 남긴다. 8,192개 이하면 그대로 둔다.
3. model의 논문식 균일 sampling factor `R=event_sampling_factor=1`을 적용한다.

두 번째 단계는 graph memory를 제한하기 위한 과제용 cap이며 ASGCN 논문의 공식 `R`이 아니다.
`raw_event_count`, `cropped_event_count`, `retained_event_count`, `dataset_sampling_ratio`를 분리해
기록한다. 동일 source sequence의 crop은 seed와 상대 sequence identity로 결정되므로 worker 수와
resume 여부에 따라 바뀌지 않는다.

EventHDR에서 연속 target 사이 event가 0개인 interval도 sample로 보존한다. EventAid-R의 빈 event
text도 빈 `[0,4]` tensor로 유지한다. 빈 sample은 0 node·0 edge graph와 zero raster를 거쳐 analog
decoder/ConvGRU로 처리되며 임의 event를 합성하지 않는다. 보정에서는 비어 있는 activation이 layer의
유효 calibration observation으로 집계되지 않고, 어느 graph layer든 non-empty observation이 0개면
변환을 거부한다.

## 3. graph와 모델 고정값

기본 graph node feature는 정규화한 `(x,y,t,polarity)`이고, 거리 계산은 `[0,1]`로 정규화한
`(x,y,t)` 3차원에서 한다. `graph_radius=0.08`보다 Euclidean 거리가 **엄격히 작은** 서로 다른 node를
연결하고 양 방향 directed edge를 저장한다. edge pseudo-coordinate는 `distance/radius` 한 값이다.

구현은 폭이 radius인 uniform cell과 인접 `3^3` cell로 후보를 찾은 뒤 exact Euclidean 조건을 다시
적용한다. 이는 근사 k-NN이나 edge truncation이 아니다. directed edge 수가
`max_graph_edges=2,000,000`을 넘으면 일부 edge를 버리지 않고 실패한다. isolated node도 유지하며
비율과 최대 degree를 결과에 기록한다.

고정 모델은 6-layer, hidden 64의 pure-PyTorch open degree-1 B-spline graph encoder, feature
rasterization, residual U-Net과 analog ConvGRU decoder다. ANN→SNN 변환과 IF timestep은 graph
encoder에만 적용된다. decoder는 두 모드 모두 analog이다.

## 4. 학습과 checkpoint 선택

본학습 설정은 다음과 같다.

- 40 epochs, chronological `batch_size=1`, shuffle 없음
- Adam + gradient centralization, learning rate `1e-3`, weight decay `5e-3`
- MultiStepLR milestones 20/30, gamma 0.1
- CUDA에서 AMP, gradient norm clip 1.0
- Charbonnier 1.0 + SSIM 0.2 + gradient 0.1 + temporal 0.2
- train/validation sample cap 없음, 마지막 epoch에서만 전체 EventHDR eval 평가

ConvGRU state와 temporal loss는 같은 H5 sequence에서 index가 1씩 이어지고 sensor shape가 같을 때만
이어진다. 경계에서는 state와 이전 prediction/target을 초기화한다. 내부 평가도 모든 frame을
사용하므로 별도 표본 추출은 없다.

training artifact는 `runs/eventhdr_asgcn/`에 기록한다.

- `config.json`: 실행 시 resolve된 전체 config
- `history.json`: epoch별 loss, 마지막 epoch validation, learning rate, CUDA peak memory
- `last.pt`: 매 epoch 끝에 저장하는 model/optimizer/scheduler/scaler/RNG 포함 재개 checkpoint
- `best.pt`: 현재 protocol에서는 마지막 epoch의 clean ANN inference checkpoint
- `.data_hash_cache.json`: full source hash 계산을 가속하는 로컬 cache

새 학습은 위 핵심 artifact가 이미 있는 run directory를 덮어쓰지 않는다. 중단된 run은 `last.pt`로
재개한다.

## 5. 전체 ANN→SNN 보정

기본 본실험은 EventHDR train의 모든 frame을 사용한다.

```bash
bash scripts/calibrate.sh \
  configs/hdr_train.json \
  runs/eventhdr_asgcn/best.pt \
  runs/eventhdr_asgcn/best_snn.pt
```

wrapper 기본값 `CALIBRATION_SAMPLES=all`은 모든 training frame을 선택한다. model을 eval mode로
전환해 graph BN을 convolution parameter에 fold하고, non-empty sample의 layer별 feature-wise ReLU
maximum을 측정한 뒤 식 (6) parameter normalization과 unit threshold를 적용한다. 출력
`best_snn.pt`는 optimizer와 training history를 제거한 SNN inference checkpoint이며 calibration
표본 수·유효 표본 수·dead channel 수·sampling summary와 model tensor SHA-256을 포함한다.

기존 출력은 기본적으로 보호한다. 명시적으로 다시 보정해야 할 때만
`OVERWRITE_CALIBRATION=1`을 wrapper에 전달한다. engine은 새 checkpoint를 끝까지 만든 뒤 atomic
replace하므로 변환 실패 전에 기존 파일을 먼저 삭제하지 않는다. ANN 평가는 변환 전 `best.pt`, SNN
평가는 `best_snn.pt`를 써야 하며 서로 바꾸면 checkpoint 검증에서 거부된다.

## 6. 고정 평가·benchmark 행렬

`scripts/full.sh`는 full data/environment 검사와 모든 선택 sample의 decode 검증, 학습 또는 재개,
전체 보정, 아래 행렬의 evaluate와 benchmark를 순서대로 실행한다.

| dataset | mode | dynamics | T | checkpoint |
|---|---|---|---|---|
| EventHDR eval 19 | ANN | 해당 없음 | 해당 없음 | `best.pt` |
| EventHDR eval 19 | SNN | `literal_eq15` | 4, 8, 16, 32 | `best_snn.pt` |
| EventHDR eval 19 | SNN | `standard_if` | 4, 8, 16, 32 | `best_snn.pt` |
| EventAid-R 14 | ANN | 해당 없음 | 해당 없음 | `best.pt` |
| EventAid-R 14 | SNN | `literal_eq15` | 4, 8, 16, 32 | `best_snn.pt` |
| EventAid-R 14 | SNN | `standard_if` | 4, 8, 16, 32 | `best_snn.pt` |

즉 dataset마다 ANN 1개와 SNN 8개, 전체 18개 quality evaluation과 18개 compute benchmark를 만든다.
benchmark 기본값은 warmup 10 frame, 측정 100 frame이다. recurrent benchmark는 각 측정 window 앞의
같은 H5/ZIP sequence predecessor를 최대 32개까지 timer 밖에서 replay한다. dataset read와
host-to-device 이동도 timer 밖이며, CUDA에서는 CUDA Event로 model forward를 측정한다. graph 생성은
model forward 안에 있으므로 측정에 포함된다.

`literal_eq15`는 공개 식 (15)의 `+h(t-1)`까지 문자 그대로 실행한다. `standard_if`는 그 항을 뺀
rate-conversion 대조군이며 저자 공식 dynamics라고 주장하지 않는다. 두 모드 모두 T회의 IF
recurrence를 실제 실행하지만 analog decoder 때문에 결과를 완전 SNN hardware 성능으로 해석하지
않는다.

## 7. 지표와 산출물

quality는 frame별 PSNR, Gaussian-window SSIM, RMSE와 연속 frame 사이의 `temporal_l1`을 기록한다.
`temporal_l1`은
`L1((pred_t-pred_{t-1}), (target_t-target_{t-1}))`이며 sequence 경계의 첫 frame은 집계에서 제외되고
CSV에는 null이다. 요약은 모든 frame의 `micro`, H5/ZIP group 평균을 다시 평균한 `macro`, 호환 key
`per_scene`으로 나뉜다. LPIPS는 `eval.lpips=true`와 optional dependency를 명시한 별도 run에서만
계산한다.

evaluate는 quality, model-forward latency, RTF/deadline miss, graph topology, dataset coverage와
CUDA peak allocated/reserved memory를 기록한다. benchmark는 mean/p50/p90/p95/p99/max latency, FPS,
raw/retained event rate, graph node rate, event retention, node/edge/isolated-node 수, SNN firing rate,
RTF, state reset과 peak GPU memory를 기록한다.

```text
runs/eventhdr_official_eval_ann/ann/
runs/eventhdr_official_eval_snn/snn_<literal_eq15|standard_if>_T<4|8|16|32>/
runs/eventaid_r_external_ann/ann/
runs/eventaid_r_external_snn/snn_<literal_eq15|standard_if>_T<4|8|16|32>/
```

각 directory의 `metrics.json`, `frames.csv`, `predictions/`는 evaluate가 만들고
`benchmark.json`은 benchmark가 만든다. prediction은 config 기본값에 따라 처음 20 frame의 pred/GT
PNG를 저장한다. 같은 mode/dynamics/T artifact가 있으면 묵시적으로 덮어쓰지 않고 실패하므로 재실행
전 기존 결과를 보존 위치로 옮기거나 별도 `eval.output_dir` config를 사용한다.

## 8. epoch-boundary exact resume

```bash
RESUME_CHECKPOINT="$PWD/runs/eventhdr_asgcn/last.pt" \
  bash scripts/full.sh
```

`last.pt`는 완료된 epoch 뒤에만 저장된다. 따라서 중간에 종료된 epoch의 일부 step부터 이어가는 것이
아니라 마지막으로 완료된 epoch 다음부터 동일 trajectory를 재개한다. resume checkpoint는 configured
`run_dir` 안에 있어야 하고, 검증 score가 이미 있으면 같은 run의 historical `best.pt`도 필요하다.

exact resume은 다음 항목의 일치를 강제한다.

- model, optimizer/GC, scheduler, loss, batch/order/worker, AMP·TF32·determinism 설정
- seed, validation protocol, official manifest, transform와 선택 sample identity
- train 51개와 eval 19개 source의 상대 파일 identity·size·full SHA-256
- `src/**/*.py` byte의 SHA-256, Git commit과 source dirty 상태
- device type, PyTorch/CUDA/cuDNN, GPU 이름·compute capability와 CUDA backend flags
- CUDA RNG state 수와 현재 visible CUDA device 수

따라서 다른 GPU 종류, CUDA/PyTorch 조합, source checkout, worker protocol이나 dataset byte로 옮긴
checkpoint는 “exact” 재개로 허용되지 않는다. 절대 data mount path와 mtime/ctime 자체는 checkpoint
protocol에 넣지 않는다. 같은 상대 파일과 byte를 다른 mount에 복사하면 다시 full hash한 뒤 일치할
수 있다. `.data_hash_cache.json`은 같은 절대 경로의 size/mtime/ctime이 모두 같을 때만 기존 full
hash를 재사용하며, 원본을 교체·복원했으면 `train.rehash_data=true`인 별도 config로 cache를 무시한다.

## 9. 과학적 한계와 중단 조건

- 공개된 저자 공식 코드가 확인되지 않아 논문에 없는 graph normalization, spline hyperparameter,
  threshold 세부값은 이 저장소의 명시적 가정이다. “공식 ASGCN 완전 재현”으로 표현하지 않는다.
- 원 ASGCN의 classification pooling/MLP와 dynamic asynchronous K-hop update는 구현하지 않았다.
  현재 graph는 frame interval마다 정적으로 다시 만든다.
- residual U-Net, rasterization, analog ConvGRU, 복원 loss, tone mapping, 정확한 `max_events` cap과
  EventAid-R offset은 과제용 확장이다.
- graph encoder만 IF로 변환된다. PyTorch GPU latency·발화율은 FPGA/ASIC latency, power 또는 energy
  측정이 아니며 event 전송·압축 protocol이나 RTL도 구현하지 않았다.
- EventHDR 공식 train/eval의 물리 scene 독립성을 입증하지 못했으므로 내부 결과에 독립 test-set
  일반화 주장을 붙이지 않는다. EventAid-R도 sensor/domain 차이를 통제한 benchmark는 아니다.
- `literal_eq15`의 self-feedback은 표준 rate conversion과 수학적으로 동등하지 않을 수 있다.
  dynamics와 T를 생략한 ANN↔SNN 비교는 보고하지 않는다.
- file coverage/HDF5·ZIP decode 실패, non-finite loss/metric, 비단조 timestamp, 좌표 범위 오류,
  `max_graph_edges` 초과 또는 OOM이 발생하면 해당 sample을 조용히 제외하지 말고 run을 폐기해 원인과
  설정을 고친 뒤 새 artifact로 실행한다.
