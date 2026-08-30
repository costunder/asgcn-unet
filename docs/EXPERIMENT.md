# 본실험 프로토콜

이 문서는 `configs/train.json`, `configs/hdr.json`, `configs/aid.json`과 현재 실행 코드를
기준으로 한다. ANN/SNN 실행은 같은 평가 config를 공유하고 CLI 인자로 모드를 구분한다. 전체 기본 실행은
`bash scripts/run.sh`이며, 일부 파일이나 일부 frame만 사용한 결과는 본실험 결과로 합치지 않는다.
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

### EventHDR 이벤트 인덱스

이미지의 유효한 `event_idx`는 그대로 사용한다. 공식 train `38.h5`처럼 이 속성이 없는 파일은
이미지 `timestamp`와 정렬된 `events/ts`로 누락 인덱스만 복원한다. 복원 규칙은
`max(searchsorted(events/ts, timestamp, side="left") - 1, 0)`이다. 이는 EventHDR가 연결한
[참조 HDF5 packager](https://github.com/TimoStoff/events_contrast_maximization/blob/ab74aba1ab3481689628ce374f8dcc92b5383b00/tools/event_packagers.py#L74-L90)의
predecessor 규칙을 따르는 호환 정책이며, 표준 `[이전 timestamp, 현재 timestamp)` 경계와는 다르다.
전역 lower bound를 사용하므로 upstream의 buffer 시작점 clamp 동작까지 복제하는 것은 아니다.
기존 인덱스를 일괄 재계산하거나 원본 H5를 변경하지 않는다.

복원 시 전체 event timestamp를 1,048,576개 이하의 block으로 읽어 유한값과 block 간 단조성까지
검사한다. 이미지 timestamp가 없거나 유효하지 않은 저장 인덱스가 있으면 실패하며, 비어 있지 않은
event stream과 이미지의 시간 범위가 완전히 분리된 경우에도 복원하지 않는다. 빈 event stream은
0 경계를 사용한다. timestamp 단위나 기준시각을 추측해 변환하지 않는다.

`inspect`의 `event_indexing`에 파일별 저장·복원 이미지 수를, sample metadata의 `event_idx_source`에
`stored` 또는 `timestamp_predecessor_v1`을 기록한다. 선택된 `start_idx/end_idx`는 기존 dataset index
identity hash에도 반영된다. 로더가 변경되었으므로 이전 코드에서 생성한 profile/checkpoint와 exact
resume을 강제하지 않고 현재 source·data로 사전검증한다.

## 2. 고정 전처리

두 dataset의 target은 config에 명시한 다음 고정 계약으로 `[0,1]` luminance domain에 놓는다.

```text
target_normalization.mode = integer_dtype_max
  -> integer image / dtype maximum
  -> RGB이면 BT.709 luminance
  -> y = log1p(5000*x) / log1p(5000)
```

float target은 `known_scale` 또는 `already_normalized`를 명시해야 한다. NaN/Inf, 잘못된 dtype과
정규화 후 `[0,1]` 범위 위반은 조용히 보정하지 않고 실패한다. frame별 percentile은
`percentile_debug_only`와 `debug_only=true`를 함께 둔 비보고용 데이터 진단에서만 허용된다.

EventAid-R은 event block `i`와 GT `i+1`을 짝짓는 `target_offset=1`을 사용한다. 이는 이 과제를
위한 정렬 가정이며 bool·실수의 묵시적 정수 변환을 허용하지 않는다. full inspect는 event 원 timestamp
min/max와 `timestamps.txt` interval의 span ratio, offset, 범위 이탈 수를 기록한다. 공식 14 ZIP의
timestamp basis·단위가 확인되기 전에는 이 값을 hard rejection에 사용하지 않는다. 다른 offset 결과와
섞지 않는다. 두 dataset의 sensor response와 exposure가 같다는
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
EventHDR image dataset key는 `image<정수>` 형식과 numeric suffix uniqueness를 강제하고 숫자순으로
읽는다. 따라서 zero-padding이 다른 key에서도 문자열 정렬로 frame 순서가 뒤집히지 않는다.

## 3. graph와 모델 고정값

기본 graph node feature는 정규화한 `(x,y,t,polarity)`이고, 거리 계산은 `[0,1]`로 정규화한
`(x,y,t)` 3차원에서 한다. `graph_radius=0.08`보다 Euclidean 거리가 **엄격히 작은** 서로 다른 node를
연결하고 양 방향 directed edge를 저장한다. edge pseudo-coordinate는 `distance/radius` 한 값이다.

구현은 폭이 radius인 uniform cell과 인접 `3^3` cell로 후보를 찾은 뒤 exact Euclidean 조건을 다시
적용한다. 이는 근사 k-NN이나 edge truncation이 아니다. directed edge 수가
`max_graph_edges=2,000,000`을 넘으면 일부 edge를 버리지 않고 실패한다. isolated node도 유지하며
비율과 최대 degree를 결과에 기록한다.

graph의 incoming degree와 spline basis는 frame당 한 번 계산해 layer와 SNN timestep이 공유한다.
`spline_chunk_size=65536`은 edge message gather의 peak memory만 제한하며 graph 의미나 집계를
바꾸지 않는다.

고정 모델은 6-layer, hidden 64의 pure-PyTorch open degree-1 B-spline graph encoder, feature
rasterization, residual U-Net과 analog ConvGRU decoder다. ANN→SNN 변환과 IF timestep은 graph
encoder에만 적용된다. decoder는 두 모드 모두 analog이다.

## 4. 학습과 checkpoint 선택

40 epoch 전에 다음 CUDA 사전검증을 먼저 완료한다.

```bash
bash scripts/run.sh profile
```

`profile`은 EventHDR train의 모든 frame interval에서 `raw/cropped/retained/model-sampled events`,
radius cell search의 candidate pair, 실제 directed edge, 최대 incoming degree와 isolate ratio를 기록한다.
실제 edge 수로 정렬한 상위 10개를 `runs/profile.json`에 남기고, 기본 상위 3개에는 현재
model/loss/optimizer/AMP를 그대로 사용한 forward+backward와 optimizer step을 수행해 CUDA step time,
peak allocated/reserved VRAM을 측정한다. 전체 topology scan, edge guard, selected step, CUDA OOM-free가
모두 통과해야 `report_eligible=true`다.

학습 직전 verifier는 profile을 현재 public config, EventHDR train 전체 content SHA-256·transform·manifest,
source tree, PyTorch/CUDA/cuDNN과 GPU에 다시 결합한다. report가 없거나 하나라도 달라지면 본학습을
열지 않는다. 명시적 `--allow-unverified-preflight`는 합성·진단 학습만 허용하고 그 사실을
`preflight_gate.json`, public config와 checkpoint에 `report_eligible=false`로 영구 기록한다. `run.sh all`,
`scripts/train.sh` 기본 경로와 scheduler wrapper는 이 우회를 사용하지 않는다.

이 측정은 전수 topology scan과 **선택된 최고 밀도 표본의 실제 학습 step**을 결합한 경험적 gate다.
profile의 `measurement_scope.absolute_vram_guarantee=false`처럼, 전체 40 epoch 중 발생할 모든 allocator
상태나 미래 code/config까지 포괄하는 절대 VRAM 보증으로 해석하지 않는다.

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

training artifact는 `runs/train/`에 기록한다.

- `config.json`: host absolute path를 논리 label로 바꾼 shareable config
- `history.json`: epoch별 loss, 마지막 epoch validation, learning rate, CUDA peak memory
- `last.pt`: 매 epoch 끝에 저장하는 model/optimizer/scheduler/scaler/RNG 포함 재개 checkpoint
- `best.pt`: 현재 protocol에서는 마지막 epoch의 clean ANN inference checkpoint
- `preflight_gate.json`과 checkpoint의 `preflight_gate`: 검증된 CUDA profile identity
- `.data_hash_cache.json`: full source hash 계산을 가속하는 로컬 cache

새 학습은 위 핵심 artifact가 이미 있는 run directory를 덮어쓰지 않는다. 중단된 run은 `last.pt`로
재개한다.

## 5. 전체 ANN→SNN 보정

기본 본실험은 EventHDR train의 모든 frame을 사용한다.

```bash
bash scripts/calibrate.sh \
  configs/train.json \
  runs/train/best.pt \
  runs/train/best_snn.pt
```

wrapper 기본값 `CALIBRATION_SAMPLES=all`은 모든 training frame을 dataset index 순서로 정확히 한 번씩
선택한다. 더 작은 숫자를 지정한 기본 실행은 reporting seal을 만들지 않고 변환 전에 실패한다. model을 eval mode로
전환해 graph BN을 convolution parameter에 fold하고, non-empty sample의 layer별 feature-wise ReLU
maximum을 `calibration_activation_max`에 보존한 뒤 식 (6) parameter normalization과 unit threshold를
적용한다. 0인 dead channel은 `dead_channel_mask=true`로 남기고 실제 나눗셈에 쓰는
`normalization_scale`만 1로 둔다. 출력
`best_snn.pt`는 optimizer와 training history를 제거한 SNN inference checkpoint이며 calibration
표본 수·유효 표본 수·dead channel 수·sampling summary와 model tensor SHA-256을 포함한다.

보정은 clean `ann_inference`인 `best.pt`만 받는다. checkpoint의 학습 train content SHA-256,
transform, final split manifest, source tree/Git 계약과 terminal validation 완료 상태를 현재 EventHDR train과
대조해 모두 일치해야 `calibration_protocol.sealed=true`가 된다. source ANN의 검증된 CUDA preflight
gate도 seal에 포함되어 비보고용 profile 우회가 SNN 결과에서 사라질 수 없다. protocol은 source ANN
checkpoint와 model SHA-256, 전체 sample identity, data/transform/manifest, source와 runtime을 기록한다. CLI의
`--allow-unsealed-calibration`은 합성 테스트 전용이며 `sealed=false`와 불일치 이유를 영구 기록하므로
보고용 결과에 사용할 수 없다. override를 쓴 사실 자체가 taint이므로 전체 sample 실행도 봉인되지 않는다.
validation protocol v7은 train/validation dataset index의 전체 sample 수, group별 수, ordered identity
SHA-256과 source file fingerprint를 저장한다. 보고용 ANN은 두 sample cap이 `null`이고 full validation
sampling이 validation index commitment와 같아야 한다. SNN은 calibration identity/sampling을 train
commitment와 비교해 부분 calibration metadata의 사후 재라벨링을 거부한다.
또한 SNN model state의 persistent `calibration_attempts`, 32-byte
`calibration_commitment_digest`, `calibration_commitment_sealed`가 실제 시도 횟수와
protocol/count/sampling 및 valid/minimum/dead-channel summary core commitment를 보존한다. load 시
metadata summary와 layer별 `calibration_samples_seen`, raw maximum, effective scale, dead mask를
대조한다. normalization 뒤에도 raw/mask가 바뀌지 않아 dead channel checkpoint를 저장·strict reload할
수 있으며, 전체 metadata를 부분 보정 tensor에 이식하거나 mask/dead-channel 수만 바꿔도 거부된다.

기존 출력은 기본적으로 보호한다. 명시적으로 다시 보정해야 할 때만
`OVERWRITE_CALIBRATION=1`을 wrapper에 전달한다. engine은 새 checkpoint를 끝까지 만든 뒤 atomic
replace하므로 변환 실패 전에 기존 파일을 먼저 삭제하지 않는다. ANN 평가는 변환 전 `best.pt`, SNN
평가는 `best_snn.pt`를 써야 하며 서로 바꾸면 checkpoint 검증에서 거부된다.

## 6. 고정 평가·benchmark 행렬

`scripts/run.sh`는 full data/environment 검사와 모든 선택 sample의 decode 검증, GPU memory/topology
preflight, 학습 또는 재개, 전체 보정, 아래 행렬의 evaluate와 benchmark를 순서대로 실행한다.

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
model forward 안에 있으므로 측정에 포함된다. compute-only GPU peak memory에는 model과 event input은
포함하지만 품질 계산용 ground-truth target은 CPU에 유지해 제외한다.
quality evaluation은 `eval.max_samples=null`인 전체 dataset에서만 `report_eligible=true`가 될 수 있다.
cap이 설정된 부분 quality 실행은 기본 거부되며 명시적 합성-test 우회도 비보고용으로 영구 표시된다.
benchmark의 100 측정 frame은 품질 dataset cap이 아닌 별도 compute sampling이다.
두 실행 모두 underlying root에 explicit expected file count와 final EventHDR/fixed EventAid-R manifest가
필요하다. EventHDR는 현재 content SHA-256·transform·manifest가 source ANN validation 계약과 같아야 한다.

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

target, prediction, metric, graph diagnostic, latency와 `dt_us`는 저장 전에 finite/range 계약을
검사한다. 완전 일치의 PSNR은 문서화한 120 dB로 cap하며 strict JSON은 NaN/Infinity를 허용하지 않는다.
JSON/CSV/checkpoint는 unique temporary file을 flush한 뒤 atomic replace한다. 기본 quality config는
`precision=fp32`, `tf32=false`이고, 별도 `amp_fp16`/`bf16` run은 requested/effective precision,
autocast dtype, parameter dtype, device와 TF32 적용 여부를 artifact에 기록한다.

evaluate는 quality, model-forward latency, RTF/deadline miss, graph topology, dataset coverage와
CUDA peak allocated/reserved memory를 기록한다. benchmark는 mean/p50/p90/p95/p99/max latency, FPS,
raw/retained event rate, graph node rate, event retention, node/edge/isolated-node 수, SNN firing rate,
RTF, state reset과 peak GPU memory를 기록한다. benchmark의 target finite 검사는 CPU에서 수행하고
ground-truth target은 GPU로 옮기지 않으므로 peak GPU memory는 model과 GPU event input의 compute-only
범위다. target을 사용하는 quality evaluate는 기존대로 target을 model device에 올린다.

```text
runs/eval/hdr/ann/
runs/eval/hdr/snn_<literal_eq15|standard_if>_T<4|8|16|32>/
runs/eval/aid/ann/
runs/eval/aid/snn_<literal_eq15|standard_if>_T<4|8|16|32>/
```

각 directory의 `metrics.json`, `frames.csv`, `predictions/`는 evaluate가 만들고
`benchmark.json`은 benchmark가 만든다. prediction은 config 기본값에 따라 처음 20 frame의 pred/GT
PNG를 저장한다. 같은 mode/dynamics/T artifact가 있으면 묵시적으로 덮어쓰지 않고 실패하므로 재실행
전 기존 결과를 보존 위치로 옮기거나 별도 `eval.output_dir` config를 사용한다.

보고용 ANN 평가에는 검증된 preflight gate를 가진 clean `ann_inference`, finite final selection과
training/validation protocol이 필요하다. 보고용 SNN에는 `calibration_protocol.sealed=true`인
`snn_inference`가 필요하다. `metrics.json`의 `evaluation_protocol`과 `benchmark.json`의
`benchmark_protocol`은 public config, model config, checkpoint file/tensor와 ANN training/validation 또는
SNN calibration lineage, 현재 eval data content·transform·manifest·coverage·sampling, source,
runtime과 precision을 각각 canonical SHA-256에 결합한다. 18-run 연속 실행의 대용량 파일 hash는 host
path를 저장하지 않는 path-token cache로 재사용하며 stat signature가 달라지면 파일을 다시 읽는다.
SNN checkpoint를 열 때 calibration transform·final manifest·전체 selected identity/sampling·runtime,
source ANN의 training data와 source 계약도 다시 검증하므로 해당 field의 삭제나 일관된 digest 재작성도
기본 보고 평가를 통과하지 못한다.
source ANN의 model/epoch/selection/terminal-validation 상태를 묶은 reporting contract도 calibration 때
hash하고 SNN 평가 때 다시 실행 검증한다. 단, 내부 SHA-256은 전자서명이 아니므로 최종 checkpoint와
결과 file hash는 별도 immutable log 또는 서명 archive에도 기록해야 한다.
EventHDR source ANN은 계획 epoch에서 완료된 `single_final_epoch`만 보고 가능하며, 반복 검증의
`best_validation_macro_ssim` 경로는 다른 항목이 모두 유효해도 거부한다. planned/completed/checkpoint
epoch가 같고 selection rule이 `single_final_epoch_macro_ssim`이어야 한다.
source ANN dataset type은 EventHDR로 고정하며 training protocol의 seed·optimizer·scheduler·loss·
data order·mixed precision·runtime을 public config와 교차검증한다. reporting protocol은 top-level
checkpoint lineage뿐 아니라 실제 inference mode, SNN simulation step T와 effective dynamics도 hash한다.

합성 fixture용 `--allow-unsealed-checkpoint-for-non-reporting`은 누락·불일치 checkpoint를 진단할 수
있게 하지만 결과에 `report_eligible=false`와 이유를 영구 기록한다. 이 옵션은 public shell/scheduler
wrapper에서 사용하지 않으며 해당 산출물을 표에 포함하면 안 된다.

## 8. epoch-boundary exact resume

```bash
RESUME_CHECKPOINT="$PWD/runs/train/last.pt" \
  bash scripts/run.sh train
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
- 검증된 CUDA preflight report와 현재 config/data/source/runtime의 동일성
- final-only validation의 처음 계획한 terminal epoch와 완료 상태

따라서 다른 GPU 종류, CUDA/PyTorch 조합, source checkout, worker protocol이나 dataset byte로 옮긴
checkpoint는 “exact” 재개로 허용되지 않는다. 절대 data mount path와 mtime/ctime 자체는 checkpoint
protocol에 넣지 않는다. 같은 상대 파일과 byte를 다른 mount에 복사하면 다시 full hash한 뒤 일치할
수 있다. `.data_hash_cache.json`은 absolute path 자체가 아니라 path SHA token을 local key로 쓰며,
같은 source의 size/mtime/ctime이 모두 같을 때만 기존 full hash를 재사용한다. 기본
`train.rehash_data=true`는 첫 sealed run에서 cache를 무시하고 51+19 H5를 다시 읽는다.

`validate_every=null`이면 training protocol에 `planned_epoch=40`을 포함하고 checkpoint가 terminal
validation 완료 여부와 epoch를 봉인한다. 40 epoch 최종 평가를 이미 수행한 `last.pt`에 epochs만 늘려
같은 run에서 eval을 다시 보는 것은 거부한다. 연장 학습은 새 output directory와 새 protocol로 시작한다.

## 9. 과학적 한계와 중단 조건

- 공개된 저자 공식 코드가 확인되지 않아 논문에 없는 graph normalization, spline hyperparameter,
  threshold 세부값은 이 저장소의 명시적 가정이다. “공식 ASGCN 완전 재현”으로 표현하지 않는다.
- 원 ASGCN의 classification pooling/MLP와 dynamic asynchronous K-hop update는 구현하지 않았다.
  현재 graph는 frame interval마다 정적으로 다시 만든다.
- residual U-Net, rasterization, analog ConvGRU, 복원 loss, tone mapping, 정확한 `max_events` cap과
  EventAid-R offset은 과제용 확장이다.
- graph encoder만 IF로 변환된다. PyTorch GPU latency·발화율은 FPGA/ASIC latency, power 또는 energy
  측정이 아니며 event 전송·압축 protocol이나 RTL도 구현하지 않았다.
- CUDA profile은 선택한 최고 밀도 표본에서의 empirical peak measurement이지 전체 학습의 절대 메모리
  보증이 아니다. A100/A6000 각각에서 실제 `runs/profile.json`이 없으면 해당 GPU 통과를 주장하지 않는다.
- EventHDR 공식 train/eval의 물리 scene 독립성을 입증하지 못했으므로 내부 결과에 독립 test-set
  일반화 주장을 붙이지 않는다. EventAid-R도 sensor/domain 차이를 통제한 benchmark는 아니다.
- `literal_eq15`의 self-feedback은 표준 rate conversion과 수학적으로 동등하지 않을 수 있다.
  dynamics와 T를 생략한 ANN↔SNN 비교는 보고하지 않는다.
- file coverage/HDF5·ZIP decode 실패, non-finite loss/metric, 비단조 timestamp, 좌표 범위 오류,
  `max_graph_edges` 초과 또는 OOM이 발생하면 해당 sample을 조용히 제외하지 말고 run을 폐기해 원인과
  설정을 고친 뒤 새 artifact로 실행한다.
