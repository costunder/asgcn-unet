# ASGCN-U-Net의 ASGCN paper-core 구현 범위

이 저장소는 AAAI 2025 ASGCN 논문의 공개 수식에서 확인할 수 있는 event graph와 ANN→SNN
변환 핵심을 구현한 뒤, event-to-frame 복원용 decoder를 연결한 연구 코드다. 원 논문은 event
classification을 다루지만 이 프로젝트의 출력은 luminance frame이다. 따라서 이 코드는 저자 공식
구현, 원 논문의 classification pipeline, 또는 공식 성능표의 완전 재현본이 아니다.

근거로 삼은 공개 자료는 [AAAI 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32154)와
[공식 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)다. 확인 가능한 논문에는
저자 코드와 checkpoint 링크가 없으며, 복원 과제에 필요한 graph·spline·decoder 설정도 모두
주어지지 않는다. 문서와 결과에서는 반드시 `ASGCN paper-core 기반 복원 적응` 또는 `공개 수식
기반 구현`으로 표현하고, `저자 공식 코드`, `공식 ASGCN 완전 재현`, `논문 성능 재현`이라고
표현하지 않는다.

## 1. 논문 수식에서 가져온 core

현재 `architecture_version=2`의 graph encoder가 다음 항목을 구현한다.

1. event sequence를 고정 정수 sampling factor `R`로 균일하게 선택한다.
2. event를 node로 만들고, 좌표 거리 조건 `d(i,j) < D`를 만족하는 node 쌍을 잇는다.
3. scalar edge distance를 pseudo-coordinate로 쓰는 weighted B-spline graph convolution과 실제
   incoming degree 기반 mean aggregation을 적용한다.
4. ANN 경로에서는 graph affine update, BatchNorm, ReLU를 순서대로 실행한다.
5. ANN 학습 뒤 식 (13)–(14)의 BatchNorm folding과 식 (6)의 layer-wise parameter
   normalization을 적용한다.
6. SNN 경로에서는 별도 stochastic/rate input encoder 없이 IF membrane을 명시적 timestep으로
   전개한다. 초기 membrane은 threshold의 절반, spike amplitude는 threshold이며 발화 뒤 soft
   reset을 사용한다.

원 논문의 graph clustering, graph pooling, edge remapping과 classification head는 구현하지 않았다.
이 프로젝트는 graph encoder의 node feature를 raster로 바꿔 복원 decoder에 전달한다.

## 2. 논문에 없는 명시적 가정

논문만으로 결정할 수 없는 값은 config에 노출했고 checkpoint의 `model_config`에 보존한다. 기본
본실험 설정은 다음과 같다.

| 항목 | 저장소의 선택 | 해석 |
|---|---:|---|
| node feature | `[x,y,t,p]` | `x,y,t`는 `[0,1]`, polarity는 `-1/+1` |
| graph distance | 정규화된 `x,y,t` 3차원 | polarity는 기본 거리에서 제외 |
| radius | `D=0.08` | 원 논문의 복원 과제용 공식값이 아님 |
| edge pseudo | `u=d/D` | 원시 distance를 SplineConv의 `[0,1]` 정의역으로 재매개화 |
| graph width/depth | 64 features, 6 layers | 복원 과제 가정 |
| spline | scalar, open, degree 1, `K=5` | degree/kernel/open 여부는 논문 미공개 |
| update terms | mean message + root transform + bias | root/bias는 PyG 계열 동작을 참고한 선택 |
| paper sampling `R` | `event_sampling_factor=1` | dataset의 8,192-event cap과 별개 |
| raster | 4배 downsample grid에서 cell mean | 논문에 없는 event-to-frame bridge |
| decoder | base 48 residual U-Net + analog ConvGRU | 논문에 없는 복원 확장 |
| SNN dynamics | `literal_eq15` | 공개 식을 우선한 선택; 아래 모호성 참조 |
| edge guard | directed edge 2,000,000개 | 초과 시 graph를 자르지 않고 실패 |

dataset loader의 `max_events=8192`는 논문의 `R`이 아니다. crop 뒤 event가 cap을 넘으면
`np.linspace(0, N-1, 8192)`로 정확히 8,192개를 시간축 전체에서 결정적으로 선택하며 양 끝 event를
포함한다. 그 뒤 model의 고정 factor `R`이 `events[::R]`로 적용된다. 결과에는 raw/cropped/retained
event 수, dataset sampling ratio, model factor와 두 비율의 곱을 따로 기록한다.

## 3. exact cell radius graph

`build_radius_graph`는 dense `N×N` distance matrix를 만들지 않고 uniform-cell candidate search를
사용하지만, 만들어지는 graph의 의미는 brute-force strict radius graph와 같다.

- 선택한 `d`차원 좌표에서 cell 폭을 정확히 `D`로 둔다.
- 각 source node의 cell과 축별 offset `{-1,0,1}`의 조합인 `3^d` 인접 cell만 검색한다. cell 폭이
  `D`이므로 그 밖의 cell에는 `d < D`인 이웃이 존재할 수 없다.
- sorted cell hash의 범위를 `searchsorted`로 찾아 후보를 만들고, 모든 후보에 대해 Euclidean
  distance를 다시 계산한다.
- `source != destination`와 `distance < D`를 모두 만족할 때만 edge를 남긴다. 경계 `distance=D`는
  포함하지 않는다.
- 모든 node를 source로 처리하므로 각 유효한 무방향 쌍의 두 ordered direction을 모두
  materialize한다. self-loop와 중복 edge는 없다.
- 최종 edge는 `(source,destination)` 순서로 정렬하며 edge attribute는 `distance/D`다.

`graph_chunk_size=512`와 내부 candidate chunk 축소는 계산을 나누는 메모리 최적화일 뿐 neighbor를
근사하거나 누락하지 않는다. 다만 한 cell에 event가 몰리는 최악의 경우 후보 수와 실제 edge 수는
여전히 `O(N²)`가 될 수 있다. `max_graph_edges`는 이 경우 조용히 edge를 버리지 않고 오류를 내는
fail-fast 장치다. radius가 만든 graph를 강제로 연결하지 않으므로 isolated node는 root transform과
bias만 받으며 isolate ratio와 maximum degree가 결과에 기록된다.

## 4. degree-1 open B-spline과 계산 최적화

scalar pseudo-coordinate `u`에 대해 `scaled=u(K-1)`를 계산한다. degree 1이므로 edge마다 활성
basis는 인접한 두 control point뿐이고, 가중치는 `1-frac(scaled)`와 `frac(scaled)`다. 두 가중치의
합은 endpoint를 포함해 1이다. `u=1`에서는 마지막 control point의 가중치가 정확히 1이 되도록
endpoint 동작과 pseudo-coordinate gradient를 테스트로 고정했다.

각 `PaperSplineConv` layer의 계산은 다음과 같다.

```text
node projection for every control point: [N,Cin] x [K,Cin,Cout] -> [N,K,Cout]
edge gather: source node의 활성 control point 두 개만 선택
weighted message: 두 basis weight로 합산
aggregation: destination의 실제 incoming degree로 평균
update: mean message + optional root transform + bias
ANN activation: BatchNorm -> ReLU
```

edge마다 작은 matrix multiplication을 반복하지 않고 node를 `K`개 control point에 한 번씩 projection한
뒤 두 활성 항만 gather한다. 고정 graph의 basis index와 weight도 sample당 한 번 계산해 6개 graph
layer와 모든 IF timestep에서 재사용한다. 이는 연산 중복을 줄이는 exact optimization이며 graph나
spline 값을 바꾸지 않는다. 구현은 순수 PyTorch라 `torch-spline-conv` binary extension에
의존하지 않는다.

`EventGraph`의 destination incoming degree도 graph 생성 시 한 번 계산해 모든 layer와 IF timestep이
공유한다. 기본 `spline_chunk_size=65536`은 최대 2,000,000개 edge의 message gather를 고정 크기
chunk로 나눠 peak memory를 제한한다. chunk마다 같은 순서로 `index_add_`하므로 neighbor나 edge를
줄이는 근사가 아니며, chunked/unchunked 출력과 gradient 동등성을 회귀검사한다.

weight 초기화 bound는 `1/sqrt(K*Cin)`, root bound는 `1/sqrt(Cin)`으로 고정했다. open degree 1,
scalar pseudo-coordinate, mean aggregation, root weight와 bias 범위에서만 구현·테스트했으며 이를
ASGCN 저자의 미공개 hyperparameter와 동일하다고 주장하지 않는다.

## 5. ANN 학습과 ANN→SNN 변환

ANN graph layer는 B-spline affine update, BatchNorm, ReLU로 학습한다. 기본 optimizer는 Adam에
gradient centralization을 추가한 `adam_gc`이며 learning rate `1e-3`, weight decay `5e-3`, epoch
20/30의 MultiStepLR과 gamma `0.1`을 사용한다. milestone과 gamma, tensor별 centralization 축은
저자 코드로 검증된 공식값이 아니라 config 선택이다.

변환 순서는 다음과 같다.

1. 학습된 ANN checkpoint를 불러온다.
2. 각 graph layer의 BatchNorm scale을 spline kernel, root와 bias에 fold한다.
3. EventHDR train 전체에서 각 graph layer의 feature별 ReLU maximum `lambda_l`를 측정한다.
4. 식 (6)에 따라 kernel과 root에 `lambda_(l-1)/lambda_l`, bias에 `1/lambda_l`를 적용한다.
5. 첫 layer의 입력 scale `lambda_0`은 `[1,1,1,1]`로 둔다.
6. calibration에서 항상 0인 channel은 raw maximum 0과 dead mask를 보존하고, epsilon으로
   폭증시키지 않도록 effective normalization scale만 1을 사용한다.
7. 변환 뒤 threshold를 정확히 1로 두고, 마지막 spike rate에 `lambda_L`를 곱해 analog decoder의
   학습 단위로 복원한다.

각 layer는 관측 raw maximum `calibration_activation_max`, 식 (6)의 effective
`normalization_scale`, `dead_channel_mask`를 별도 persistent buffer로 저장한다. BN-fold/normalization
flag, valid calibration count, dead-channel summary, threshold, model tensor SHA-256과 checkpoint
metadata를 load 시 교차검증한다. 따라서 dead channel이 있어도 저장·strict reload 뒤 raw summary가
바뀌지 않으며 mask·scale·metadata 변조는 거부된다. ANN checkpoint와 변환된 SNN checkpoint의
inference mode를 서로 바꿔 사용하는 것도 거부한다.

## 6. IF dynamics와 식 (15)의 모호성

논문 식 (15)는 다음 self-feedback 항을 포함한다.

```text
v_tilde(t) = v(t-1) + c(t) + h(t-1)
```

`literal_eq15`는 이 `+h(t-1)`을 문자 그대로 실행한다. 각 timestep에서 threshold 이상이면 threshold
크기의 spike를 내고 membrane에서 그 값을 빼는 soft reset을 한다. 그러나 이전 spike의 재주입은
작은 양의 정전류도 첫 발화 뒤 장기 firing rate 1에 가깝게 만들 수 있어 표준 ANN rate-conversion
유도와 충돌한다. 저자 코드나 정정 자료가 없으므로 임의로 오타 처리하지 않았다.

- `literal_eq15`: 기본 경로. 공개 식 (15)–(17)을 문자 그대로 실행한다.
- `standard_if`: `+h(t-1)`을 제거한 대조군이다. 공식 ASGCN 설정이 아니다.

마지막 `lambda_L` 곱은 decoder 입력의 단위 변환이지, `literal_eq15`가 유한 timestep에서 ANN과
동등하다는 증명이 아니다. 두 dynamics는 같은 calibrated checkpoint로 비교할 수 있지만 결과는
반드시 별도 dynamics와 timestep으로 표기한다.

## 7. hybrid event-to-frame decoder

전체 forward는 다음과 같다.

```text
event interval [N,x,y,t,p]
  -> exact-size dataset cap
  -> fixed paper sampling factor R
  -> strict undirected radius graph
  -> B-spline graph encoder (ANN 또는 explicit IF-SNN)
  -> downsampled feature raster: pixel-cell별 node feature mean
  -> residual U-Net encoder
  -> bottleneck analog ConvGRU
  -> bilinear upsampling + skip connections
  -> sigmoid luminance frame, 원 sensor size로 interpolation
```

U-Net은 두 번 downsample하고 bottleneck에 residual block 두 개를 둔다. ConvGRU state는 같은 H5/ZIP
sequence group, 연속 sequence index, 같은 sensor shape일 때만 전달되고 불연속에서는 초기화된다.
학습 시 state와 이전 prediction은 frame마다 detach하므로 전체 sequence backpropagation이 아니라
frame 단위 truncated recurrence다. SNN 변환 대상은 graph encoder뿐이며 rasterization, U-Net,
ConvGRU와 output head는 모두 analog 연산이다.

EventHDR의 zero-event target interval도 버리지 않는다. 빈 event tensor는 zero-node/zero-edge graph와
zero raster를 만들고 recurrent decoder가 해당 target frame을 학습·평가하도록 한다. `frame_stride>1`을
사용하면 건너뛴 interval의 event를 다음 선택 frame까지 합치지만 기본 본실험은 `frame_stride=1`이다.

## 8. 구현하지 않은 범위와 주장 한계

- 새 event에 영향받는 K-hop subgraph만 갱신하는 asynchronous incremental 실행
- sliding window node 만료와 graph/SNN state의 하드웨어 친화적 관리
- 논문 식 (18)–(19)의 clustering, pooling, edge remapping
- 원 논문의 classification MLP, 원 데이터셋 전처리와 성능표
- 실제 DVS sensor ingest, event compression·전송 protocol
- FPGA/ASIC RTL, synthesis, 실제 latency·전력·에너지 측정
- analog U-Net/ConvGRU까지 포함한 완전한 spiking network

현재 구현으로 보고할 수 있는 것은 PyTorch GPU/CPU에서의 paper-core 기반 복원 품질, graph 통계,
compute latency와 firing-rate 통계다. 이를 저자 공식 재현, neuromorphic hardware latency, 반도체
전력·에너지 또는 통합 칩 구현 완료로 확대해 해석하면 안 된다.
