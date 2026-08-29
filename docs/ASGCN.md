# ASGCN 구현 범위

이 저장소의 graph core는 AAAI 2025 논문에 공개된 ASGCN 수식 중 정적 graph/ANN→SNN core를
직접 구현하지만,
저자 공식 코드를 그대로 실행한 **완전 재현본**은 아니다. 2026-08-29 기준 공개 배포된 저자
코드는 확인할 수 없었다. 확인 가능한
[AAAI 공식 논문 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32154)와
[공식 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32154/34309)에는 저자 코드 링크가
없고, 아래에 적은 구현 세부값 일부도 논문에 명시되지 않았다. 따라서 논문으로 확인되는 부분과
이 저장소의 가정·과제용 확장을 분리해 기록한다.

## 논문에 정의된 graph/SNN core

- 입력 event를 sampling factor `R`로 균일하게 줄인다.
- event 좌표 사이의 유클리드 거리가 `D`보다 작은 두 node를 잇는 simple undirected radius graph를
  만들고, scalar distance를 edge pseudo-coordinate로 쓴다.
- 식 (11)은 실제 이웃 수로 평균한 weighted tensor-product B-spline message aggregation을,
  ANN update는 ReLU를 정의한다.
- 식 (6)은 layer-wise lambda를 이용한 ANN parameter normalization을 정의한다. 별도 threshold
  설명은 feature dimension별 maximum activation을 사용한다.
- batch normalization은 식 (13)–(14)에 따라 convolution parameter에 fold한다.
- SNN 경로는 별도 rate encoding 없이 IF membrane을 timestep마다 전개한다. 초기 membrane은
  `theta/2`, 발화량은 `theta`, 발화 뒤에는 soft reset을 적용한다. 기본
  `snn_dynamics=literal_eq15`는 식 (15)의 `+h_i^l(t-1)`까지 문자 그대로 실행한다.
- 학습 optimizer는 Adam과 gradient centralization을 사용하며, 초기 learning rate는 `1e-3`, L2
  weight decay는 `5e-3`이다. 논문은 milestone decay를 명시하지만 정확한 epoch와 gamma는 공개하지
  않는다.

`architecture_version: 2` checkpoint만 이 의미론을 나타낸다. 현재 공통 설정은
`graph_operator: spline`, `spline_backend: torch`,
`spline_pseudo: distance_over_radius`다. 마지막 이름은 논문이 말한 원시 거리값 자체가 아니라
SplineConv 정의역에 맞춘 `distance / D` 재매개화임을 의도적으로 드러낸다.

## 논문에 없는 구현 가정

논문만으로 단일한 정답을 정할 수 없는 값은 config에 노출했다. 기본값은 다음과 같다.

| 항목 | 이 저장소의 명시적 가정 |
|---|---|
| graph position | `x, y, t`를 각각 `[0,1]`로 정규화한 3차원 좌표 |
| polarity node feature | 음극성 `-1`, 양극성 `+1` |
| graph radius/pseudo | 정규화 좌표에서 `D=0.08`, scalar distance를 `D`로 나눠 `[0,1]` pseudo-coordinate로 사용 |
| B-spline/update | open degree-1 spline, kernel size 5, 별도 root weight·affine bias와 현재 ReLU/BN 적용 순서 |
| graph width/depth | hidden feature 64, graph layer 6; 논문이 복원 과제용 값을 제공하지 않아 설정값으로 고정 |
| paper sampling | model의 고정 `event_sampling_factor=1`로 기본 비축소 |
| adaptive safety cap | crop 뒤 `ceil(N/max_events)` 정수 stride로 먼저 줄이는 reconstruction/server 가정이며 논문의 공식 `R`이 아님 |
| graph construction | 결과를 바꾸지 않는 512-node chunk 계산 |
| ANN→SNN scale | layer-wise lambda와 feature-wise threshold 결합이 모호해 feature-wise lambda와 정규화 뒤 unit threshold 사용 |
| first-layer scale | lambda^0은 `[1,1,1,1]`; 이후 lambda는 calibration activation maxima. `[0,1]`의 `x,y,t`와 `±1` polarity를 근거로 한 저장소 선택이며 논문에는 없음 |
| dead activation channel | calibration ReLU maximum이 0인 feature는 epsilon으로 나누지 않고 lambda 1을 사용하며, 변환 전 dead-channel 수를 checkpoint metadata에 기록 |
| gradient centralization axes | spline `[K,Cin,Cout]`와 root `[Cin,Cout]`는 마지막 output axis 외 차원, Conv/Linear는 첫 output axis 외 차원에서 평균을 제거; 저자 코드로 확인되지 않은 tensor-layout 선택 |
| edge memory guard | directed edge가 2,000,000개를 넘으면 자르지 않고 실패; 8,192-node 밀집 graph의 O(N²) OOM을 막는 서버 안전장치 |
| connectivity | `D`가 만든 graph를 강제로 연결하지 않고 평가 결과에 isolated node 비율과 max degree를 기록 |

실제 전처리 순서는 spatial crop → adaptive `max_events` stride → model의 고정 paper sampling factor
`R`이다. 두 factor와 raw/cropped/retained count를 분리해 기록하며 adaptive cap을 ASGCN 논문의 공식
sampling 설정으로 보고하지 않는다.

순수 PyTorch spline 구현은 Linux 서버 설치를 단순하게 하기 위한 선택이다. B-spline
pseudo-coordinate의 `[0,1]` 범위, open spline, degree, kernel과 root-weight 개념은
[PyTorch SplineConv 공식 연산자 소스](https://github.com/rusty1s/pytorch_spline_conv)를 참고했다.
지원하는 scalar/open/degree-1 범위에서는 endpoint index·pseudo gradient와 weight/root 초기화 bound도
공식 연산자 동작에 맞추고 회귀 테스트로 고정했다. 고정 graph의 basis/index는 한 번만 계산해 모든
layer와 IF timestep에서 재사용한다.
이는 ASGCN 저자 코드와 동일하다는 증거가 아니며, 논문에 없는 kernel size·degree·root 설정을
공식값으로 오인하면 안 된다.

### 식 (15)와 rate conversion의 공개 모호성

논문 식 (15)는 `v_tilde(t)=v(t-1)+c(t)+h(t-1)`로 적혀 있다. 이 `+h(t-1)` self-feedback을
문자 그대로 실행하면 작은 양의 정전류도 첫 발화 뒤 firing rate 1에 가까워질 수 있어, 표준
soft-reset IF의 `firing rate ≈ normalized ANN activation` 유도와 양립하지 않는다. 저자 코드나
인덱스 설명이 없어 이를 임의로 오타 처리하지 않았다.

- `literal_eq15`: 저장소 기본값. 공개 식 (15)–(17)을 그대로 실행한다.
- `standard_if`: `+h(t-1)`를 제거한 rate-conversion 대조군이며, 공식 ASGCN 값이라고 주장하지 않는다.

Eq. (6) 뒤 마지막 `lambda_L`를 곱하는 단계는 spike 출력을 학습된 analog decoder의 단위로 보내기
위한 차원 변환이다. `standard_if`에서는 보정 ANN activation과의 parity test가 있지만,
`literal_eq15`에서 유한 T의 ANN-rate 동등성을 보장한다는 뜻은 아니다. 두 dynamics의 장기-T 차이를
단위 테스트로 고정하고 checkpoint의 model config에 선택값을 보존한다.

## Event-to-frame 과제용 확장

원 논문은 event graph를 pooling한 뒤 MLP로 분류한다. 이 저장소는 분류기를 재현하는 대신
EventHDR에서 luminance frame을 복원하도록 다음 모듈을 연결한다.

```text
uniform event sampling
  -> undirected radius graph
  -> B-spline graph encoder (ANN 또는 literal-Eq15/standard-IF timestep)
  -> feature rasterization
  -> residual U-Net + analog ConvGRU
  -> luminance frame
```

따라서 residual U-Net, ConvGRU, rasterization, EventHDR/EventAid-R loader, tone mapping,
Charbonnier·SSIM·gradient·temporal loss는 ASGCN 논문의 구성요소가 아니라 연구과제에 맞춘
재구성 확장이다. SNN 전환은 graph encoder에만 적용되고 decoder와 ConvGRU는 analog 연산이다.

## 아직 구현하지 않은 논문·하드웨어 범위

- 새 event가 들어올 때 영향받는 K-hop ego-network만 갱신하는 동적 asynchronous 실행
- 논문 식 (18)–(19)의 node clustering, graph pooling과 edge remapping
- 원 논문의 classification MLP와 N-MNIST/CIFAR10-DVS/N-Caltech101/N-Cars 성능표 재현
- 논문의 연산 수 기반 energy 추정 및 실제 FPGA/ASIC 전력·에너지 측정
- event compression·transport protocol과 반도체 RTL

현재 코드는 논문의 공개 정의에 맞춘 **ASGCN paper-core 기반 event-to-frame 연구
프로토타입**이다. 누락된 저자 구현 세부정보와 다른 출력 과제 때문에 “공식 ASGCN 완전 재현”이나
원 논문 성능 재현으로 표현하지 않는다. 해당 주장은 저자 코드·동일 데이터 전처리·분류 head·공식
checkpoint를 확보해 parity를 검증한 뒤에만 가능하다.
