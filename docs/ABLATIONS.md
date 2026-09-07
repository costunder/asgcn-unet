# 구조 비교 실험

이 코드는 기존 `runs/fast` 실험과 분리된 신규 학습·보정·평가를 실행한다.
기존 ANN/SNN 결과에 이름만 바꿔 새 비교군으로 사용하지 않는다.

| 주 비교군 | 학습 family | 인코더 → 복원기 | 평가 |
| --- | --- | --- | --- |
| A | `unet` | 정규화된 원본 이벤트 특징 → 평균 raster → 기존 recurrent U-Net | ANN |
| B | `pointwise_unet` | 그래프 이웃 전달 없는 6층 pointwise 발화 인코더 → 기존 recurrent U-Net | SNN + ANN 대조군 |
| C | `graph_unet` | 기존 6층 일반 GNN → 기존 recurrent U-Net | ANN |
| D | `graph_unet` | 동일 가중치의 보정된 6층 Spiking GNN → 기존 recurrent U-Net | SNN |
| E | `transformer` | A와 동일한 정규화 원본 이벤트 특징 → 평균 raster → recurrent Transformer 복원기 | ANN |

4개 ANN 학습이 A–E를 구성한다. B, D의 SNN은 각각 자기 ANN을 전체
EventHDR 학습 프레임으로 보정한다. 그래프 없는 SNN도 학습된 pointwise
인코더의 발화 변환이며, 빈 그래프를 넣고 기존 GNN이라고 부르는 구현이 아니다.
두 데이터셋 모두 네 family의 ANN을 평가하고 B/D에만 `literal_eq15`,
`standard_if` × `T=4,8,16,32`를 추가한다. 각 조건의 compute-only benchmark도
별도로 실행한다. A/E에는 GNN·SNN 인코더가 없으며 SNN 보정·T·발화 동역학
실험을 추가하지 않는다. 총 4개 전체 학습, 2개 전체 보정, 40개 품질 평가와
40개 benchmark(두 데이터셋 합계)이다.

주 비교는 **A↔E: 같은 입력의 U-Net 단독 대 Transformer 단독**이다.
E는 GNN 뒤에 Transformer를 붙인 모델이 아니다. 이전
`configs/ablations/graph_transformer-*.json`과 `runs/ablations/graph_transformer`
경로가 있다면 삭제하거나 E로 재해석하지 않는다. 이 이름은 보존된 이전 설계이며
현재 기본 계획·실행 family·요약표에서 제외된다. E는 반드시 별도
`configs/ablations/transformer-{train,hdr,aid}.json`과
`runs/ablations/transformer`를 사용해 새로 학습한다.

## 유지하는 계약과 해석상의 한계

`configs/ablations/{family}-{train,hdr,aid}.json`은 기존 `fast.json`,
`hdr-fast.json`, `aid-fast.json`의 전체 규모 설정을 보존한다. seed=2026,
40 epochs, physical batch=16, 6 encoder layers/hidden=64(존재하는 인코더),
decoder base=48, 기존 이벤트 8192개 제한과 sampling factor=1, 입력 전체 해상도,
학습 51 H5/평가 19 H5/EventAid-R 14 ZIP, loss, optimizer, learning-rate schedule,
validation/calibration/eval의 batch/worker 후보가 그대로다. 이 값들은 기존
실험 계약이며 새로 추가한 축소가 아니다. 출력만 `runs/ablations/{family}`에 둔다.
인코더가 없는 A/E에서 존재하지 않는 6층/hidden=64를 사용했다고 보고하지 않는다.
두 모델 모두 같은 원본 `x,y,t,p` 4채널을 같은 위치에 평균 rasterization하여 입력한다.
실제 모델의 parameter 수와 활성 모듈은 실행 전 구조 로그를 따른다.

Transformer는 U-Net과 같은 기본 channel 수와 recurrent 조건을 유지하고,
전체 공간을 window attention으로 처리한다. 설정은 depths `[1,1,2,1,1]`,
heads `[3,6,12,6,3]`, window=8, MLP ratio=4.0이다. 48/96/192 channel 폭에
각 head dimension=16을 배정하고, bottleneck에 두 attention block을 둔다.
이는 명시적인 신규 비교 설계이며 논문의 Transformer 재현이나 parameter-matched
모델이라고 주장하지 않는다. window 경계와 padding은 모델 코드의 mask를 따른다.
Transformer 단독의 의미는 GNN/SNN 인코더와 U-Net 복원 블록을 쓰지 않는다는 것이다.
두 복원기 모두 기존 ConvGRU 시간 기억 조건을 유지한다. 입력 투영·해상도 복원 연산까지
전부 attention 연산이라는 주장은 하지 않는다. 이 범위 변경에서 attention mask나
모델 해상도를 줄이는 별도 최적화는 하지 않는다.

체크인된 실제 4개 train config를 CPU에서 구성했을 때 전체 모델 parameter 수는
다음과 같다. 학습이나 GPU 실행 없이 구조만 구성해 센 값이며, 결과 요약은
이 표를 하드코딩하지 않고 실제 평가 보고서의 parameter 수를 읽는다.

| Family | 전체 parameters |
| --- | ---: |
| `unet` | 4,258,129 |
| `pointwise_unet` | 4,305,937 |
| `graph_unet` | 4,409,617 |
| `transformer` | 3,380,443 |

- A↔E는 원본 `x,y,t,p` 평균 raster, 입력 해상도, 시간 기억, 데이터 분할과
  학습 조건을 맞춘 주 비교다. U-Net과 Transformer의 공간 처리 구조 및
  parameter 수는 다르며, parameter-matched 비교라고 주장하지 않는다.
  A/E는 E2VID 또는 특정 voxel-grid baseline의 재현이 아니다.
- B/D↔A는 학습된 인코더/그래프를 추가하는 보조 비교다. 원본 특징과 학습 특징의
  차이, 인코더 용량 차이를 함께 포함하므로 순수 복원기 비교와 구분한다.
- B↔D는 동일 이벤트/node 전처리에서 이웃 message passing을 제거하는 비교다.
  parameter 수는 같지 않으므로 이를 반드시 함께 보고한다.
- C↔D는 자기 학습 ANN에서 발화 변환·T·동역학을 바꾸는 paired 비교다.
- D↔E는 그래프/발화 인코더와 복원기가 모두 달라진다. 이를 Transformer만의
  효과로 해석하지 않으며, 그 효과의 주 비교는 A↔E다.
- EventAid-R test 결과를 보고 T나 구조를 골랐다면 그 선택은 탐색적 결과로
  구분한다. 최종 선택 기준은 학습/validation에서 고정해야 한다.
- 현재 단일 seed 실험이다. 추가 반복은 별도 승인된 seed 및 독립 경로가 필요하며,
  이 코드는 통계적 유의성·여러 번 재현 성공을 주장하지 않는다.

## 실행

먼저 코드가 서버에 실제 반영된 상태에서 저장소 루트에서 계획을 확인한다.
이 단계는 CUDA 초기화, 데이터 읽기, 결과 쓰기를 하지 않는다.

```bash
python -B scripts/run_ablations.py --stage plan
```

실제 할당된 GPU 식별자는 기존 스케줄러/컨테이너 환경 그대로 보존한다.
이 스크립트는 임의의 GPU 번호를 선택하거나 `CUDA_VISIBLE_DEVICES`를 변경하지 않는다.
할당 검사가 미확인으로 중단됐다면 현재 할당을 확인하기 전에는 실행하지 않는다.
과거 실험의 GPU 번호를 재사용하거나 안전 검사를 우회하지 않는다. 현재 작업에 실제로
할당된 동일 mask/장치 제한과 Conda 환경을 유지한 셸에서 다음 검사를 먼저 통과시킨다.

```bash
python -B scripts/run_ablations.py --stage all --families unet transformer --execute
```

위 명령은 주 비교인 **A/E 각각 40 epoch 학습 + 두 데이터셋 4개 평가 및
4개 benchmark**다. 두 모델 모두 ANN 전용이므로 SNN 보정은 실행하지 않는다.
보조 실험까지 모두 실행하려면 `--families unet transformer`를 생략한다.
그 경우 **4개 전체 학습 + 2개 전체 보정 + 40개 평가 및 40개 benchmark**다.
시간·VRAM 측정 없이 완료 시간을 약속할 수 없다.
첫 단계는 환경/전체 데이터 검사와 전체 decode 검사다. 선택한 모든 family의 전체
이벤트/토폴로지 scan과 기존 batch forward/backward preflight가 통과해야 첫 학습을 시작한다.
반경 그래프는 C/D에만 생성하며 A/B/E에는 모델이 사용하지 않는 그래프를 만들지 않는다.
따라서 마지막 Transformer의 메모리 부족을 앞선 세 학습이 끝난 뒤 발견하지 않는다.
실패하면 결과를 축소하거나 GPU/CPU fallback하지 않고 해당 단계에서 멈춘다.
하나의 할당에 여러 대형 학습을 동시에 올리지 않는다. 각 학습 내부의 기존
physical batching과 평가의 batch/worker 후보 측정은 유지된다. 여러 장이 실제
할당되면 이미 할당이 분리된 작업별로 `--families`를 나눠 실행할 수 있다.
같은 family/output을 두 작업에서 동시에 실행하면 안 된다.

각 단계도 명시적으로 실행할 수 있다.

```bash
python -B scripts/run_ablations.py --stage profile --families transformer --execute
python -B scripts/run_ablations.py --stage train --families transformer --execute
python -B scripts/run_ablations.py --stage eval --families transformer --execute
```

Transformer 단독 E에는 `calibrate` 단계나 `best_snn.pt`가 없다. 보정은
`pointwise_unet`과 `graph_unet`에만 실행한다. A와 E만 비교할 때는
`--families unet transformer`를 사용한다.

중단 후에는 동일 코드·설정·데이터·할당 계약에서 다음 명령을 사용한다.

```bash
python -B scripts/run_ablations.py --stage all --families unet transformer --execute --resume
```

`--resume`은 통과한 profile을 다시 검증하고, 중단된 profile scan을 재개하며,
`last.pt`가 있으면 기존 exact-resume 검사를 거쳐 학습을 잇는다. 기존 보정은
sealed checkpoint와 원본 ANN 파일 hash를 검증해 보존한다. 완료된 평가와
benchmark는 기존 read-only resume inspector로 출처·runtime·설정·데이터를
검증한 뒤 건너뛴다. 기존 미완료 평가 폴더는 자동 이동·삭제·덮어쓰지 않는다.
그 경우 정확한 경로와 이유를 보고하고 중단한다. 기존 eval recovery 도구를
사용한 복구는 별도 명시적인 사용자 판단이 필요하다.

## 결과와 확인

```bash
python -B scripts/run_ablations.py --stage summary
```

기본 출력은 데이터셋별 **A(U-Net 단독)와 E(Transformer 단독)의 나란한 비교표**다.
PSNR/SSIM·파라미터·지연·FPS·VRAM의 원값, 차이 `E-A`, 비용 비율 `E/A`를 표시한다.
품질과 benchmark의 저장된 출처/정밀도/데이터 조건은 각각 따로 비교하며,
조건이 불일치하거나 비보고용 결과이면 해당 차이와 비율은 `N/A`로 둔다.
큰 품질 dataset의 hash는 저장된 주장끼리의 비교이지 원본 데이터 재검증이 아니다.
학습시간을 추론 지연으로 추정하지 않는다. 전체 학습시간은 이 표에서 측정하지 않는다.
보조 B/C/D, B의 ANN 대조군, 전체 T/동역학 및 출처 검증 열은 다음과 같이 확인한다.

```bash
python -B scripts/run_ablations.py --stage summary --details
```

A/E는 ANN 행만 있고 T/발화 동역학은 해당하지 않는다. 아직
학습하지 않은 결과는 `N/A`로 표시하며 0점이나 가짜 지표를 넣지 않는다.
PSNR/SSIM의 micro(프레임 평균), macro(그룹 평균), ms/FPS/VRAM과 원래 저장된
report eligibility를 구분한다. summary는 저장된 모델·모드·checkpoint 식별자를
비교하지만 원본 데이터 내용과 장비를 재측정하지 않는다. eligibility=true만으로
공정 비교나 원논문 재현을 주장하면 안 된다. `runs/fast`는 별도 과거 기준 결과다.
`quality_*` 일치 열은 FPS 비교 조건이 아니다. `benchmark_*` 열에서 속도 측정의
모델/장비/source/precision/선택 dataset 계약 hash를 검증하고 별도로 비교한다.
큰 전체 품질 평가 dataset의 hash는 여기서 원본 프레임 identity 배열을 재보관해
계산하지 않으며 `quality_dataset_hash_verified=false`로 명시한다. 반면 통상
100개 측정 프레임의 benchmark dataset 계약은 전체를 hash 검증한다. 소규모
보고서 메타데이터 메모리 guard를 초과하면 축소해 읽지 않고 전체 요약을 거부한다.

품질 평가에서 저장한 PNG를 실제 이벤트/그래프와 함께 확인하려면 각 family의
해당 config를 정확하게 지정한다. 예를 들어 graph_unet:

```bash
python -B scripts/generate_result_visualizations.py --eval-root runs/ablations/graph_unet/eval --aid-config configs/ablations/graph_unet-aid.json --hdr-config configs/ablations/graph_unet-hdr.json --cpu-threads 4 --memory-budget-mib 1024 --reserve-memory-mib 1024
```

그래프 없는 A/B/E도 같은 생성기에 해당 family config를 지정할 수 있다.
실제 이벤트와 복원 PNG를 표시하되 `no_graph`로 구분하고 가짜 반경 엣지를 만들지 않는다. 모든 family의 실제
GT/복원 PNG는 각 `eval/{hdr,aid}/{mode}/predictions`에서 확인할 수 있다.
E의 예측은 `runs/ablations/transformer/eval/{hdr,aid}/ann/predictions`에 저장된다.
이 변경의 로컬 단위 테스트는 합성 CPU/mock 명령 검사이며, 전체 학습·평가나
실제 서버 데이터 성능 검증이 아니다.
