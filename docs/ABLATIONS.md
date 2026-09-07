# 구조 비교 실험

이 코드는 기존 `runs/fast` 실험과 분리된 신규 학습·보정·평가를 실행한다.
기존 ANN/SNN 결과에 이름만 바꿔 새 비교군으로 사용하지 않는다.

| 주 비교군 | 학습 family | 인코더 → 복원기 | 평가 |
| --- | --- | --- | --- |
| A | `unet` | 정규화된 원본 이벤트 특징 → 평균 raster → 기존 recurrent U-Net | ANN |
| B | `pointwise_unet` | 그래프 이웃 전달 없는 6층 pointwise 발화 인코더 → 기존 recurrent U-Net | SNN + ANN 대조군 |
| C | `graph_unet` | 기존 6층 일반 GNN → 기존 recurrent U-Net | ANN |
| D | `graph_unet` | 동일 가중치의 보정된 6층 Spiking GNN → 기존 recurrent U-Net | SNN |
| E | `graph_transformer` | 기존 6층 Spiking GNN → recurrent Transformer 복원기 | SNN + ANN 대조군 |

4개 ANN 학습이 A–E를 구성한다. B, D, E의 SNN은 각각 자기 ANN을 전체
EventHDR 학습 프레임으로 보정한다. 그래프 없는 SNN도 학습된 pointwise
인코더의 발화 변환이며, 빈 그래프를 넣고 기존 GNN이라고 부르는 구현이 아니다.
두 데이터셋 모두 ANN과 해당 family의 `literal_eq15`, `standard_if` ×
`T=4,8,16,32`를 평가하고 각각 compute-only benchmark를 실행한다.
A에는 SNN 보정이나 의미 없는 T 실험을 추가하지 않는다. 총 56개 품질 평가와
56개 benchmark(두 데이터셋 합계)이다.

## 유지하는 계약과 해석상의 한계

`configs/ablations/{family}-{train,hdr,aid}.json`은 기존 `fast.json`,
`hdr-fast.json`, `aid-fast.json`의 전체 규모 설정을 보존한다. seed=2026,
40 epochs, physical batch=16, 6 encoder layers/hidden=64(존재하는 인코더),
decoder base=48, 기존 이벤트 8192개 제한과 sampling factor=1, 입력 전체 해상도,
학습 51 H5/평가 19 H5/EventAid-R 14 ZIP, loss, optimizer, learning-rate schedule,
validation/calibration/eval의 batch/worker 후보가 그대로다. 이 값들은 기존
실험 계약이며 새로 추가한 축소가 아니다. 출력만 `runs/ablations/{family}`에 둔다.
인코더가 없는 A에서 존재하지 않는 6층/hidden=64를 사용했다고 보고하지 않는다.
실제 모델의 parameter 수와 활성 모듈은 실행 전 구조 로그를 따른다.

Transformer는 U-Net과 같은 기본 channel 수와 recurrent 조건을 유지하고,
전체 공간을 window attention으로 처리한다. 설정은 depths `[1,1,2,1,1]`,
heads `[3,6,12,6,3]`, window=8, MLP ratio=4.0이다. 48/96/192 channel 폭에
각 head dimension=16을 배정하고, bottleneck에 두 attention block을 둔다.
이는 명시적인 신규 비교 설계이며 논문의 Transformer 재현이나 parameter-matched
모델이라고 주장하지 않는다. window 경계와 padding은 모델 코드의 mask를 따른다.
CPU에서 구성한 decoder 자체의 parameter 수는 Transformer 3,383,323개,
U-Net 4,284,049개였다(input=64/base=48/output=1/recurrent 조건).
이는 decoder parameter 측정이며 GPU 처리량·VRAM 실측이나 전체 모델 크기가 아니다.

체크인된 실제 4개 train config를 CPU에서 구성했을 때 전체 모델 parameter 수는
다음과 같다. 학습이나 GPU 실행 없이 구조만 구성해 센 값이며, 결과 요약은
이 표를 하드코딩하지 않고 실제 평가 보고서의 parameter 수를 읽는다.

| Family | 전체 parameters |
| --- | ---: |
| `unet` | 4,258,129 |
| `pointwise_unet` | 4,305,937 |
| `graph_unet` | 4,409,617 |
| `graph_transformer` | 3,508,891 |

- A↔D는 입력의 원본 특징/학습 특징 차이와 인코더 용량 차이를 함께 포함한다.
  A는 E2VID 또는 특정 voxel-grid baseline의 재현이 아니다.
- B↔D는 동일 이벤트/node 전처리에서 이웃 message passing을 제거하는 비교다.
  parameter 수는 같지 않으므로 이를 반드시 함께 보고한다.
- C↔D는 자기 학습 ANN에서 발화 변환·T·동역학을 바꾸는 paired 비교다.
- D↔E는 인코더 설계·입력·T·동역학·시간 기억을 맞춘 복원기 구조 비교다.
  별도 end-to-end 학습이므로 인코더 가중치까지 고정한 비교는 아니다.
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
이 스크립트는 GPU 0 또는 4를 선택하거나 `CUDA_VISIBLE_DEVICES`를 변경하지 않는다.
할당을 검증할 수 없으면 기존 CUDA 안전 검사에서 실행이 거부된다.
사용자가 현재 할당을 확인한 동일 Conda 환경에서 명시적으로 실행한다.

```bash
python -B scripts/run_ablations.py --stage all --execute
```

이는 짧은 검사 명령이 아니라 **4개 전체 학습 + 3개 전체 보정 + 56개 평가 및
56개 benchmark**다. 시간·VRAM 측정 없이 완료 시간을 약속할 수 없다.
첫 단계는 환경/전체 데이터 검사와 전체 decode 검사다. 선택한 모든 family의 전체
graph scan과 기존 batch forward/backward preflight가 통과해야 첫 학습을 시작한다.
따라서 마지막 Transformer의 메모리 부족을 앞선 세 학습이 끝난 뒤 발견하지 않는다.
실패하면 결과를 축소하거나 GPU/CPU fallback하지 않고 해당 단계에서 멈춘다.
하나의 할당에 여러 대형 학습을 동시에 올리지 않는다. 각 학습 내부의 기존
physical batching과 평가의 batch/worker 후보 측정은 유지된다. 여러 장이 실제
할당되면 이미 할당이 분리된 작업별로 `--families`를 나눠 실행할 수 있다.
같은 family/output을 두 작업에서 동시에 실행하면 안 된다.

각 단계도 명시적으로 실행할 수 있다.

```bash
python -B scripts/run_ablations.py --stage profile --families graph_transformer --execute
python -B scripts/run_ablations.py --stage train --families graph_transformer --execute
python -B scripts/run_ablations.py --stage calibrate --families graph_transformer --execute
python -B scripts/run_ablations.py --stage eval --families graph_transformer --execute
```

중단 후에는 동일 코드·설정·데이터·할당 계약에서 다음 명령을 사용한다.

```bash
python -B scripts/run_ablations.py --stage all --execute --resume
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

A/B/C/D/E와 두 ANN 대조군을 dataset/T/동역학별로 한 표에 출력한다. 아직
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

그래프 없는 A/B도 같은 생성기에 해당 family config를 지정할 수 있다.
실제 이벤트와 복원 PNG를 표시하되 `no_graph`로 구분하고 가짜 반경 엣지를 만들지 않는다. 모든 family의 실제
GT/복원 PNG는 각 `eval/{hdr,aid}/{mode}/predictions`에서 확인할 수 있다.
이 변경의 로컬 단위 테스트는 합성 CPU/mock 명령 검사이며, 전체 학습·평가나
실제 서버 데이터 성능 검증이 아니다.
