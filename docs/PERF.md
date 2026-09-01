# Graph encoder 최적화 검증

앞부분은 GPU 병목 후속 수정과 동일 실데이터 비교 방법이고, 뒷부분은 이전 commit 대비
CPU 연산 단위 측정 기록이다. 독립 시퀀스 미니배치 학습 설계와 별도 실험 실행은
[TRAIN.md](TRAIN.md)를 따른다. CPU 수치를 GPU 가속 실측으로 해석하지 않는다.

## 동일 실데이터 GPU 비교

현재 병목 판단 자료는 사용자가 제공한 B4의 50-update CUDA stream 측정이다.
graph 22.16 ms, encoder 47.59 ms, decoder 6.14 ms, loss 1.82 ms, backward 50.19 ms였다.
이 값은 exclusive kernel time이나 GPU utilization이 아니며, 서로 중첩될 수 있다.
loss의 host 41.30 ms를 CPU loss 계산 41.30 ms로 해석하지 않는다. 앞서 제출된 GPU 작업을 기다린
시간이 포함될 수 있다. `gradient_check`는 update당 두 scope이므로 scope 평균과 step 평균도 다르다.
PyTorch의 [동기화 설명](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)을 참고한다.

후속 변경은 다음과 같다.

- 그래프: 여러 인접-cell 조회 청크의 allocation count를 한 번에 가져온다. candidate distance/edge는
  기존 청크 단위로 만들며 4-million candidate 한계와 max-edge guard, strict radius, edge 정렬을 유지한다.
  유효성 검사는 삭제하지 않고 결과 전송을 묶는다. 작은 hash 상수만 LRU 32개로 재사용하며 CUDA
  device와 stream별로 구분한다. 이벤트별 graph나 전체 dataset을 GPU에 cache하지 않는다.
- Temporal loss: 모든 lane의 이전 context가 있으면 전체 prediction/target tensor를 직접 사용한다.
  일부 context만 유효한 배치에는 기존 indexing과 유효 비율을 그대로 적용한다.
- `torch_fused`: 두 활성 spline basis 항을 묶어 청크당 gather/scatter dispatch 수를 줄인다.
  청크 working storage는 두 basis 항을 포함하므로 기준보다 클 수 있다. backward까지 모든 edge의
  channel message를 저장하지는 않는다.
- `triton`: gather·basis 곱·scatter를 하나의 GPU kernel로 실행하고 projected-feature gradient도
  융합한다. edge-channel message를 GPU 메모리에 중간 텐서로 저장하지 않는다. basis gradient와
  고차 미분 요청도 지원하되 `create_graph=True`의 backward는 명시적으로 미분 가능한 Torch 수식을 쓴다.

기준 `torch`는 유지하며 두 후보는 선택형이다. `triton`은 NVIDIA SM80 이상 CUDA와 Triton이 필요하고,
CPU/지원하지 않는 GPU/누락된 Triton/결정론 강제 모드에서는 오류를 낸다. 다른 backend로 조용히
바꾸지 않는다. 서버의 기존 `constraints/server.json`·`server.txt`에는 이미 Triton 3.7.1이 고정돼 있어
별도의 최신 버전 설치를 요구하지 않는다. atomic reduction의 누적 순서가 달라질 수 있으므로 두 후보의
bitwise 동일한 trajectory를 주장하지 않는다. 기존 AMP 재시도·유한 gradient 검사·clipping은 유지한다.

### 실행

**기존 학습과 같은 GPU/MIG에서 동시에 실행하지 않는다.** 실행 중인 checkout을 pull하거나 학습을
임의 종료하지 않는다. 다음 명령은 변경된 소스를 받은 checkout에서 GPU가 다른 학습에 사용되지
않을 때 실행한다. 완료된 기존 run/config/checkpoint는 변경하지 않는다.

```bash
conda activate asgcn
python -m pytest -q tests/test_spline_backend.py tests/test_graph_lookup.py &&
python scripts/bench.py --config configs/batch.json \
  --batches 4 8 16 --backends torch torch_fused triton --output runs/bench
```

같은 실제 EventHDR 16개 독립 시퀀스의 연속 구간을 사용한다. 시퀀스당 warmup 8프레임과 측정
32프레임으로, 각 조건은 **같은 warmup 128프레임 + 같은 측정 512프레임**을 처리한다.
`--batches 1 4 8 16`으로 B1도 비교할 수 있다. `--frames-per-stream 128`은 조건당 측정 프레임을
2,048개로 늘린다. batch마다 같은 step 수를 주어 서로 다른 frame 수를 비교하지 않는다.
frame_stride·max_events·해상도·모델 크기를 줄이지 않으며 production 데이터 대체 경로는 없다.

각 조건은 fresh subprocess에서 동일 seed의 모델·optimizer·scaler로 시작한다. 기본 반복 2회이고,
비교 순서는 seed로 섞는다. 동일 batch의 첫/최소·최대 raw-event 입력에서 기준 backend 대비
prediction·loss·gradient·BN 통계를 검사한 뒤 실제 `_training_step`과 시퀀스 context를 실행한다.
gradient는 clipping 전 norm과 clipping 후 tensor의 성분별 오차·벡터 상대 L2 오차를 함께 검사한다.
작아진 gradient에 큰 absolute tolerance만 적용해 방향 오류를 숨기지 않는다.
수치 검사 대상이 모두 edge 없는 그래프이면 spline 연산 자체를 검증하지 못하므로 통과시키지 않는다.
배치 크기가 다르면 BN pooling/update 횟수가 달라지므로 배치 크기 사이의 수치 동등성을 요구하지 않는다.

출력은 별도의 `runs/bench/report.json`과 조건별 JSON이다. 기존 출력이 있으면 덮어쓰지 않으므로
다음 측정은 `--output runs/bench2`처럼 새 경로를 지정한다. OOM·수치 불일치·backend 오류는 실패로
기록한다. 실패 조건을 더 작은 배치로 몰래 실행하거나 CPU로 바꿔 성공 처리하지 않는다.

### 서버에서 완료된 비교

사용자 제공 서버 출력에서는 CUDA 관련 pytest **123개가 24.88초에 통과**했고, 위 명령의
B4/B8/B16 × 3 backend × 2회, 총 18개 trial이 완료됐다. 조건별 측정 프레임은 512개다.

| batch | Torch | Torch-fused | Triton | Triton peak allocated/reserved |
|---:|---:|---:|---:|---:|
| 4 | 8.83 / 8.85 | 8.80 / 8.85 | 32.80 / 32.98 | 416.5 / 490 MiB |
| 8 | 8.99 / 9.03 | 8.95 / 8.95 | 34.96 / 35.45 | 743.9 / 822 MiB |
| 16 | 9.07 / 9.02 | 9.02 / 9.01 | **36.38 / 36.51** | 1244.0 / 1426 MiB |

B16의 두 반복 평균은 Triton 36.445 frame/s, Torch 9.045 frame/s로 약 **4.03배**다.
B16 Triton은 B8 Triton 평균보다 약 3.5% 빠르다. Torch-fused는 이 장비/입력에서 Torch보다 유의하게
빠르지 않았으므로 선택하지 않았다. 이 결과를 반영한 `configs/fast.json`은 B16+Triton을 사용하고
`runs/fast`에 별도 저장한다. 기존 B1/B4 config와 checkpoint는 변경하지 않는다.

### 해석과 후속 학습

측정 wall time은 해당 구간의 실제 HDF5 decode·전송·그래프·forward/loss/backward·검사·optimizer·
context 저장과 종료 CUDA 동기화를 포함한다. benchmark loader는 동기식이며 production의 4-worker
prefetch와 다르다. OS file cache를 지우지 않는다. 따라서 특정 window의 비교 결과이지 전체 epoch
처리량/수렴이나 모든 입력의 OOM 안전성을 입증하는 결과가 아니다. 준비·파일 hash·수치 비교와
warmup 시간은 throughput에서 제외한다. 이 단계에서 처음 사용하는 kernel을 준비하지만, 새로운
dtype/stride 등으로 측정 구간에서 추가 JIT가 발생한다면 그 비용은 포함된다. 이벤트 수 N은
runtime 인자로 전달해 frame마다 node 수가 바뀐다는 이유로 재컴파일하지 않도록 한다.
모든 비교 backend에 후속 graph 수정이 공통 적용되므로, 이 비교의 `torch`는 spline 기준 구현이지
이전 commit 전체를 실행한 결과가 아니다. 이전 학습 화면과 비교한 값을 통제된 가속률로 보고하지 않는다.

`--trace-steps 8`을 추가하면 throughput과 분리된 추가 pass에서 bounded PyTorch operator trace를
저장한다. 이 trace용 실행의 시간은 가속률 계산에 사용하지 않는다. trace·보고서는 사용자별 경로와
hostname을 가리며, source/설정/선택 프레임 identity는 hash로 기록한다.
측정 API의 범위는 [PyTorch Profiler](https://docs.pytorch.org/docs/2.13/profiler.html)를 참고한다.

메모리를 많이 쓴 조건이 아니라 **수치 검사를 통과하고 반복 측정 처리량이 좋아진 조건**을 후보로
선택한다. benchmark는 config를 자동 변경하거나 학습을 시작하지 않는다. 선택한 batch/backend는
새 run/config의 실제 dense/full-batch CUDA preflight를 거쳐야 한다. 새 소스에 이전 checkpoint를
exact-resume하도록 source 검사를 우회하지 않는다. 검증되는 이전 topology 통계만 재사용할 수 있으며,
이전 GPU probe 통과는 새 backend/B의 통과 근거가 아니다.

서버의 123개 CUDA 검사와 위 18개 trial은 해당 환경에서 실제 실행됐다. 다만 전체 EventHDR epoch,
40-epoch convergence와 전체 평가 행렬은 아직 완료된 결과가 아니다. 로컬 Windows CPU 환경에서는
CUDA 컴파일·실행을 재검증할 수 없었으며, 아래 과거 CPU 수치로 서버 결과의 범위를 확대하지 않는다.

### 이번 변경의 로컬 검증

전체 Windows CPU suite는 **1,286 passed, 82 skipped, 1 warning (73.36초), exit 0**이다.
skip은 CUDA 장비·Linux shell·Windows symlink 제약을 포함한다. warning은 기존 test-only quantized
buffer의 deprecated API 경고이며 이 실행에 native access-violation 출력은 없었다.
Ruff와 diff whitespace 검사를 통과했다. CUDA 커널 테스트는 작성됐지만 실행되지 않은 항목을
통과로 세지 않는다. CPU `torch_fused` 비교에서는 일부 조건이 기준보다 느렸으며, dispatch 감소를
실제 속도 향상으로 환산하지 않고 선택형 후보로 유지한다.

## 이전 연산 최적화 기록

2026-08-31. 비교 기준은 변경 전 commit `0eae40f`의 graph 구현이다. 모델 구조, parameter/state-dict
key, 두 SNN dynamics, 전체 데이터 범위, 8,192-event 제한과 40 epoch는 변경하지 않았다.
이 문서의 CPU 연산 측정은 학습된 모델의 품질 평가, 전체 pipeline 처리량 또는 GPU 실측이 아니다.

## 변경한 연산

1. **Radius graph:** valid/nonempty cell의 반복 boolean compaction을 zero-count masking으로 대체하고,
   candidate group을 한 번 확장해 source/destination을 구한다. retained edge index를 한 번만 만들어
   source, destination, distance에 재사용한다. topology scan은 고정 크기 integer scatter로 degree를
   누적하고 마지막에 통계를 한 번 전송한다. strict `< radius`, directed edge, 출력 정렬과 edge guard는
   유지한다. 입력 검증과 candidate 수 확인, variable-size edge compaction의 동기화가 모두 없어진 것은 아니다.
2. **Spline:** 기존 eager autograd는 chunk를 순회하더라도 두 basis의 `[E,Cout]` message를 역전파까지
   저장했다. `ops.py`는 topology·scalar basis를 저장하고 backward에서 chunk별 destination derivative를
   한 번 gather해 두 항에 재사용한다. node projection gradient는 직접 누적하고 basis derivative에
   필요한 경우만 node projection을 추가 저장한다. 학습 gradient를 detach하거나 생략하지 않는다.
3. **SNN:** 고정 analog input에 대한 첫 layer affine 호출을 `T`회에서 1회로 줄였다. threshold
   conversion을 재사용하고 spike의 zero branch에서 `[N,C]` zero tensor를 매번 할당하지 않는다.
   membrane·spike recurrence와 soft reset은 유지한다.

별도 C++/CUDA extension이나 compiler 의존성은 추가하지 않았다. 기존 PyTorch native tensor 연산의
구성과 autograd 저장 방식을 변경한 것이다. 새 CUDA kernel을 구현·검증했다고 주장하지 않는다.

## 검증 조건과 경계

- Windows CPU, PyTorch `2.13.0+cpu`; 연산 시간 비교는 `torch.set_num_threads(1)`.
- old/new는 같은 입력과 parameter state를 사용한다. 시간은 warmup 후 순서를 교차한 반복 측정의 중앙값.
- GPU 장비가 없어 CUDA FP32/FP16/BF16 동등성·지연시간·peak VRAM은 여기서 확인하지 못했다.
- floating-point reduction 순서는 backward에서 달라질 수 있다. FP32/FP64 gradient는 수치 허용 오차로,
  CPU bfloat16은 dtype의 반올림 오차를 허용해 비교했다. CPU bfloat16 테스트에서 최대 absolute gradient
  차이 `0.0078125`를 관찰했다. bitwise 동일한 학습 trajectory를 주장하지 않는다.
- forward의 basis-major 누적 순서는 유지했고 CPU 비교에서 forward가 일치했다. SNN은 timestep별 spike,
  firing rate, 임계점 및 바로 인접한 float 경계까지 기존 loop와 비교했다.
- 첫·두 번째 미분을 `gradcheck`/`gradgradcheck`로 검사했다. 고차 미분 graph의 메모리까지 ordinary
  backward와 동일한 상한으로 제한된다는 주장은 하지 않는다.

## 연산 단위 측정

아래 두 시간 측정은 고정 seed로 생성한 연산 입력이다. production dataset이나 학습 sample을
대체하지 않으며, 실제 데이터 성능으로 해석하지 않는다.

| 대상 | 변경 전 | 변경 후 | 조건 |
|---|---:|---:|---|
| sparse radius graph 생성 | 58.208 ms | 51.270 ms | N=8,192, 3D, radius=0.08, chunk=512 |
| 동일 graph topology scan | 47.338 ms | 41.600 ms | 동일 입력 |
| SNN standard IF, T=32 | 212.333 ms | 181.860 ms | N=512, E=8,192, C=32, 6 layers; warmup 후 7회 |

Spline의 **역전파용 고유 저장 tensor storage**는 N=1,024 / E=65,536 / K=5 / Cin=Cout=64 /
chunk=8,192 / FP32 / basis gradient off 조건에서 **36,524,032 → 2,969,600 bytes**로 약 91.9% 감소했다.
같은 underlying storage를 공유하는 view는 한 번만 센 값이며 process RAM이나 peak VRAM이 아니다.
node projection, graph와 다른 layer의 allocation이 있으므로 전체 모델 메모리가 같은 비율로 줄지는 않는다.

## 실제 EventHDR 입력 측정

공식 `26.h5`와 `38.h5`의 dataset index 100 (`image000000100`)을 read-only로 읽었다. 입력은 full-frame
240×320이며 기본 event 제한 8,192를 유지했다. 각각 N=2,109/E=58,834와 N=8,192/E=210,200이다.
38.h5의 원래 interval event 9,266개에서 8,192개를 선택하는 기존 sampling도 변경하지 않았다.

seed=2026, hidden=64, graph layers=6, spline K=5/chunk=65,536, graph radius=0.08/chunk=512. 같은 초기
parameter state와 입력을 사용했다. ANN은 train mode에서 `output.square().mean().backward()`를 실행하고
매번 parameter gradient를 초기화했다. input feature/position에는 gradient를 요청하지 않았다.
SNN은 별도로 초기화한 encoder의 선택된 실제 frame activation으로 normalization한 동일 state를
`no_grad`로 실행했다. 이는 timing용 state이며 전체 calibration을 완료한 checkpoint가 아니다.
각 구현 1회 warmup 후 old/new 순서를 교차한 각 5회 측정의 중앙값이다. 데이터 로딩과 U-Net, optimizer step,
metric·전송 시간은 포함하지 않는다. **학습된 복원 모델이나 전체 실험 성능이 아닌 graph encoder 측정**이다.

| 연산 / 실제 입력 | 변경 전 | 변경 후 |
|---|---:|---:|
| radius graph / 26.h5 | 17.280 ms | 14.992 ms |
| radius graph / 38.h5 | 68.817 ms | 61.029 ms |
| ANN encoder forward+backward / 26.h5 | 168.506 ms | 173.512 ms |
| ANN encoder forward+backward / 38.h5 | 675.631 ms | 571.890 ms |
| SNN literal Eq15 T=16 / 26.h5 | 1,723.525 ms | 1,283.198 ms |
| SNN literal Eq15 T=16 / 38.h5 | 5,400.837 ms | 4,686.728 ms |
| SNN standard IF T=16 / 26.h5 | 1,500.482 ms | 1,284.872 ms |
| SNN standard IF T=16 / 38.h5 | 5,381.739 ms | 4,679.053 ms |

26.h5의 ANN 중앙값은 약 3.0% 느려졌다. 반복 범위도 old 160.9–174.6 ms/new 157.1–178.8 ms로 겹쳤다.
모든 입력에서 latency가 개선됐다고 주장하지 않는다. 해당 sample에서도 저장 메모리는 크게 줄었고,
더 큰 38.h5 graph의 ANN 중앙값은 약 15.4% 짧아졌다. 전체 데이터 분포/GPU에서의 trade-off는 별도 측정 대상이다.

| ANN encoder의 역전파용 고유 저장 tensor storage | 변경 전 | 변경 후 |
|---|---:|---:|
| 26.h5 입력 | 190,153,816 bytes | 9,415,768 bytes |
| 38.h5 입력 | 680,135,104 bytes | 34,400,704 bytes |

shared view 중복을 제외하되 saved parameter/graph storage를 포함한 값이다. process peak RAM/VRAM은
아니며 backward working buffer는 별도다. 두 입력 모두 ANN forward와 graph가 일치했고, parameter
gradient의 최대 absolute 차이는 각각 `1.49e-8`, `1.86e-8`이었다. 두 SNN dynamics의 출력과 발화율도
기존 loop와 일치했다. 작은 표본의 수치·시간 비교가 전체 학습 trajectory 동등성을 입증하지는 않는다.

## 회귀검사

```bash
python -m pytest -q tests/test_graph_opt.py tests/test_spline_opt.py tests/test_snn_opt.py
```

각 test의 이전 연산은 독립적인 동등성 기준이며 production backend나 대체 dataset으로 사용되지 않는다.
전체 suite는 **860 passed, 35 skipped**였다. skip은 Linux 전용 shell test 22개, Windows symlink 권한
관련 5개와 CUDA 의존 검사 8개다. Ruff와 `git diff --check`도 통과했다.

## 기존 실험과의 관계

기본 실행 명령과 config는 그대로다. 기존 profile/checkpoint의 source contract는 이전 소스를 기록하므로
새 소스에서 exact-resume를 강제하거나 검사 값을 수동 변경하지 않는다. 기존 결과는 보존하고 새
preflight 및 실험 lineage로 측정한다. 이 최적화 검증은 전체 학습·전체 평가를 대신하지 않는다.

후속 AMP/사전검사 수정은 위 CPU operator benchmark와 별도다. 새 전수검사는 CUDA topology,
event-only HDF5 읽기, CPU helper thread 제한, 구간별 저장·재개를 사용한다. 출처·데이터·설정이
검증되는 이전 전수 통계는 명시적으로 이관할 수 있으나 GPU probe는 새로 실행한다. 이 변경의
서버 CUDA 처리량·전체 학습 속도는 아직 측정하지 않았으며 위 표의 수치로 대신하지 않는다.
