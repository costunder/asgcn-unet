# Graph encoder 최적화 검증

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
