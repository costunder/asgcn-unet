# 실행·평가의 규모 보존과 검증 범위

이 문서는 `AGENTS.md`를 대체하거나 예외를 허용하지 않는다. 실제 적용 설정은
실행 보고서와 checkpoint/protocol에서 확인한다.

## 보존하는 연구 계약

- 기존 model config, 6 graph layers, hidden 64, decoder 48, graph radius 0.08,
  기존 명시적 8,192-event 입력 계약, 원래 해상도와 전체 파일·프레임 범위를 유지한다.
- `fast`의 기존 physical batch 16, 40 epochs와 loss/optimizer 설정을 축소하지 않는다.
- ANN과 SNN 두 dynamics(`literal_eq15`, `standard_if`), T=4/8/16/32 비교를 유지한다.
  T=4만으로 최종 실험을 바꾸지 않는다.
- AID의 7,475,202-edge guard는 측정된 dense sample에 근거한 **실행 시 안전 한도 상향**이다.
  edge를 자르거나 graph radius를 바꾸지 않으며 checkpoint model hash에도 포함시키지 않는다.
  더 큰 입력에서 한도를 넘으면 명확하게 실패한다. 임의 cap 확대나 sample 건너뛰기는 하지 않는다.

## 실제 physical batching

학습·품질 평가에서는 동일한 shape의 **서로 다른 시퀀스**를 묶는다. 한 시퀀스의 연속
프레임은 시간 순서대로 처리하고, state는 배치 lane 번호가 아니라 sequence/source identity로
관리한다. 마지막 partial batch, 빈 event frame, shape 변경 및 시간 간격의 불연속도 보존한다.

CPU에서 event를 하나의 flat tensor로, target을 BCHW tensor로 묶어 pin/transfer한다.
각 sample의 시간 정규화와 sampling phase를 유지한 disjoint graph를 한 번에 만들고,
graph namespace가 다른 sample 사이의 연결을 차단한다. encoder, rasterization, decoder와
PSNR/SSIM/RMSE는 physical batch로 계산한다. 프레임별 반복은 metadata와 artifact 기록용이다.

보정에서는 사용하지 않는 target을 CPU에서 쌓거나 GPU로 전송하지 않는다. 이벤트,
metadata, 선택된 전체 프레임, 활성값 최대치와 보정 카운트는 그대로 보존한다.

명시적 B1 baseline의 수치 경로는 비교·회귀 검증을 위해 보존한다. 이를 새 production 경로의
최적 처리량 설정이라고 주장하지 않는다. 단일 프레임 compute benchmark도 같은 baseline
측정이며, full quality evaluation의 배치 처리량과 구별한다.

## 자동 batch/worker 측정

production inference의 `batch_size: "auto"`는 할당된 CUDA 장치에서 후보를 실제 측정한다.
설정 항목은 `batch_candidates`, `worker_candidates`, `profile_warmup`, `profile_steps`,
`profile_memory_fraction`, `batch_probe_indices`다. 후보는 모델·데이터 규모 제한이 아니다.
CPU에서 자동 선택을 시험하려면 **진단 전용** `profile_debug_cpu: true`가 명시되어야 한다.
CPU 측정값을 서버 GPU의 최적 batch라고 사용하지 않는다.

GPU 번호를 임의의 기본값으로 정하지 않고 기존 `CUDA_VISIBLE_DEVICES`를 보존한다.
할당 선택의 근거를 확인할 수 없으면 CUDA 사용 전에 중단한다. GPU가 하나 보인다는
사실만으로 사용자에게 할당됐다고 단정하지 않는다. 마스크 값 자체도 소유권 증명은 아니다.

각 시퀀스의 대표 위치와 명시된 dense probe를 포함하며, 후보마다 실제 batch occupancy,
I/O·전송·forward 시간, 처리량 및 CUDA 메모리를 기록한다. `profile_steps`는 최소 측정량이며
그 값 때문에 필수 대표 probe를 제외하지 않는다. OOM 후보만 실패로 기록하고 그 밖의 오류는
전파한다. 모든 후보가 실패하면 원래 모델을 줄이거나 CPU로 전환하지 않는다.

대표 표본은 전체 데이터의 최악 메모리 사용을 증명하지 않는다. 평가 중 실제 peak VRAM과
live sequence 수를 계속 기록하며, runtime OOM이 생기면 원인과 실행 보고서를 확인한다.
CUDA allocator memory는 GPU utilization이 아니고, process CPU 사용량은 worker CPU까지
측정한 숫자가 아니다. 미측정 항목은 명시적으로 구별한다.

품질 보고서의 latency는 **physical batch가 끝날 때까지 걸린 시간**이다. batch 크기로
나눈 값을 프레임 응답 지연으로 표시하지 않는다. model-only 처리량과 I/O·metric·저장까지
포함한 end-to-end 처리량은 별도 필드다.

## 기존 결과와 재실행

완료된 평가 결과와 기존 checkpoint를 삭제하거나 새 코드의 결과로 재표시하지 않는다.
동일 mode에 동시 writer가 들어오지 못하도록 sibling lock을 쓰며, 출처가 불명확한 lock은
자동 삭제하지 않는다. 복구 시에도 같은 lock 아래에서 검사와 보존을 수행한다.
직접 호출한 학습 run과 보정 출력에도 같은 소유권 확인 잠금을 적용한다. 비정상 종료로
남은 잠금은 자동 회수하지 않으며, 해당 작업의 상태를 확인한 뒤 처리해야 한다.

모델 학습 재개와 평가 복구는 다르다. 성공한 학습 batch의 checkpoint는 기존 exact-resume
계약을 지켜야 한다. 평가 중간의 recurrent state를 저장하지 않은 과거 실패 결과는 정확히
그 프레임부터 이어갈 수 있다고 주장하지 않는다. 이미 완료되고 protocol이 일치한 mode는
그대로 재사용하고, 불완전한 mode의 재평가는 원본을 보존한 뒤 실행한다.

완료 결과는 서버 터미널에서 한 번에 요약할 수 있다. 최신 스크립트가 서버에 반영된 뒤
저장소 루트에서 실행한다. 다운로드나 JSON 전체 붙여넣기는 필요 없다.

```bash
python scripts/summarize_eval.py runs/fast --format markdown
```

두 데이터셋 결과가 포함되는 공통 상위 경로를 지정한다. 입력 범위 밖의 EventHDR 결과는
자동으로 찾아내거나 추측하지 않는다. 결과를 다른 곳에 저장했다면 해당 경로도 인자로
추가한다. 서로 다른 성공 run은 원래 결과 경로로 구별하며 수치를 합치지 않는다.

## 검증과 보고

로컬 CPU unit/smoke test는 작은 **명시적 진단 fixture**를 사용한다. 이것은 실제 서버의
CUDA/Triton 성능 검증, 전체 학습, 전체 EventHDR/EventAid-R 평가 완료가 아니다.
소스 변경 후 신규 학습은 현재 source/config/data/runtime에 맞는 CUDA preflight가 필요하다.
기존 preflight를 현재 소스에서 검증한 것처럼 재사용하지 않는다.

실행 보고서에는 모델·trainable parameter 수, 실제 physical/effective batch,
전체/사용 데이터 수, loader 설정, 할당 CPU/RAM/CUDA 정보와 측정 범위를 남긴다.
모델 config를 바꾸지 않았더라도 새 실행 소스와 이전 결과의 provenance를 구분한다.
