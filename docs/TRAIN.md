# 단일 프레임 기준선과 독립 시퀀스 배치 학습

`configs/train.json`은 기존 batch 1 기준선(training protocol v5)이다.
`configs/batch.json`은 batch 상한 4의 별도 실험(training protocol v6)이며,
ASGCN 모델의 `architecture_version=2`를 바꾸는 설정은 아니다.
새 배치 경로의 실제 서버 GPU 속도·최대 메모리·40-epoch 수렴은 아직 검증하지 않았다.

## 계산과 순서

각 프레임의 이벤트로 기존과 같은 그래프를 만든 뒤, 노드 인덱스만 이동해 분리된 그래프들을 합친다.
프레임 사이의 edge는 만들지 않는다. 합친 그래프에 encoder ANN을 **한 번**, 각 프레임의 raster를
쌓은 텐서에 recurrent U-Net decoder를 **한 번** 호출한다. `forward_sample`을 B번 부르는
방식이나 gradient accumulation은 아니다. 기존 batch 1 경로는 유지한다.

시퀀스 키는 `(sequence_id 또는 scene, source_file)`이다. 같은 scene에 속해도 다른 원본 파일은
독립된 recurrent stream으로 취급한다. 샘플러는 다음 조건을 지킨다.

- 전체 선택 프레임을 epoch마다 한 번씩 포함하며, 각 시퀀스 내부 시간 순서를 보존한다.
- 동시에 최대 B개 시퀀스만 활성화한다. 한 배치에 같은 시퀀스의 두 프레임을 넣지 않는다.
- crop 이후 해상도가 같은 활성 시퀀스끼리 묶는다. 다른 해상도는 순환 처리하고,
  시퀀스가 끝나면 빈 자리에 다음 시퀀스를 넣는다.
- 마지막 배치·해상도 불일치 배치는 작아질 수 있다. padding이나 `drop_last`로 프레임을 버리지 않는다.
- 이벤트가 0개인 프레임도 포함한다. 샘플러는 HDF5 image의 크기 메타데이터만 읽고 GT 픽셀은 읽지 않는다.

Recurrent state와 temporal loss의 이전 prediction/GT는 배치 슬롯 번호가 아니라 시퀀스 키로 연결한다.
처음 보는 시퀀스, frame index의 불연속, 해상도 변경, 새 epoch에서는 context를 초기화한다.
유효 state가 없는 슬롯은 0으로 초기화하며 다른 슬롯의 state를 가져오지 않는다.
성공한 optimizer update 뒤에만 context를 갱신하고 detach한다. 선택된 마지막 프레임을 처리한
시퀀스의 context는 제거해 최대 B개 활성 시퀀스만 유지한다. 프레임 간 장기 BPTT는 하지 않는다.
AMP overflow 재시도는 같은 배치와 같은 입력 context를 다시 사용하며, 성공 전에는 순서를 진행하지 않는다.

## 별도 실험이어야 하는 이유

Encoder의 학습 BatchNorm은 합친 그래프의 **전체 노드**를 기준으로 통계를 계산한다.
따라서 노드 수가 많은 프레임이 BN 통계에 더 많이 기여한다. 배치마다 optimizer update도 한 번만 하므로,
기존의 프레임별 BN·프레임별 parameter update와 학습 궤적이 다르다.

Reconstruction loss는 배치에 실제 포함된 프레임들의 평균이다. 같은 해상도끼리 묶으므로 픽셀 평균은
프레임별 평균과 같다. Charbonnier/SSIM/gradient/temporal 가중치는 각각 `1.0/0.2/0.1/0.2`다.
Temporal 항은 연속 context가 있는 프레임만 계산하되 `유효 context 수 / 실제 배치 크기`를 곱한다.
Context가 없는 프레임의 temporal 항을 0으로 놓은 전체 프레임 평균에 해당한다.
부분 배치도 실제 크기로 평균내며, epoch loss는 처리한 프레임 수로 가중 평균한다.

기본 40 epochs와 학습률 `0.001`은 그대로지만, epoch당 update 횟수는 배치 구성에 따라 줄어든다.
동일한 프레임 노출 횟수가 동일한 optimizer update 횟수나 동일한 수렴 결과를 뜻하지 않는다.
기준선 checkpoint를 batch 실험에 `--resume`으로 연결하지 않는다. Batch 실험은 새로 학습하며,
그 이후의 재개만 일치하는 batch protocol/checkpoint 계약을 따른다.
검증·calibration·평가는 기존의 시간 순서 batch 1 경로를 사용한다.

## 실제 학습 시간 기록

Batch 설정은 실제 학습의 첫 10개 성공 step을 warmup으로 두고, 다음 50개 step에 `StageTimer`를 켠다.
여기서 step은 배치 하나에 대한 성공한 optimizer update이며 frame 수와 구분한다.
별도 모델을 돌리거나 측정을 위해 추가 optimizer update를 실행하지 않는다. 수집 창이 끝나면
`runs/batch/timing.json`을 저장한다. 기록 대상은 data loading, transfer, graph, encoder, decoder,
loss, backward, gradient 검사, optimizer 등의 단계다. AMP 재시도가 있으면 해당 step의 재시도 작업도 포함한다.

`host_wall`은 Python 실행·연산 제출·대기 시간을 포함한 host 경과시간이다. `cuda_elapsed`는
선택한 CUDA stream의 event 사이 경과시간이며, GPU utilization이나 다른 작업을 제외한 kernel 순수 실행시간이
아니다. 중첩된 단계는 시간이 겹칠 수 있으므로 합산해서 전체 step 시간으로 해석하지 않는다.
측정 창을 회수할 때 CUDA를 한 번 동기화한다. 각 단계 경계마다 동기화하지는 않는다.
측정 자체의 overhead와 첫 50개 step의 대표성 한계도 고려해야 한다.
`window_complete`, `measured_steps`, `dropped_scopes`를 확인해 미완료·기록 누락을 구분한다.

`runs/batch/history.json`에는 epoch의 `training_seconds`, `frames`, `optimizer_steps`,
`frames_per_second`, `batch_size_limit`, AMP 재시도와 GPU peak allocated/reserved 메모리가 기록된다.
`performance` 시간은 epoch 종료 CUDA 동기화까지 포함하고 validation 시간은 제외한다.
반면 현재 `gpu_memory` peak는 epoch 학습 시작부터 해당 epoch의 선택적 validation까지 포함한다.
따라서 validation을 실행한 epoch의 peak를 순수 학습 VRAM으로 해석하지 않는다.
진행률의 단위도 optimizer step이 아니라 **frame**이다. CUDA timing이 없으면 GPU 측정으로 보고하지 않는다.

## 서버 실행과 결과 분리

현재 기준선 학습이 실행 중이면 그 checkout에서 `git pull`하거나 임의로 작업을 종료하지 않는다.
기존 작업이 끝난 뒤, 또는 사용자가 보존할 checkpoint와 종료 시점을 정한 뒤 소스를 갱신한다.
기존 `runs/train`과 checkpoint는 그대로 보존한다. 아래 명령은 갱신된 저장소 root에서,
Conda `asgcn`과 할당된 GPU를 사용하며 기존 학습이 실행 중이지 않을 때의 새 실험 명령이다.
기존 run을 가리키는 `TRAIN_CONFIG`, `ANN_CHECKPOINT`, `SNN_CHECKPOINT`, `RESUME_CHECKPOINT` 등의
수동 override가 남아 있지 않은 terminal을 사용한다.

완료된 호환 topology 보고서 `runs/profile2.json`이 있는 경우:

```bash
export EXPERIMENT=batch PROFILE_OUTPUT=runs/batch-profile.json &&
PROFILE_REUSE_REPORT=runs/profile2.json bash scripts/run.sh profile &&
bash scripts/run.sh train &&
bash scripts/run.sh calibrate &&
bash scripts/run.sh eval
```

이전 보고서는 topology 기록만 검증 후 재사용한다. 예전 GPU 통과 결과는 재사용하지 않고,
현재 설정의 실제 batch schedule과 밀집/첫/빈/희소 입력 배치에 대해 CUDA 학습 경로를 새로 검사한다.
Batch 전체 CUDA 검사를 통과한 현재 보고서가 있어야 학습을 시작한다. 이 검사는 40 epochs의 성공이나
모든 입력에서 OOM이 없다는 보증이 아니다. 구현된 gate와 실제 서버 통과 결과를 구분한다.

호환 보고서가 없으면 위 `profile` 명령에서 `PROFILE_REUSE_REPORT=...`만 빼고 전수 scan부터 수행한다.
재사용이 거부되면 검증을 우회하지 않는다. 이미 만들어진 출력/scan을 덮어쓰지 말고 실패 원인을 확인하거나
`runs/batch-profile2.json`처럼 새 `PROFILE_OUTPUT` 경로를 선택한다.
새 terminal에서도 `EXPERIMENT`와 `PROFILE_OUTPUT`을 다시 설정한다.

출력은 `runs/batch`(학습·calibration), `runs/batch-profile.json`(사전검사),
`runs/batch-status`(stage 상태), `runs/batch/eval/hdr`와 `runs/batch/eval/aid`(평가)로 분리한다.
기존 EventHDR/EventAid-R 데이터는 다시 다운로드하거나 복제하지 않는다.

## 아직 해결하지 않은 병목

그래프는 **프레임마다, epoch마다 다시 생성**한다. 전체 데이터의 graph cache나 디스크 graph cache는
구현하지 않았다. 작은 LRU cache는 시간순으로 한 번씩 읽는 학습에서 적중률이 낮을 수 있고,
전체 edge/index/attribute cache는 설정과 이벤트 밀도에 따라 100 GB를 넘을 수 있다.
실제 topology 크기와 I/O 시간을 측정한 뒤 명시적인 저장 공간 예산으로 판단할 다음 작업이다.

기존 spline 최적화의 중간 텐서 보관 감소와 이번 호출 배치화를 실제 GPU 속도 향상으로 환산하지 않는다.
Batch는 동시에 처리하는 노드·edge·decoder activation이 늘어 peak VRAM이 더 커질 수 있다.
A100 전체 GPU와 MIG 1g.10gb는 같은 실행 자원이 아니며, B=4가 해당 MIG에서 충분한지도 실제 gate로 확인해야 한다.
4배 가속을 약속하지 않는다. 프로젝트 전용 C++/CUDA 확장을 새로 만든 것도 아니다.
