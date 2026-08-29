# 실험 프로토콜

## 1. 고정할 연구 질문

1. EventHDR의 실제 이벤트에서 luminance frame을 얼마나 잘 복원하는가?
2. ASGCN-inspired graph message passing이 graph layer 없는 baseline보다 유효한가?
3. EventAid-R의 다른 장면·운동에서 품질이 얼마나 유지되는가?
4. ANN과 calibrated rate proxy의 품질·연산 지연 차이는 무엇인가?

현재 `snn` 모드는 실제 timestep IF/LIF가 아니므로 저전력·에너지 우위를 검증하는 질문에는 답할
수 없다. 해당 주장은 별도 spike simulation 또는 hardware 측정 후에만 추가한다.

## 2. 데이터 역할과 누수 방지

| 단계 | 데이터 | 허용되는 사용 |
|---|---|---|
| 학습 | EventHDR train | weight 최적화·crop |
| 검증 | EventHDR holdout | 물리 scene split 확정 후 macro SSIM checkpoint 선택 |
| 보정 | EventHDR train | BN folding·rate threshold |
| 내부 최종시험 | EventHDR 공식 eval | 학습 종료 후 1회 |
| 외부 최종시험 | EventAid-R | 학습·보정·threshold 선택 금지 |

`manifests/eventhdr_split.json`은 현재 물리 scene 대응표가 없는 `provisional` 상태다. 같은 장면의
파일이 train/validation에 동시에 들어가지 않도록 목록을 고친 뒤 `status: final`로 바꾸기 전에는
`configs/hdr_train.json`이 중단된다. `configs/hdr_smoke.json`만 비보고용으로 이 제한을 우회한다.

validation sample limit은 채점 frame 수에만 적용된다. `hdr_train`은 최대 500개, `hdr_smoke`는
최대 32개를 채점하며 아래 recurrent context frame은 이 수에 포함하지 않는다.

1. file/scene별 quota를 round-robin으로 배정한다.
2. 각 파일에서 deterministic contiguous window를 선택한다.
3. recurrent 모델은 window 앞의 같은 file/scene predecessor를 `validation_context_frames` 한도에서
   metric 없이 replay해 streaming ConvGRU state를 예열한다. 기본값은 본학습 64, smoke 8이며
   `null`이면 전체 prefix다. non-recurrent 모델은 context를 replay하지 않는다.
4. sample limit이 file/scene 수보다 작으면 일부 group을 버리지 않고 오류로 중단한다.
5. checkpoint 선택에는 scene별 SSIM 평균의 평균인 macro SSIM을 쓴다.

calibration은 recurrent state를 쓰지 않으므로 각 train file 전체 시간축을 `linspace`로 덮는다.
benchmark는 recurrent 모델이면 group별 연속 window와 최대 `eval.recurrent_context_frames`개의
unmeasured predecessor(현재 eval config 기본 32), 비순환 모델이면 time-spread sample을 사용한다.
`--warmup`은 recurrent context가 아니라 device/kernel warmup이다.
장면, sequence index, sensor shape가 끊기는 경계에서는 state를 초기화하며 benchmark 결과에 reset 수와
비율을 기록한다.

## 3. output domain

EventHDR와 EventAid-R 모두 다음 target 변환을 쓴다.

```text
integer image -> dtype range로 [0,1] 정규화
RGB이면 BT.709 luminance
y = log1p(5000*x) / log1p(5000)
```

이는 output 수치 domain만 통일한다. 센서 response와 exposure가 동일하다는 보장은 없으므로 두
dataset의 절대 PSNR/SSIM을 동일 분포처럼 해석하지 않는다.

## 4. 실행 순서

```bash
# 환경과 전체 파일 수
python scripts/check_env.py --require-cuda --require-full-data \
  --lock constraints/py312.txt

# 모든 event block 검증
asgcn-recon inspect --config configs/hdr_train.json --samples 2 --validate-all
asgcn-recon inspect --config configs/hdr_ann.json --samples 2 --validate-all
asgcn-recon inspect --config configs/aid_ann.json --samples 2 --validate-all

# 1 epoch real-data smoke
asgcn-recon train --config configs/hdr_smoke.json

# scene split final 이후 ANN 본학습
asgcn-recon train --config configs/hdr_train.json

# EventHDR train만으로 rate proxy 보정
asgcn-recon calibrate \
  --config configs/hdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500

# 고정 내부시험
asgcn-recon evaluate \
  --config configs/hdr_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/hdr_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16

# 고정 외부시험
asgcn-recon evaluate \
  --config configs/aid_ann.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/aid_snn.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16
```

SNN 명령은 ANN checkpoint, 0개 calibration, BN 미fold 상태, `simulation_steps < 1`을 거부한다.

## 5. 품질 지표

- PSNR: `[0,1]` data range
- SSIM: 11×11, σ=1.5 Gaussian valid window; 작은 영상은 fitting odd window
- RMSE
- `temporal_l1`: 같은 scene·sensor shape에서 sequence index가 정확히 1 증가하는 frame 사이에만
  `L1((pred_t-pred_t-1), (gt_t-gt_t-1))`
- LPIPS: `eval.lpips=true`일 때만 선택적으로 실행

결과는 micro, scene macro, per-scene으로 저장한다. 첫 frame과 장면·index·shape 불연속 뒤 첫
frame은 `temporal_l1` 집계에서 제외되고 CSV에는 null이 들어간다.

기존 논문과 SSIM을 비교할 때는 해당 논문의 구현, crop, border, color space, tone mapping까지
동일하게 맞춰 별도 검증한다. 현재 Gaussian 구현을 사용했다는 이유만으로 공식 수치와 완전히
동일하다고 가정하지 않는다.

## 6. 지연·메모리 지표

- evaluate: graph 생성과 model forward latency, 첫 frame cold start 포함
- benchmark: dataset read와 host-to-device 이동 제외, warmup 이후 CUDA Event 측정
- mean, p50, p90, p95, p99, max, FPS
- events/s, 평균 node/edge
- timestamp 기반 RTF, deadline miss ratio
- peak allocated/reserved GPU memory

`snn` rate conversion은 T번 graph propagation을 실행하지 않는다. 따라서 `T=4/8/16/32`는
activation 양자화 해상도 ablation이며, 실제 timestep latency ablation이 아니다.

## 7. 최소 비교표

1. `graph_layers=0` vs 6
2. ANN vs calibrated rate proxy `T=4/8/16/32`
3. `max_events=4096/8192/16384`
4. `causal_candidates=16/32/64`
5. ConvGRU on/off
6. EventHDR 내부 macro/per-scene vs EventAid-R 외부 macro/per-scene

모든 비교는 split, seed, tone mapping, crop, 해상도, checkpoint selection rule을 고정한다. 비교마다
config 원문, Git commit, `check_env.py` 출력, GPU 이름, CUDA/PyTorch, peak memory와 wall-clock을 함께
보존한다.

exact resume protocol은 선택 frame identity, group 길이, transform, manifest와 train/validation 원본
전체의 SHA-256을 저장한다. 절대경로와 mtime은 protocol에서 비교하지 않아 상대 파일 identity와
byte가 같은 복사본은 다른 mount에서도 재개할 수 있다. 최초 실행은 전체 원본을 읽고, 같은 경로의
resume은 run 폴더 sidecar에서 size/mtime/ctime이 모두 같은 파일의 기존 full hash를 재사용한다.
원본을 교체·복원했거나 전수 확인하려면 `rehash_data=true`로 cache를 무시한다.

## 8. 중단 조건

- manifest가 provisional이면 본학습 금지
- 전체 dataset validation 실패 시 해당 파일 제외가 아니라 원본 재다운로드/검증
- A100 10GB smoke OOM이면 full training 전에 기본 graph/model 설정을 재검토
- NaN loss/metric, 비단조 timestamp, 범위 밖 좌표 발생 시 결과 폐기
- EventAid-R 결과를 본 뒤 hyperparameter나 threshold를 바꾸면 기존 결과를 잠금시험으로 표기 금지
- A6000/A100 latency를 FPGA/ASIC latency 또는 에너지로 환산해 주장 금지
