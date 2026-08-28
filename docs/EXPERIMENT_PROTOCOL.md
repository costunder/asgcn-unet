# 재현 가능한 실험 프로토콜

## 데이터 역할

| 단계 | 데이터 | 허용되는 사용 |
|---|---|---|
| 학습 | EventHDR train | weight 최적화와 augmentation |
| 검증 | EventHDR train holdout | checkpoint 선택과 hyperparameter 결정 |
| 내부 최종시험 | EventHDR 공식 eval | 학습 완료 후 품질·지연 평가 |
| 외부 일반화 | EventAid-R | 학습·BN·threshold 보정 없이 최종 평가만 수행 |

EventHDR 공개 train H5와 논문의 물리 장면 사이 공식 대응표가 없으므로 현재 manifest는
파일 단위 임시 분할이다. 대응표를 확보하면 동일 장면이 양쪽에 걸리지 않도록 group split으로
교체해야 한다.

## 기준 실행 순서

```bash
# 1. 데이터 구조
asgcn-recon inspect --config configs/eventhdr_train.json

# 2. ANN 학습
asgcn-recon train --config configs/eventhdr_train.json

# 3. EventHDR train으로만 BN folding 및 SNN threshold calibration
asgcn-recon calibrate \
  --config configs/eventhdr_train.json \
  --checkpoint runs/eventhdr_asgcn/best.pt \
  --output runs/eventhdr_asgcn/best_snn.pt \
  --samples 500

# 4. ANN/SNN 내부시험
asgcn-recon evaluate \
  --config configs/eventhdr_eval.json \
  --checkpoint runs/eventhdr_asgcn/best.pt

asgcn-recon evaluate \
  --config configs/eventhdr_eval.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16

# 5. 잠근 외부시험
asgcn-recon evaluate \
  --config configs/eventaid_r_eval.json \
  --checkpoint runs/eventhdr_asgcn/best_snn.pt \
  --inference-mode snn --simulation-steps 16
```

## 기록 항목

- 품질: PSNR, SSIM, RMSE, 선택적으로 LPIPS
- 지연: mean, p50, p90, p95, p99, max, FPS
- 스트리밍: 실제 timestamp 기반 RTF와 deadline miss ratio
- 그래프: 평균 event/node/edge 수
- SNN: simulation step과 layer 평균 firing rate
- 시스템: GPU 모델, PyTorch/CUDA/cuDNN, peak allocated GPU memory

`evaluate`는 품질과 frame별 CSV를 저장한다. `benchmark`는 파일 I/O를 timer 밖에 두어
연산 latency를 측정한다. 처음 한 번의 임의 가중치 forward 시간은 성능 결과로 사용하지 않는다.

## 필수 비교 실험

1. `graph_layers=0`과 ASGCN graph encoder
2. ANN과 SNN `T=4/8/16/32`
3. `max_events=4096/8192/16384`
4. `causal_candidates=16/32/64`
5. ConvGRU 사용/미사용

모든 비교는 같은 split, seed, tone mapping, crop과 평가 해상도를 사용한다. EventAid-R 결과로
설정이나 threshold를 다시 선택하면 외부시험이 아니므로 별도 탐색 실험으로 표시해야 한다.
