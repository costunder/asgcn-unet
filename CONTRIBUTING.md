# 기여 가이드

버그 수정과 재현성 개선을 환영합니다. 기본 단위 테스트는 공식 데이터셋이나 GPU를 요구하지
않으며 `tests/` 내부 fixture만 사용해야 합니다.

## 개발 환경

서버 실험과 모델 개발은 [README의 설치 및 실행](README.md#설치-및-실행)에 정의한
Linux Conda 환경을 사용합니다. Python 3.12.14, PyTorch 2.13.0+cu126 및 전이 의존성을
고정하고, 설치·다운로드·학습·평가에 같은 환경을 사용합니다.

```bash
conda activate asgcn
bash scripts/setup.sh
```

CPU 호환성 테스트만 개발하는 경우에는 별도 Conda 환경을 사용합니다. 이 환경은 고정된 서버
실험 profile과 구분하며, GPU 학습·평가 결과를 산출하는 용도로 사용하지 않습니다.

```bash
conda create -n asgcn-dev --override-channels -c conda-forge python=3.12 pip
conda activate asgcn-dev
python -m pip install -e ".[dev]"
```

GitHub Actions는 Linux 고정 Conda profile의 실제 설치·전체 테스트와 Ubuntu/Windows의
Python 3.10·3.11·3.12 호환성 matrix를 검사합니다. 테스트 연산은 CPU에서 실행되며,
GPU 실험 검증은 실제 데이터와 할당된 GPU에서 별도로 수행해야 합니다.

## 변경 전 검증

Pull request를 열기 전에 저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m ruff check .
python -m pytest -q
```

Windows와 Linux의 Python 3.10, 3.11, 3.12 조합은 GitHub Actions에서 `tests/` 내부 fixture만
사용해 검사합니다. 데이터 로더나 config 경로를 변경했다면 공식 데이터를 별도로 배치한 뒤
다음 CLI 검사도 수행합니다.

```bash
asgcn-unet inspect --config configs/train.json --samples 2
asgcn-unet inspect --config configs/hdr.json --samples 2
asgcn-unet inspect --config configs/aid.json --samples 2
```

## Pull request 원칙

- 한 PR에는 가능한 한 하나의 논리적 변경만 포함합니다.
- 모델, 손실 함수, 그래프 생성 또는 데이터 정렬을 변경하면 근거와 예상 영향을 설명합니다.
- 새로운 기능에는 `tests/` 내부 fixture를 사용하는 최소 단위 테스트를 추가합니다.
- CLI나 설정 형식이 바뀌면 README와 예제 config도 함께 갱신합니다.
- EventHDR 공식 eval과 EventAid-R은 gradient update나 hyperparameter tuning에 사용하지 않습니다.
  기본 프로토콜은 official eval을 마지막 epoch에서 한 번만 계산하고 EventAid-R은 그 뒤 외부평가에만
  사용합니다.
- 포매터가 아닌 Ruff 검사 결과를 기준으로 기존 코드 스타일을 유지합니다.

## 데이터와 생성물

공식 데이터셋, 압축 파일, checkpoint, 학습 로그와 대규모 출력은 Git에 커밋하지 않습니다.
재현에 필요한 메타데이터와 `tests/` 내부의 작은 fixture만 저장하세요. 버그 보고에 실제 데이터
일부가 필요하다면 먼저 배포 라이선스와 개인 정보 포함 여부를 확인하고, 가능하면 이를 대신하는
최소 fixture를 제공하세요.

## 데이터 로더 변경 시 주의사항

- EventAid-R 이벤트 구간의 target은 동일 번호가 아니라 다음 번호의 GT입니다.
- 프레임 속도를 상수로 가정하지 말고 제공된 timestamp 차이를 사용합니다.
- EventHDR 공개 H5 번호를 물리 장면 ID로 주장하지 않습니다. 공식 eval은 마지막 epoch의 내부 결과,
  EventAid-R은 외부 일반화 결과로 구분합니다.
- 데이터 형식 검증 실패는 가능한 한 파일명과 기대한 구조를 포함한 명확한 오류로 보고합니다.
