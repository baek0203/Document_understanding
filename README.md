# Document Understanding - DocLayNet YOLO

DocLayNet 데이터셋을 활용한 문서 레이아웃 객체 탐지 프로젝트입니다.

## 프로젝트 구조

```
Document_understanding/
├── scripts/           # 데이터 처리 및 학습 스크립트
├── config/           # 설정 파일 (YAML)
├── data/             # 데이터셋 (gitignore)
├── runs/             # 학습 결과 (gitignore)
└── out/              # 출력 파일 (gitignore)
```

## 설치 방법

### 1. 가상환경 생성 및 활성화
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터셋 다운로드
```bash
python scripts/1_download_dataset.py
```

### 2. 데이터 전처리
```bash
python scripts/2_preprocess_dataset.py
```

### 3. YOLO 포맷으로 변환
```bash
python scripts/3_build_doclaynet_yolo.py
```

### 4. 학습
```bash
python scripts/4_train.py --config config/yolo_train.yaml
```

### 5. 테스트
```bash
python scripts/5_test.py
```

## 주요 기능

- DocLayNet 데이터셋 자동 다운로드 및 전처리
- YOLO 포맷 변환 (bbox 좌표 변환 및 정규화)
- 병렬 처리 (multiprocessing)
- 분산 컴퓨팅 지원 (PySpark, MPI)

## 기술 스택

- **ML/DL**: YOLO (Ultralytics), OpenCV, PIL
- **Data**: HuggingFace Datasets, NumPy, Pandas
- **Parallel Computing**: Numba, multiprocessing
- **Distributed Computing**: PySpark, MPI4Py

## 요구사항

- Python 3.8+
- CUDA (GPU 사용 시)

## 라이선스

MIT License
