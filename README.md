# Document Understanding - DocLayNet YOLO

DocLayNet 데이터셋을 활용한 문서 레이아웃 객체 탐지 프로젝트입니다.

## 프로젝트 구조

```
Document_understanding/
├── utils/                 # 유틸리티 모듈 (NEW!)
│   ├── consts.py          # 프로젝트 상수 정의
│   ├── image_utils.py     # 이미지 전처리
│   └── visualization.py   # 결과 시각화
├── scripts/               # 데이터 처리 및 학습 스크립트
│   ├── 0_eda.py           # 데이터 탐색
│   ├── 0_splitdataset.py  # 데이터셋 분할
│   ├── 1_download_dataset.py  # 데이터셋 다운로드
│   ├── 2_preprocess_dataset.py  # 전처리
│   ├── 3_build_doclaynet_yolo.py  # YOLO 포맷 변환
│   ├── 4_train.py         # 모델 학습
│   └── 5_test.py          # 모델 테스트 (개선됨!)
├── config/                # 설정 파일 (YAML)
├── data/                  # 데이터셋 (gitignore)
├── runs/                  # 학습 결과 (gitignore)
├── out/                   # 테스트 결과 샘플
└── out_compare/           # 모델 비교 결과 샘플
```

## 주요 개선 사항 (v2.0)

### 1. 모듈화된 유틸리티
- **consts.py**: DocLayNet 카테고리, 색상 팔레트, 기본 설정 관리
- **image_utils.py**: 이미지 전처리 (대비/선명도 향상, 노이즈 제거)
- **visualization.py**: 탐지 결과 시각화, 모델 비교, 통계 표시

### 2. 개선된 테스트 스크립트
- 배치 처리 지원
- 여러 모델 자동 비교
- 성능 메트릭 계산 (추론 시간, 탐지 수)
- Side-by-side 비교 이미지 생성
- 카테고리별 통계 시각화

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

### 5. 테스트 (개선된 버전!)
```bash
# 기본 사용 (5장, 기본 설정)
python scripts/5_test.py

# 커스텀 설정
python scripts/5_test.py \
    --num_images 10 \
    --output_dir results \
    --conf 0.3 \
    --iou 0.5 \
    --preprocess
```

#### 테스트 옵션
- `--num_images N`: 테스트할 이미지 수 (기본: 5)
- `--output_dir DIR`: 결과 저장 디렉터리 (기본: out_compare)
- `--test_dir DIR`: 테스트 이미지 경로
- `--conf FLOAT`: Confidence threshold (기본: 0.25)
- `--iou FLOAT`: IoU threshold for NMS (기본: 0.45)
- `--preprocess`: 이미지 전처리 활성화

## DocLayNet 카테고리 (11 classes)

| ID | Category | Description |
|----|----------|-------------|
| 0 | Caption | 그림/표 캡션 |
| 1 | Footnote | 각주 |
| 2 | Formula | 수식 |
| 3 | List-item | 목록 항목 |
| 4 | Page-footer | 페이지 하단 |
| 5 | Page-header | 페이지 상단 |
| 6 | Picture | 이미지/그림 |
| 7 | Section-header | 섹션 제목 |
| 8 | Table | 표 |
| 9 | Text | 본문 텍스트 |
| 10 | Title | 문서 제목 |

## 주요 기능

- DocLayNet 데이터셋 자동 다운로드 및 전처리
- YOLO 포맷 변환 (bbox 좌표 변환 및 정규화)
- 병렬 처리 (multiprocessing)
- 분산 컴퓨팅 지원 (PySpark, MPI)
- **이미지 전처리** (대비/선명도 향상, 노이즈 제거)
- **자동 모델 비교** (YOLOv8n vs YOLO11n)
- **시각화** (bbox, 카테고리, 통계)

## 기술 스택

- **ML/DL**: YOLO (Ultralytics), OpenCV, PIL
- **Data**: HuggingFace Datasets, NumPy, Pandas
- **Parallel Computing**: Numba, multiprocessing
- **Distributed Computing**: PySpark, MPI4Py
- **Visualization**: Matplotlib, OpenCV

## 예제 결과

테스트 결과는 [out_compare/](out_compare/) 디렉터리에서 확인할 수 있습니다.

## 참고 자료

- [DocLayNet Dataset](https://huggingface.co/datasets/ds4sd/DocLayNet)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Inspired by [2025 Samsung AI Challenge](https://github.com/dlxogh0906/2025-Samsung-AI-Challenge)

## 요구사항

- Python 3.8+
- CUDA 11.0+ (GPU 사용 시)
- 16GB+ RAM 권장

## 라이선스

MIT License
