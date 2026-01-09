#!/usr/bin/env python3
"""
DocLayNet → YOLO 변환 스크립트 (최종 안정화 버전)

핵심 수정사항:
- DocLayNet bbox = (x, y, w, h) in page coords → YOLO (xc, yc, w, h) [0~1]
- category_id 1-based → YOLO 0-based 변환
- coco_width / coco_height 로 스케일 보정
- numba 제거 (object-array 문제 해결)
- bboxes shape 강제 (N,4) float32
- multiprocessing 안전 동작
"""

import yaml
from datasets import load_dataset
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import numpy as np


GLOBAL_DS = None
GLOBAL_IMG_DIR = None
GLOBAL_LABEL_DIR = None
GLOBAL_SAVE_FORMAT = "jpg"


# ============================================================
# bbox 변환: DocLayNet xywh → YOLO xc,yc,w,h
# ============================================================
def convert_bbox_xywh_to_yolo(
    bboxes: np.ndarray,
    page_w: float,
    page_h: float,
    img_w: float,
    img_h: float,
) -> np.ndarray:

    # bboxes를 확실하게 N x 4 float32로 강제 변환
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)

    out = np.zeros((len(bboxes), 4), dtype=np.float32)

    # 페이지 → 실제 이미지 좌표 스케일링
    sx = img_w / page_w
    sy = img_h / page_h

    for i in range(len(bboxes)):
        x, y, bw_page, bh_page = bboxes[i]

        # 페이지 좌표 → 실제 이미지 좌표
        x_img = x * sx
        y_img = y * sy
        bw_img = bw_page * sx
        bh_img = bh_page * sy

        # YOLO normalized format
        xc = (x_img + bw_img / 2) / img_w
        yc = (y_img + bh_img / 2) / img_h
        bw = bw_img / img_w
        bh = bh_img / img_h

        out[i] = (xc, yc, bw, bh)

    return out


# ============================================================
# 단일 샘플 처리
# ============================================================
def process_one(index: int):
    global GLOBAL_DS, GLOBAL_IMG_DIR, GLOBAL_LABEL_DIR, GLOBAL_SAVE_FORMAT

    example = GLOBAL_DS[index]

    # 이미지 읽기
    img: Image.Image = example["image"].convert("RGB")
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]

    # 원본 페이지 크기 (coco_width, coco_height)
    meta = example.get("metadata", {})
    page_w = float(meta.get("coco_width", img_w))
    page_h = float(meta.get("coco_height", img_h))

    # bbox + label
    bboxes = np.asarray(example["bboxes"], dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(example["category_id"], dtype=np.int32)

    # bbox 변환
    yolo_boxes = convert_bbox_xywh_to_yolo(
        bboxes,
        page_w=page_w,
        page_h=page_h,
        img_w=img_w,
        img_h=img_h,
    )

    # 파일 이름
    stem = f"{index:06d}"
    img_path = GLOBAL_IMG_DIR / f"{stem}.{GLOBAL_SAVE_FORMAT}"
    label_path = GLOBAL_LABEL_DIR / f"{stem}.txt"

    # 이미지 저장
    img.save(img_path)

    # YOLO label 저장: class_id 1→0
    lines = []
    for box, cls in zip(yolo_boxes, labels):
        cls_id = int(cls) - 1
        if cls_id < 0:
            continue
        xc, yc, bw, bh = box.tolist()
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")

    return index


# ============================================================
# train / val / test 변환
# ============================================================
def process_split(dataset_name, cache_dir, split_name, yolo_split, output_root, workers):
    global GLOBAL_DS, GLOBAL_IMG_DIR, GLOBAL_LABEL_DIR

    print(f"\n=== Processing split: {split_name} → {yolo_split} ===")

    ds = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
    GLOBAL_DS = ds

    # 디렉터리 생성
    img_dir = output_root / "images" / yolo_split
    label_dir = output_root / "labels" / yolo_split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    GLOBAL_IMG_DIR = img_dir
    GLOBAL_LABEL_DIR = label_dir

    indices = list(range(len(ds)))

    # multiprocessing
    with Pool(processes=workers) as pool:
        for i, idx in enumerate(pool.imap(process_one, indices, chunksize=32), start=1):
            if i % 500 == 0:
                print(f"   processed {i}/{len(ds)}")

    print(f"✓ Finished split: {split_name}")


# ============================================================
# main
# ============================================================
def main():
    cfg_data = yaml.safe_load(open("config/data.yaml"))
    cfg_sys = yaml.safe_load(open("config/system.yaml"))

    dataset_name = cfg_data["dataset"]["name"]
    cache_dir = cfg_data["dataset"]["cache_dir"]
    splits = cfg_data["dataset"]["splits"]
    output_root = Path(cfg_data["output"]["root"])
    workers = int(cfg_sys["system"]["workers"])

    split_map = {"train": "train", "validation": "val", "test": "test"}

    print("\nDocLayNet → YOLO 변환 시작")

    for split in splits:
        yolo_split = split_map.get(split, split)
        process_split(dataset_name, cache_dir, split, yolo_split, output_root, workers)

    print("\n🎉 모든 split 변환 완료!")


if __name__ == "__main__":
    main()
