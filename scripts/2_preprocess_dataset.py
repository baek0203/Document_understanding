import yaml
from datasets import load_dataset
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import numpy as np
import numba
import h5py
import os


# =============================
#      1. YAML LOAD
# =============================
cfg_data = yaml.safe_load(open("config/data.yaml"))
cfg_sys = yaml.safe_load(open("config/system.yaml"))

DATASET_NAME = cfg_data["dataset"]["name"]
CACHE_DIR = cfg_data["dataset"]["cache_dir"]
SPLITS = cfg_data["dataset"]["splits"]  # 이제 train, validation, test 모두 처리

WORKERS = cfg_sys["system"]["workers"]
USE_VEC = cfg_sys["system"]["use_vectorization"]
USE_HDF5 = cfg_sys["system"]["use_hdf5"]
H5_ROOT = cfg_sys["paths"]["hdf5_output"]  # 예: "/mnt/d/doclaynet_preprocessed"


# =============================
#   2. SIMD (Vectorization)
# =============================
@numba.jit(nopython=True)
def convert_bbox_numba(bboxes, img_w, img_h):
    out = np.zeros((len(bboxes), 4), dtype=np.float32)
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        out[i] = (xc, yc, bw, bh)
    return out


# =============================
#    3. 단일 샘플 처리 함수
# =============================
def process_one(args):
    idx, example = args

    img = example["image"].convert("RGB")
    img_np = np.array(img)  # variable size allowed

    h, w = img_np.shape[:2]

    # bbox → YOLO normalized
    if USE_VEC:
        bbox_arr = np.array(example["bboxes"], dtype=np.float32)
        yolo_boxes = convert_bbox_numba(bbox_arr, w, h)
    else:
        yolo_boxes = []
        for b in example["bboxes"]:
            x1, y1, x2, y2 = b
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_boxes.append([xc, yc, bw, bh])
        yolo_boxes = np.array(yolo_boxes, dtype=np.float32)

    labels = np.array(example["category_id"], dtype=np.int32)

    return img_np, yolo_boxes, labels


# =============================
#     4. Split 처리 함수
# =============================
def process_split(split_name):
    print(f"\n===========================")
    print(f"▶ Processing split: {split_name}")
    print(f"===========================")

    ds = load_dataset(
        DATASET_NAME,
        split=split_name,
        cache_dir=CACHE_DIR
    )

    print(f"▶ Loaded {len(ds)} samples for {split_name}")

    # 저장 파일 이름 구성
    save_path = f"{H5_ROOT}_{split_name}.h5"  
    print(f"▶ Saving to: {save_path}")

    # =============================
    # HDF5 VARIABLE LENGTH 저장소 초기화
    # =============================
    if USE_HDF5:
        h5 = h5py.File(save_path, "w")

        dt_img = h5py.special_dtype(vlen=np.dtype('uint8'))
        img_ds = h5.create_dataset("images", (len(ds),), dtype=dt_img)

        bbox_ds = h5.create_dataset("bboxes", (len(ds), 50, 4), dtype="float32")
        label_ds = h5.create_dataset("labels", (len(ds), 50), dtype="int32")

        print("▶ HDF5 datasets initialized.")

    # =============================
    # 병렬 처리 시작
    # =============================
    print("▶ Starting multiprocessing...")

    with Pool(WORKERS) as p:
        for idx, (img, bbox, label) in enumerate(
            p.imap(process_one, list(enumerate(ds)))
        ):

            if USE_HDF5:
                # variable-length bytes 형태로 저장
                img_bytes = img.tobytes()
                img_ds[idx] = np.frombuffer(img_bytes, dtype=np.uint8)

                bbox_ds[idx, :len(bbox)] = bbox
                label_ds[idx, :len(label)] = label

            if idx % 500 == 0:
                print(f"   Processed {idx}/{len(ds)}")

    if USE_HDF5:
        h5.close()
        print(f"▶ Split saved: {save_path}")


# =============================
#     5. 메인 실행
# =============================
def main():
    for split in SPLITS:
        process_split(split)

    print("\n===========================")
    print("▶ All splits processed successfully!")
    print("===========================\n")


if __name__ == "__main__":
    main()
