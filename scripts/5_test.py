from ultralytics import YOLO
import cv2
import os
import glob

# -----------------------------
# 1. 두 모델 로드
# -----------------------------
model_y11 = YOLO("/mnt/d/bigdata_project/runs/doclaynet/yolo11n/weights/best.pt")
model_y8  = YOLO("/mnt/d/bigdata_project/runs/doclaynet/yolov8n_doclaynet4/weights/best.pt")

# 테스트 이미지 경로
test_dir = "/mnt/d/bigdata_project/data/doclaynet_yolo_split/images/test"

img_list = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
if len(img_list) == 0:
    raise FileNotFoundError(f"No JPG files found in: {test_dir}")

# 5장만 사용
img_list = img_list[:5]

# 출력 디렉토리
out_dir = "out_compare"
os.makedirs(out_dir, exist_ok=True)

for img_path in img_list:
    base = os.path.basename(img_path)
    stem, ext = os.path.splitext(base)

    print(f"Running inference on: {img_path}")

    # -----------------------------
    # YOLO11n 추론
    # -----------------------------
    res_y11 = model_y11(img_path)
    img_y11 = res_y11[0].plot()
    save_y11 = os.path.join(out_dir, f"{stem}_y11n{ext}")
    cv2.imwrite(save_y11, img_y11)
    print("Saved:", save_y11)

    # -----------------------------
    # YOLOv8n 추론
    # -----------------------------
    res_y8 = model_y8(img_path)
    img_y8 = res_y8[0].plot()
    save_y8 = os.path.join(out_dir, f"{stem}_y8n{ext}")
    cv2.imwrite(save_y8, img_y8)
    print("Saved:", save_y8)

print("\n=== Inference complete for 5 images (y11n vs y8n) ===")
