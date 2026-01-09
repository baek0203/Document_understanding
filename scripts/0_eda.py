import cv2
import os
import glob
import random

# 경로 설정
img_dir = "data/doclaynet_yolo_split/images/train/"
label_dir = "data/doclaynet_yolo_split/labels/train/"
save_dir = "debug_5/"
os.makedirs(save_dir, exist_ok=True)

# 라벨 파일 5개 랜덤 선택
label_files = glob.glob(os.path.join(label_dir, "*.txt"))
sampled = random.sample(label_files, 5)

print("샘플 5개 라벨 파일:")
for lf in sampled:
    print(" -", lf)

def draw_yolo_bbox(img, bbox, cls):
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox

    # YOLO → 픽셀 변환
    cx *= w
    cy *= h
    bw *= w
    bh *= h

    x1 = int(cx - bw/2)
    y1 = int(cy - bh/2)
    x2 = int(cx + bw/2)
    y2 = int(cy + bh/2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, str(cls), (x1, y1-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

for label_path in sampled:
    img_name = os.path.basename(label_path).replace(".txt", ".jpg")
    img_path = os.path.join(img_dir, img_name)

    if not os.path.exists(img_path):
        print("❌ 이미지 없음:", img_path)
        continue

    img = cv2.imread(img_path)
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        draw_yolo_bbox(img, (cx, cy, bw, bh), int(cls))

    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, img)

print("\n완료! → debug_5/ 폴더에서 결과 확인하세요.")
