import random
import shutil
from pathlib import Path

SRC_ROOT = Path("data/doclaynet_yolo")          # 기존 데이터
DST_ROOT = Path("data/doclaynet_yolo_split")    # 새로 만들 데이터

N_TRAIN = 8000
N_VAL   = 1000
N_TEST  = 1000

random.seed(42)

def make_dirs(root):
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def split_from_train():
    img_train_dir = SRC_ROOT / "images/train"
    lbl_train_dir = SRC_ROOT / "labels/train"

    img_files = sorted(list(img_train_dir.glob("*.jpg")))
    # 1:1 매칭이 있는 것만 필터링
    paired_imgs = [p for p in img_files
                   if (lbl_train_dir / f"{p.stem}.txt").exists()]

    assert len(paired_imgs) >= N_TRAIN + N_VAL + N_TEST, "쌍이 맞는 이미지 수가 부족함"

    random.shuffle(paired_imgs)
    test_imgs  = paired_imgs[:N_TEST]
    val_imgs   = paired_imgs[N_TEST:N_TEST+N_VAL]
    train_imgs = paired_imgs[N_TEST+N_VAL:N_TEST+N_VAL+N_TRAIN]

    def copy_pair(files, split):
        img_dst = DST_ROOT / f"images/{split}"
        lbl_dst = DST_ROOT / f"labels/{split}"
        for img_path in files:
            stem = img_path.stem              # 예: 000000
            lbl_path = lbl_train_dir / f"{stem}.txt"
            # 여기서 항상 img/label 이름이 동일하게 복사됨
            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(lbl_path, lbl_dst   / lbl_path.name)

    copy_pair(train_imgs, "train")
    copy_pair(val_imgs,   "val")
    copy_pair(test_imgs,  "test")

if __name__ == "__main__":
    make_dirs(DST_ROOT)
    split_from_train()
