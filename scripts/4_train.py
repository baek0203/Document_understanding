#!/usr/bin/env python3
import yaml
import argparse
from ultralytics import YOLO


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/yolo_train.yaml")
    args = parser.parse_args()

    # Load YAML config
    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]

    print("\n============================")
    print("YOLO TRAINING CONFIG")
    print("============================")
    print(cfg)
    print("\n")

    # Load YOLO model
    model = YOLO(cfg["model"])

    # Train
    results = model.train(
        task=cfg.get("task", "detect"),
        data=cfg["dataset_yaml"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        workers=train_cfg["workers"],
        device=train_cfg["device"],
        project=train_cfg["project"],
        name=train_cfg["name"],
    )

    print("\n=====================================")
    print(" Training Complete!")
    print(" Results saved in:", results.save_dir)
    print("=====================================\n")


if __name__ == "__main__":
    main()
