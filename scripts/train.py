#!/usr/bin/env python3
"""Thin Ultralytics wrapper: fine-tune YOLO11n on the crack dataset.

Loads yolo11n.pt, trains on data.yaml, copies the best weights to
models/best.pt. Also prints the default augmentations Ultralytics
applies at train time -- important because Label Studio only
annotates, it does not augment. So all augmentation here is
Ultralytics' job.

Example:
    python train.py --data ../dataset/data.yaml --epochs 50 --imgsz 640
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

# Ultralytics' built-in default augmentations (from the training pipeline).
# These are applied on-the-fly every epoch -- Label Studio does none of this.
DEFAULT_AUGS = [
    ("hsv_h", 0.015, "hue jitter (fraction)"),
    ("hsv_s", 0.7, "saturation jitter (fraction)"),
    ("hsv_v", 0.4, "brightness/value jitter (fraction)"),
    ("degrees", 0.0, "rotation (deg) -- off by default"),
    ("translate", 0.1, "translation (fraction of image)"),
    ("scale", 0.5, "scale gain (zoom in/out)"),
    ("shear", 0.0, "shear (deg) -- off by default"),
    ("perspective", 0.0, "perspective -- off by default"),
    ("flipud", 0.0, "vertical flip prob -- off by default"),
    ("fliplr", 0.5, "horizontal flip prob"),
    ("mosaic", 1.0, "4-image mosaic prob"),
    ("mixup", 0.0, "mixup prob -- off by default"),
    ("copy_paste", 0.0, "segment copy-paste -- off by default"),
    ("erasing", 0.4, "random erasing prob (classification head)"),
]


def print_augs():
    """Show the augmentations Ultralytics adds so the user knows the pipeline."""
    print("\nUltralytics default train-time augmentations")
    print("(Label Studio does NOT augment -- this is where it happens)")
    print(f"\n{'param':<14}{'default':>9}   description")
    print("-" * 60)
    for name, val, desc in DEFAULT_AUGS:
        print(f"{name:<14}{val:>9}   {desc}")
    print("-" * 60)
    print("Override any of these via CLI, e.g. --degrees 10 --mixup 0.1\n")


def main():
    ap = argparse.ArgumentParser(description="Train YOLO11n on the crack dataset.")
    ap.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="Base checkpoint")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda idx, 'cpu', or None=auto")
    ap.add_argument("--name", default="crack_yolo11n", help="Run name under runs/")
    ap.add_argument("--out", type=Path, default=Path("models"), help="Where to copy best.pt")
    # Accept a couple of common augmentation overrides as pass-throughs.
    ap.add_argument("--degrees", type=float, default=None)
    ap.add_argument("--mixup", type=float, default=None)
    args = ap.parse_args()

    print_augs()

    # Only forward augmentation overrides the user actually set.
    extra = {}
    if args.degrees is not None:
        extra["degrees"] = args.degrees
    if args.mixup is not None:
        extra["mixup"] = args.mixup

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        **extra,
    )

    # Ultralytics writes best.pt under <save_dir>/weights/best.pt.
    best = Path(results.save_dir) / "weights" / "best.pt"
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "best.pt"
    if best.exists():
        shutil.copy2(best, dest)
        print(f"\nBest weights -> {dest.resolve()}")
    else:
        print(f"\nWARNING: expected weights not found at {best}")


if __name__ == "__main__":
    main()
