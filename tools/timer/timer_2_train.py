"""
STEP 2 (TIMER): Train YOLOv8 on the synthetic timer dataset
-----------------------------------------------------------
Trains a single-class ("timer") detector on the synthetic timer dataset.

Dataset format: the standard Ultralytics YOLO detection layout. --data points at a
dataset.yaml declaring the train/val image directories and the single class name
("timer"); each image has a matching label .txt with one "0 cx cy w h" line per
timer-widget box, coordinates normalized to the image. Any dataset in that shape
trains.

Output goes to runs/timer/detect: the logo and character models in
runs/logo and runs/characters are left untouched.

Usage:
    python timer_2_train.py --data dataset_timer/dataset.yaml

The timer is one big, high-contrast object, so the nano model (default) is
plenty. Pass --model s for a little more headroom.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def train(data_yaml: str, epochs: int, batch: int, model_size: str, patience: int,
          project: str, name: str):
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Metal (MPS): fast training!")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA CUDA GPU")
    else:
        device = "cpu"
        print("No GPU found; using CPU (will be slower)")

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        augment=True,
        patience=patience,
        save=True,
        project=project,
        name=name,
        exist_ok=True,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    return results


# Repo-root runs/timer (tools/timer/timer_2_train.py -> parents[2] is the repo root).
DEFAULT_PROJECT = str(Path(__file__).resolve().parents[2] / "runs" / "timer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=16,
                        help="Batch size (reduce to 8 if you get memory errors)")
    parser.add_argument("--model",    default="n",
                        help="YOLOv8 size: n (nano), s (small), m (medium)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience (epochs without val improvement)")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="Output project dir (default: repo-root runs/timer)")
    parser.add_argument("--name", default="detect",
                        help="Run name under --project (use a new name to avoid clobbering)")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch, args.model, args.patience,
          args.project, args.name)
