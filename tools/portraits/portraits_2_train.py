"""
STEP 2 (CHARACTERS): Train YOLOv8 on Fighter Portraits
------------------------------------------------------
Trains a YOLOv8 model on the synthetic character-portrait dataset.

Dataset format: the standard Ultralytics YOLO detection layout. --data points at a
dataset.yaml declaring the train/val image directories and the class list (one
class per fighter, in the index order the labels use); each image has a matching
label .txt with one "class cx cy w h" line per HUD-portrait box, coordinates
normalized to the image. Any dataset in that shape trains.

Output goes to runs/characters/<run-name> at the repo root: the logo model in
runs/logo/detect is left untouched. --name is REQUIRED and the run aborts if that
name already exists (pass --force to overwrite), so a stray run can't clobber the
promoted production model (detect_v3) or the detect_v2 fallback.

Usage:
    python tools/portraits/portraits_2_train.py --name detect_v4 \
        --data tool_workspace/generated_data/dataset_characters/dataset.yaml
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def train(data_yaml: str, epochs: int, batch: int, model_size: str,
          patience: int, run_name: str, force: bool):
    runs_dir = Path(__file__).resolve().parents[2] / "runs" / "characters"
    if (runs_dir / run_name).exists() and not force:
        raise SystemExit(
            f"Run '{run_name}' already exists at {runs_dir / run_name}.\n"
            f"Pick a new --name (e.g. detect_v4) or pass --force to overwrite it.")
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Metal (MPS): fast training!")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA CUDA GPU")
    else:
        device = "cpu"
        print("No GPU found: using CPU (will be slower)")

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        augment=True,
        # The in-game HUD always shows portraits at one fixed orientation
        # (mirrored chara_4 art); random flips would erase that prior and
        # blur look-alike character pairs.
        fliplr=0.0,
        patience=patience,
        save=True,
        project=str(runs_dir),
        name=run_name,
        exist_ok=force,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: runs/characters/{run_name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=16,
                        help="Batch size (reduce to 8 if you get memory errors)")
    parser.add_argument("--model",    default="s",
                        help="YOLOv8 size: n (nano), s (small), m (medium)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience (epochs without val improvement)")
    parser.add_argument("--name",     required=True,
                        help="Run name under runs/characters/ (e.g. detect_v4). REQUIRED: "
                             "stops an unnamed run from clobbering an existing model.")
    parser.add_argument("--force",    action="store_true",
                        help="Overwrite --name if it already exists (default: abort).")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch, args.model, args.patience, args.name,
          args.force)
