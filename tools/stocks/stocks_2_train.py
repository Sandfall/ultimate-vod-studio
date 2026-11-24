"""
STEP 2 (STOCK ICONS): Train YOLOv8 on the Melee stock-icon dataset
-------------------------------------------------------------------
Trains the stock-icon detector (one class per character) on the synthetic
stock-icon dataset.

Dataset format: the standard Ultralytics YOLO detection layout. --data points at a
dataset.yaml declaring the train/val image directories and the class list (one class per
character, in the index order the labels use); each image has a matching label .txt with
one "class cx cy w h" line per stock icon, coordinates normalized to the image. Any
dataset in that shape trains.

The images are STRIP-SHAPED canvases (1280 wide, ~216 tall): exactly the bottom strip of
a frame that the runtime reader (vodparse.stock_icons) resizes to 1280 wide and runs the
detector on, so the ~45 px icons of a 1080p frame train and infer at the same ~30 px. Two
settings follow from that shape and are pinned here rather than left to the defaults:

  * --imgsz defaults to 1280 (vodparse.games.MELEE_STOCK_HUD.infer_width): at Ultralytics'
    usual 640 the strip would be downscaled again and the icons shrink to ~15 px, too
    small to tell look-alike heads apart.
  * rect=True: batches are shaped to the images' aspect ratio instead of letterboxing each
    wide strip into a 1280x1280 square (four times the pixels for nothing). Rectangular
    training turns Ultralytics' mosaic augmentation off; the generator supplies the layout
    variety (feed placement, port slots, icon counts) that mosaic would otherwise fake.
  * fliplr=0.0: the icons always face one way; a horizontal flip would erase that prior and
    blur mirrored look-alikes.

Output goes to runs/stocks/<run-name> at the repo root: the timer and character models in
runs/timer and runs/characters are left untouched. --name is REQUIRED and the run aborts if
that name already exists (pass --force to overwrite), so a stray run can't clobber a
promoted model.

An interrupted run (Ctrl-C, a crash, a laptop that went to sleep) is continued with
--resume: Ultralytics reloads runs/stocks/<run-name>/weights/last.pt together with its
optimizer state and every setting the run started with (dataset, epochs, image size, batch),
so --name is the only flag that matters and the others are ignored. A run that finished has
nothing left to continue (Ultralytics refuses to resume it); start a new --name instead.

Usage:
    python tools/stocks/stocks_2_train.py --name detect_v1 \\
        --data tool_workspace/generated_data/dataset_stocks/dataset.yaml

    # continue an interrupted run from its last.pt
    python tools/stocks/stocks_2_train.py --name detect_v1 --resume

Reproduce (Mac; the container has no GPU):
    python tools/stocks/stocks_2_train.py --name detect_v1 \\
        --data tool_workspace/generated_data/dataset_stocks/dataset.yaml
    cp runs/stocks/detect_v1/weights/best.pt weights/character_dx.pt
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# The inference width the runtime reader resizes the strip to (games.MELEE_STOCK_HUD.
# infer_width); repeated here as a literal so this script trains from a clean checkpoint
# without importing the package. Keep the two in step.
IMGSZ_DEFAULT = 1280

# At 1280 wide a batch of 16 strips no longer fits a laptop GPU's memory; 8 does.
BATCH_DEFAULT = 8

# Every run lives under here, beside runs/timer and runs/characters.
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs" / "stocks"

# Ultralytics writes this checkpoint at the end of every epoch, with the optimizer state a
# resume needs; best.pt is the best epoch's weights alone.
LAST_CHECKPOINT = Path("weights") / "last.pt"


def pick_device() -> str:
    """The fastest device this machine offers, announced on the console."""
    if torch.backends.mps.is_available():
        print("Using Apple Metal (MPS): fast training!")
        return "mps"
    if torch.cuda.is_available():
        print("Using NVIDIA CUDA GPU")
        return "cuda"
    print("No GPU found: using CPU (will be slower)")
    return "cpu"


def train(data_yaml: str, epochs: int, batch: int, model_size: str,
          patience: int, imgsz: int, run_name: str, force: bool):
    if (RUNS_DIR / run_name).exists() and not force:
        raise SystemExit(
            f"Run '{run_name}' already exists at {RUNS_DIR / run_name}.\n"
            f"Pick a new --name (e.g. detect_v2), pass --force to overwrite it, or "
            f"--resume to continue it.")
    device = pick_device()

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        rect=True,
        batch=batch,
        device=device,
        augment=True,
        fliplr=0.0,
        patience=patience,
        save=True,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=force,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: runs/stocks/{run_name}/weights/best.pt")
    return results


def resume(run_name: str):
    """Continue the interrupted run runs/stocks/<run_name> from its last.pt. Ultralytics
    restores the run's own settings from the checkpoint; only the device is chosen afresh,
    so a run can move between machines."""
    last = RUNS_DIR / run_name / LAST_CHECKPOINT
    if not last.is_file():
        raise SystemExit(
            f"Nothing to resume: {last} does not exist.\n"
            f"Start the run with --name {run_name} --data <dataset.yaml> instead.")
    device = pick_device()
    print(f"Resuming '{run_name}' from {last}")

    model = YOLO(str(last))
    results = model.train(resume=True, device=device)

    print("\nTraining complete!")
    print(f"Best model saved to: runs/stocks/{run_name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     help="Path to dataset.yaml (required unless --resume)")
    parser.add_argument("--epochs",   type=int, default=80)
    parser.add_argument("--batch",    type=int, default=BATCH_DEFAULT,
                        help="Batch size (reduce to 4 if you get memory errors)")
    parser.add_argument("--model",    default="s",
                        help="YOLOv8 size: n (nano), s (small), m (medium)")
    parser.add_argument("--imgsz",    type=int, default=IMGSZ_DEFAULT,
                        help="Training image size: the strip's width, matching the runtime "
                             "reader's inference width")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience (epochs without val improvement)")
    parser.add_argument("--name",     required=True,
                        help="Run name under runs/stocks/ (e.g. detect_v1). REQUIRED: "
                             "stops an unnamed run from clobbering an existing model.")
    parser.add_argument("--force",    action="store_true",
                        help="Overwrite --name if it already exists (default: abort).")
    parser.add_argument("--resume",   action="store_true",
                        help="Continue the interrupted run --name from its last.pt; the run's "
                             "own settings are restored and the other flags are ignored.")
    args = parser.parse_args()

    if args.resume:
        resume(args.name)
    else:
        if not args.data:
            parser.error("--data is required to start a run (or pass --resume)")
        train(args.data, args.epochs, args.batch, args.model, args.patience, args.imgsz,
              args.name, args.force)
