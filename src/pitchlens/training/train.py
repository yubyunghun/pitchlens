"""Train a football detection model (ball / goalkeeper / player / referee).

    pitchlens-train --data datasets/football          # dataset folder with data.yaml

Steps:
  1. check the dataset (training/dataset.py) and stop on problems unless --force;
  2. pick the image size: the ball is tiny, so if it would be under 8 px wide at
     640 the model trains at 1280 instead (slower, but it can actually see it);
  3. fine-tune a pretrained YOLO model with ultralytics;
  4. copy the best weights to models/football.pt and print how to use them.

Training needs a GPU to be practical; on CPU it works but takes many hours.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pitchlens.training.dataset import DatasetReport, format_report, inspect_dataset

MIN_BALL_PX = 8.0


@dataclass
class TrainSettings:
    base_model: str = "yolov8m.pt"     # pretrained starting point (n/s/m/l/x: faster -> more accurate)
    epochs: int = 100
    imgsz: int | None = None           # None = choose from the ball size in the dataset
    batch: int = -1                    # -1 = let ultralytics pick what fits in GPU memory
    patience: int = 20                 # stop early after this many epochs without improvement
    device: str | None = None
    project: str = "models/runs"
    name: str = "football"
    output: str = "models/football.pt"


def suggest_imgsz(report: DatasetReport, measured_at: int = 1280) -> int:
    """640 if the ball is big enough there, else 1280 (sizes in the report are measured at `measured_at`)."""
    widths = [w for s in report.splits.values() for w in s.ball_widths_px]
    if not widths:
        return 1280
    at_640 = float(np.median(widths)) * 640 / measured_at
    return 640 if at_640 >= MIN_BALL_PX else 1280


def build_train_args(data_yaml: Path, settings: TrainSettings, imgsz: int) -> dict:
    """Keyword arguments for ultralytics YOLO.train()."""
    args = dict(
        data=str(data_yaml),
        epochs=settings.epochs,
        imgsz=imgsz,
        batch=settings.batch,
        patience=settings.patience,
        project=settings.project,
        name=settings.name,
        exist_ok=True,
        # broadcast footage: keep colors (team kits matter), no vertical flips (no upside-down pitches)
        hsv_h=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
    )
    if settings.device is not None:
        args["device"] = settings.device
    return args


def _best_weights(model, settings: TrainSettings) -> Path | None:
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    if best and Path(best).exists():
        return Path(best)
    fallback = Path(settings.project) / settings.name / "weights" / "best.pt"
    return fallback if fallback.exists() else None


def train(dataset: str | Path, settings: TrainSettings = TrainSettings(), force: bool = False,
          yolo_cls=None) -> Path | None:
    """Returns the path of the trained weights, or None if the dataset check stopped it."""
    root = Path(dataset)
    report = inspect_dataset(root)
    print(format_report(report))
    if report.problems and not force:
        print("Fix the problems above, or pass --force to train anyway.")
        return None
    imgsz = settings.imgsz or suggest_imgsz(report)
    print(f"Training {settings.base_model} at imgsz {imgsz} for up to {settings.epochs} epochs...")

    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls  # imported here: loads PyTorch
    model = yolo_cls(settings.base_model)
    model.train(**build_train_args(root / "data.yaml", settings, imgsz))

    best = _best_weights(model, settings)
    if best is None:
        print("Training finished but no best.pt was found; check the ultralytics output above.")
        return None
    out = Path(settings.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"Saved {out}. To use it, set in configs/default.yaml:\n"
          f"  detection:\n    model: {out.as_posix()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a football detection model for PitchLens")
    parser.add_argument("--data", required=True, help="dataset folder containing data.yaml")
    parser.add_argument("--model", default=TrainSettings.base_model, help="pretrained base model")
    parser.add_argument("--epochs", type=int, default=TrainSettings.epochs)
    parser.add_argument("--imgsz", type=int, default=None, help="default: chosen from the ball size")
    parser.add_argument("--batch", type=int, default=TrainSettings.batch)
    parser.add_argument("--device", default=None, help='e.g. "0" for the first GPU, "cpu"')
    parser.add_argument("--output", default=TrainSettings.output)
    parser.add_argument("--force", action="store_true", help="train even if the dataset check finds problems")
    a = parser.parse_args()
    settings = TrainSettings(base_model=a.model, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                             device=a.device, output=a.output)
    raise SystemExit(0 if train(a.data, settings, a.force) else 1)


if __name__ == "__main__":
    main()
