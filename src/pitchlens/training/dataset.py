"""Check a football detection dataset before training on it.

Expects the standard YOLO layout (as exported by Roboflow and similar):

    dataset/
      data.yaml                 names: [ball, goalkeeper, player, referee]  (or a {id: name} map)
      train/images/*.jpg   train/labels/*.txt
      valid/images/...     valid/labels/...      ("val" also accepted)
      test/images/...      test/labels/...       (optional)

Each label line is "class cx cy w h", coordinates normalized to 0..1.

The report covers what goes wrong in practice: labels without images, broken
lines, classes the data.yaml doesn't know, and whether the dataset's class
names map onto PitchLens's ball / goalkeeper / player / referee. It also
measures how big the ball is in pixels at the training image size: a ball of
only a few pixels needs a larger `imgsz` to be learnt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from pitchlens.detection.classes import BALL, CLASS_NAMES, map_class_names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}


@dataclass
class SplitReport:
    name: str
    images: int = 0
    labels: int = 0
    background: int = 0                       # images without a label file (allowed: no objects)
    orphan_labels: list[str] = field(default_factory=list)   # label files without an image
    bad_lines: list[str] = field(default_factory=list)       # "file:line: problem"
    boxes: Counter = field(default_factory=Counter)          # dataset class id -> count
    ball_widths_px: list[float] = field(default_factory=list)  # at the training image size


@dataclass
class DatasetReport:
    root: Path
    names: dict[int, str]
    mapping: dict[int, int]                   # dataset class id -> PitchLens class id
    splits: dict[str, SplitReport]

    @property
    def problems(self) -> list[str]:
        out = []
        if "train" not in self.splits:
            out.append("no train split found")
        if "val" not in self.splits:
            out.append("no validation split found (valid/ or val/)")
        for wanted in (BALL,):
            if wanted not in self.mapping.values():
                out.append(f"no dataset class maps to PitchLens '{CLASS_NAMES[wanted]}'")
        for s in self.splits.values():
            if s.images == 0:
                out.append(f"{s.name}: no images")
            if s.orphan_labels:
                out.append(f"{s.name}: {len(s.orphan_labels)} label file(s) without an image")
            if s.bad_lines:
                out.append(f"{s.name}: {len(s.bad_lines)} broken label line(s)")
        return out


def load_names(root: Path) -> dict[int, str]:
    data = yaml.safe_load((root / "data.yaml").read_text()) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def find_splits(root: Path) -> dict[str, Path]:
    out = {}
    for split, candidates in SPLIT_DIRS.items():
        for c in candidates:
            if (root / c / "images").is_dir():
                out[split] = root / c
                break
    return out


def _image_size(path: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)   # half size is enough to get the size
    if img is None:
        return None
    return img.shape[1] * 2, img.shape[0] * 2


def check_split(name: str, folder: Path, n_classes: int, ball_ids: set[int] = frozenset(),
                imgsz: int = 1280, size_samples: int = 200) -> SplitReport:
    """ball_ids: dataset class ids that are the ball (their size is measured on up to size_samples images)."""
    report = SplitReport(name)
    images = {p.stem: p for p in (folder / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels_dir = folder / "labels"
    labels = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.is_dir() else {}
    report.images, report.labels = len(images), len(labels)
    report.background = len(set(images) - set(labels))
    report.orphan_labels = sorted(set(labels) - set(images))

    sized = 0
    for stem, label in sorted(labels.items()):
        size = None
        for n, line in enumerate(label.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            where = f"{label.name}:{n}"
            if len(parts) != 5:
                report.bad_lines.append(f"{where}: expected 5 values, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                report.bad_lines.append(f"{where}: not numbers")
                continue
            if not 0 <= cls < n_classes:
                report.bad_lines.append(f"{where}: class {cls} not in data.yaml")
                continue
            if w <= 0 or h <= 0 or not all(0 <= v <= 1 for v in (cx, cy, w, h)):
                report.bad_lines.append(f"{where}: box outside 0..1 or empty")
                continue
            report.boxes[cls] += 1
            if cls in ball_ids and stem in images and sized < size_samples:
                if size is None:
                    size = _image_size(images[stem])
                if size is not None:
                    scale = imgsz / max(size)                       # YOLO resizes the long side to imgsz
                    report.ball_widths_px.append(w * size[0] * scale)
        if size is not None:
            sized += 1
    return report


def inspect_dataset(root: str | Path, imgsz: int = 1280) -> DatasetReport:
    root = Path(root)
    names = load_names(root)
    mapping = map_class_names(names)
    ball_ids = {i for i, c in mapping.items() if c == BALL}
    splits = {s: check_split(s, folder, len(names), ball_ids, imgsz) for s, folder in find_splits(root).items()}
    return DatasetReport(root, names, mapping, splits)


def format_report(r: DatasetReport, imgsz: int = 1280) -> str:
    lines = [f"Dataset {r.root}", "Classes:"]
    for i, n in sorted(r.names.items()):
        target = CLASS_NAMES.get(r.mapping.get(i), None)
        lines.append(f"  {i}: {n:<12} -> {target if target else 'ignored by PitchLens'}")
    for s in r.splits.values():
        counts = ", ".join(f"{r.names.get(c, c)} {n}" for c, n in sorted(s.boxes.items()))
        lines.append(f"{s.name}: {s.images} images, {s.labels} labels "
                     f"({s.background} background), boxes: {counts or 'none'}")
        if s.ball_widths_px:
            w = np.array(s.ball_widths_px)
            lines.append(f"  ball width at imgsz {imgsz}: median {np.median(w):.1f} px, "
                         f"{(w < 8).mean():.0%} under 8 px")
        for line in (s.bad_lines[:5] + s.orphan_labels[:5]):
            lines.append(f"  ! {line}")
    problems = r.problems
    lines.append("Problems: " + ("none" if not problems else ""))
    lines += [f"  - {p}" for p in problems]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a YOLO-format football dataset before training")
    parser.add_argument("root", help="dataset folder containing data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280, help="training image size to measure the ball at")
    args = parser.parse_args()
    report = inspect_dataset(args.root, args.imgsz)
    print(format_report(report, args.imgsz))
    raise SystemExit(1 if report.problems else 0)


if __name__ == "__main__":
    main()
