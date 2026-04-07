from pathlib import Path

import cv2
import numpy as np
import pytest

from pitchlens.detection.classes import BALL, GOALKEEPER, PLAYER, REFEREE, map_class_names
from pitchlens.training.dataset import format_report, inspect_dataset


def _make_dataset(root: Path, names=("ball", "goalkeeper", "player", "referee"), splits=("train", "valid"),
                  as_dict=False):
    """Tiny YOLO-format dataset: 1920x1080 images, a ball of 20 px in each."""
    if as_dict:
        names_yaml = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    else:
        names_yaml = f"names: [{', '.join(names)}]\n"
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(f"nc: {len(names)}\n{names_yaml}")
    img = np.zeros((1080, 1920, 3), np.uint8)
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for k in range(3):
            cv2.imwrite(str(root / split / "images" / f"f{k}.jpg"), img)
            (root / split / "labels" / f"f{k}.txt").write_text(
                f"0 0.5 0.5 {20 / 1920:.6f} {20 / 1080:.6f}\n"     # ball, 20 px wide
                "2 0.3 0.6 0.02 0.08\n2 0.6 0.6 0.02 0.08\n"      # two players
                "3 0.4 0.5 0.02 0.08\n")                          # referee
    return root


def test_class_names_map_to_pitchlens():
    assert map_class_names(["ball", "goalkeeper", "player", "referee"]) == \
        {0: BALL, 1: GOALKEEPER, 2: PLAYER, 3: REFEREE}
    assert map_class_names({0: "person", 32: "sports ball", 5: "bus"}) == {0: PLAYER, 32: BALL}
    assert map_class_names(["Football", " Keeper "]) == {0: BALL, 1: GOALKEEPER}


def test_clean_dataset(tmp_path):
    report = inspect_dataset(_make_dataset(tmp_path / "ds"), imgsz=1280)
    assert set(report.splits) == {"train", "val"}
    train = report.splits["train"]
    assert train.images == 3 and train.labels == 3 and train.background == 0
    assert train.boxes == {0: 3, 2: 6, 3: 3}
    # 20 px ball on a 1920 px image, resized to 1280: about 13 px
    assert train.ball_widths_px and np.median(train.ball_widths_px) == pytest.approx(20 * 1280 / 1920, abs=0.5)
    assert report.problems == []
    text = format_report(report)
    assert "0: ball         -> ball" in text and "Problems: none" in text


def test_problems_are_reported(tmp_path):
    root = _make_dataset(tmp_path / "ds", as_dict=True)
    labels = root / "train" / "labels"
    (labels / "f0.txt").write_text("0 0.5 0.5 0.01\n"           # 4 values
                                   "7 0.5 0.5 0.01 0.01\n"      # class not in data.yaml
                                   "2 1.2 0.5 0.01 0.01\n"      # outside the image
                                   "x 0.5 0.5 0.01 0.01\n")     # not a number
    (labels / "ghost.txt").write_text("2 0.5 0.5 0.1 0.1\n")    # no image
    (root / "train" / "images" / "empty.jpg").write_bytes(
        cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))[1].tobytes())   # background image
    report = inspect_dataset(root)
    train = report.splits["train"]
    assert len(train.bad_lines) == 4 and train.orphan_labels == ["ghost"] and train.background == 1
    problems = report.problems
    assert any("broken label line" in p for p in problems)
    assert any("without an image" in p for p in problems)
    assert "f0.txt:2: class 7 not in data.yaml" in format_report(report)


def test_missing_validation_split_and_no_ball_class(tmp_path):
    report = inspect_dataset(_make_dataset(tmp_path / "ds", names=("person", "car"), splits=("train",)))
    problems = report.problems
    assert "no validation split found (valid/ or val/)" in problems
    assert "no dataset class maps to PitchLens 'ball'" in problems
    assert report.mapping == {0: PLAYER}


def test_val_folder_name_accepted(tmp_path):
    report = inspect_dataset(_make_dataset(tmp_path / "ds", splits=("train", "val", "test")))
    assert set(report.splits) == {"train", "val", "test"}
