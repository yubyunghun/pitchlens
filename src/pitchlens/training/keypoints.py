"""Turn calibrated frames into training data for a pitch-landmark (keypoint) model.

A calibrated frame knows where *all* 33 pitch landmarks are in the image, not
just the few that were clicked: projecting every landmark through the
calibration labels each visible one. And with camera tracking, one keyframe
calibrates a whole shot, so every Nth calibrated frame becomes an example.

Output is a YOLO pose dataset:
    out/data.yaml        kpt_shape [33, 3], flip_idx, names {0: pitch}
    out/train/images, out/train/labels, out/valid/...
Each label: "0 cx cy w h" (box around the visible landmarks) followed by
33 x (x y v): v = 2 visible, 0 not in view (x = y = 0).

    pitchlens-keypoint-data --video data/raw/match.mp4 --calibration configs/calibration.json
    pitchlens-train --task pose --data datasets/pitch_keypoints
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml

from pitchlens.calibration.auto_calib import KEYPOINT_NAMES
from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.pitch import Pitch


def flip_index(pitch: Pitch = Pitch()) -> list[int]:
    """For a horizontally flipped image: which landmark each landmark becomes (left <-> right)."""
    kp = pitch.keypoints()
    pts = np.array([kp[n] for n in KEYPOINT_NAMES])
    mirrored = np.c_[pitch.length - pts[:, 0], pts[:, 1]]
    return [int(np.argmin(np.linalg.norm(pts - m, axis=1))) for m in mirrored]


def keypoint_label(calib: PitchCalibration, image_size: tuple[int, int], min_visible: int = 4,
                   pad: float = 0.02) -> str | None:
    """YOLO pose label line for one calibrated frame, or None if too few landmarks are in view."""
    w, h = image_size
    kp = calib.pitch.keypoints()
    img = calib.to_image(np.array([kp[n] for n in KEYPOINT_NAMES]))
    visible = np.isfinite(img).all(axis=1) & (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
    if visible.sum() < min_visible:
        return None
    vis = img[visible]
    x1, y1 = np.clip(vis.min(axis=0) - pad * np.array([w, h]), 0, [w, h])
    x2, y2 = np.clip(vis.max(axis=0) + pad * np.array([w, h]), 0, [w, h])
    parts = [0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    for (x, y), v in zip(img, visible):
        parts += [x / w, y / h, 2] if v else [0, 0, 0]
    return " ".join(str(p) if isinstance(p, int) else f"{p:.6f}" for p in parts)


def write_data_yaml(out: Path, pitch: Pitch = Pitch()) -> Path:
    data = {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
            "kpt_shape": [len(KEYPOINT_NAMES), 3], "flip_idx": flip_index(pitch), "names": {0: "pitch"},
            "keypoint_names": KEYPOINT_NAMES}
    path = out / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def export_keypoint_dataset(frames: Iterable[np.ndarray | None], calibs: Sequence[PitchCalibration | None],
                            out: str | Path, every: int = 10, val_every: int = 5, prefix: str = "clip") -> dict:
    """Writes every `every`-th calibrated frame as an example; every `val_every`-th example goes to valid/.

    frames: one entry per video frame, can be a generator; entries may be None for frames that
    won't be used, so they never need to be decoded or kept in memory.
    """
    out = Path(out)
    counts = {"train": 0, "valid": 0, "skipped": 0}
    n = 0
    for t, (frame, calib) in enumerate(zip(frames, calibs)):
        if calib is None or frame is None or t % every:
            continue
        h, w = frame.shape[:2]
        label = keypoint_label(calib.for_image_size(w, h), (w, h))
        if label is None:
            counts["skipped"] += 1
            continue
        split = "valid" if n % val_every == val_every - 1 else "train"
        n += 1
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_{t:06d}"
        cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), frame)
        (out / split / "labels" / f"{stem}.txt").write_text(label + "\n")
        counts[split] += 1
    write_data_yaml(out, calibs[0].pitch if calibs and calibs[0] is not None else Pitch())
    return counts


def calibrations_for_video(video: str, calibration_path: str) -> list[PitchCalibration | None]:
    """Per-frame calibrations for a whole clip: keyframes carried through by camera tracking."""
    import supervision as sv

    from pitchlens.calibration.anchored import solve_calibrations
    from pitchlens.calibration.camera_motion import CameraMotionTracker
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.calibration.shot_cuts import AppearanceChange, find_cuts

    info = sv.VideoInfo.from_video_path(video)
    keys = CalibrationSet.load(calibration_path).for_image_size(info.width, info.height)
    tracker, appearance = CameraMotionTracker(mask_non_grass=True), AppearanceChange()   # no detector here
    motions, changes = [], []
    for frame in sv.get_video_frames_generator(video):
        result = tracker.update(frame)
        motions.append(None if result is None else result.H)
        changes.append(appearance.update(frame))
    cuts = find_cuts(changes, [m is not None for m in motions])
    solved = solve_calibrations(motions, keys, (info.width, info.height), breaks=cuts)
    return [None if s is None else s.calib for s in solved]


def main() -> None:
    import supervision as sv

    parser = argparse.ArgumentParser(description="Make pitch-landmark training data from calibrated video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", default="configs/calibration.json")
    parser.add_argument("--out", default="datasets/pitch_keypoints")
    parser.add_argument("--every", type=int, default=10, help="use every Nth calibrated frame")
    args = parser.parse_args()
    calibs = calibrations_for_video(args.video, args.calibration)
    wanted = {t for t, c in enumerate(calibs) if c is not None and t % args.every == 0}
    # streamed: frames are read and written one at a time, never all held in memory
    frames = (f if t in wanted else None for t, f in enumerate(sv.get_video_frames_generator(args.video)))
    counts = export_keypoint_dataset(frames, calibs, args.out, args.every, prefix=Path(args.video).stem)
    print(f"Wrote {counts['train']} train / {counts['valid']} valid examples to {args.out} "
          f"({counts['skipped']} frames with too few landmarks in view skipped)")


if __name__ == "__main__":
    main()
