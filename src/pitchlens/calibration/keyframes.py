"""Several calibrated frames ("keyframes") for one video, stored in one JSON file.

One calibration drifts slowly as camera motion is chained frame after frame,
and a camera cut breaks the chain completely. Calibrating a few extra frames
(e.g. the start of the clip and right after each cut) gives exact anchors to
re-sync to.

File format:
    {"version": 2, "video": "...", "keyframes": [{"frame": 0, "homography": ..., ...}, ...]}
The older single-calibration format (one calibration at the top level) still loads.
"""

from __future__ import annotations

import json
from pathlib import Path

from pitchlens.calibration.homography import PitchCalibration


class CalibrationSet:
    def __init__(self, keyframes: dict[int, PitchCalibration] | None = None, video: str | None = None):
        self.keyframes: dict[int, PitchCalibration] = dict(sorted((keyframes or {}).items()))
        self.video = video

    def __len__(self) -> int:
        return len(self.keyframes)

    @property
    def frames(self) -> list[int]:
        return list(self.keyframes)

    def add(self, frame: int, calib: PitchCalibration) -> None:
        """Adds a keyframe (replacing any existing calibration of the same frame)."""
        self.keyframes[int(frame)] = calib
        self.keyframes = dict(sorted(self.keyframes.items()))

    def remove(self, frame: int) -> None:
        self.keyframes.pop(int(frame), None)

    def nearest(self, frame: int) -> int | None:
        return min(self.keyframes, key=lambda k: (abs(k - frame), k)) if self.keyframes else None

    def for_image_size(self, width: int, height: int) -> "CalibrationSet":
        return CalibrationSet({f: c.for_image_size(width, height) for f, c in self.keyframes.items()}, self.video)

    def save(self, path: str | Path) -> None:
        data = {
            "version": 2,
            "video": self.video,
            "keyframes": [c.to_dict(frame=f) for f, c in self.keyframes.items()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationSet":
        data = json.loads(Path(path).read_text())
        if "keyframes" not in data:  # older single-calibration file
            return cls({int(data.get("frame", 0)): PitchCalibration.from_dict(data)}, data.get("video"))
        return cls({int(k["frame"]): PitchCalibration.from_dict(k) for k in data["keyframes"]}, data.get("video"))
