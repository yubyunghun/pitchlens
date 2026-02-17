"""Camera image <-> pitch (meters) mapping via a planar homography.

The pitch is flat, so a 3x3 homography H maps any image pixel on the grass to
its position on the pitch. 4+ matching landmarks (image pixel <-> known pitch
position) are enough to solve for H.

Valid while the camera does not move. A panning broadcast camera needs a new H
per frame; that comes in a later step (camera-motion tracking / auto keypoints).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pitchlens.calibration.pitch import Pitch


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Maps (N, 2) points through H. Points that land behind the camera become NaN."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = homog[:, 2:3]
    out = homog[:, :2] / np.where(np.abs(w) < 1e-12, np.nan, w)
    out[w[:, 0] <= 0] = np.nan
    return out


class PitchCalibration:
    def __init__(self, H: np.ndarray, image_size: tuple[int, int], pitch: Pitch = Pitch(),
                 correspondences: list[dict] | None = None):
        self.H = np.asarray(H, dtype=float)          # image -> pitch
        self.H_inv = np.linalg.inv(self.H)           # pitch -> image
        self.image_size = tuple(int(v) for v in image_size)  # (width, height)
        self.pitch = pitch
        self.correspondences = correspondences or []  # [{"name", "image": [x, y], "pitch": [x, y]}]

    @classmethod
    def from_correspondences(cls, image_points, pitch_points, image_size, pitch: Pitch = Pitch(),
                             names: list[str] | None = None) -> "PitchCalibration":
        img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        pit = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
        if len(img) < 4 or len(img) != len(pit):
            raise ValueError("Need at least 4 matching image/pitch points")
        H, _ = cv2.findHomography(img, pit, method=0)
        if H is None:
            raise ValueError("Could not solve homography; are 3 of the points on one line?")
        names = names or [f"p{i}" for i in range(len(img))]
        corr = [{"name": n, "image": i.tolist(), "pitch": p.tolist()} for n, i, p in zip(names, img, pit)]
        return cls(H, image_size, pitch, corr)

    def to_pitch(self, image_points) -> np.ndarray:
        return apply_homography(self.H, image_points)

    def to_image(self, pitch_points) -> np.ndarray:
        return apply_homography(self.H_inv, pitch_points)

    def point_errors(self) -> np.ndarray:
        """Distance in meters between each clicked landmark and where H puts it."""
        if not self.correspondences:
            return np.zeros(0)
        img = np.array([c["image"] for c in self.correspondences])
        pit = np.array([c["pitch"] for c in self.correspondences])
        return np.linalg.norm(self.to_pitch(img) - pit, axis=1)

    def for_image_size(self, width: int, height: int) -> "PitchCalibration":
        """Same calibration for a video of a different resolution (e.g. 720p vs 1080p)."""
        if (width, height) == self.image_size:
            return self
        sx, sy = self.image_size[0] / width, self.image_size[1] / height
        S = np.diag([sx, sy, 1.0])  # new pixels -> original pixels
        corr = [{**c, "image": [c["image"][0] / sx, c["image"][1] / sy]} for c in self.correspondences]
        return PitchCalibration(self.H @ S, (width, height), self.pitch, corr)

    def to_dict(self, **meta) -> dict:
        return {
            "homography": self.H.tolist(),
            "image_size": list(self.image_size),
            "pitch": {"length": self.pitch.length, "width": self.pitch.width},
            "correspondences": self.correspondences,
            "mean_error_m": float(self.point_errors().mean()) if self.correspondences else None,
            **meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PitchCalibration":
        return cls(np.array(data["homography"]), tuple(data["image_size"]),
                   Pitch(**data.get("pitch", {})), data.get("correspondences", []))

    def save(self, path: str | Path, **meta) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(**meta), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PitchCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))
