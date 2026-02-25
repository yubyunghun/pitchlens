"""Check a calibration against the video itself: do the projected lines land on white?

No ground truth needed. Points are sampled along every pitch marking, projected
into the frame with the calibration, and checked against the white line pixels
actually visible on the grass. A good calibration puts most samples on white;
drift moves them onto plain grass.

Players standing on lines lower the score a little, so treat it as relative:
compare frames of the same clip, and look for sudden drops.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.pitch_mask import grass_mask


def _densify(line: np.ndarray, step_m: float) -> np.ndarray:
    pts = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        pts.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(pts)


def white_line_mask(frame: np.ndarray, tolerance_px: int = 3) -> np.ndarray:
    """White-ish pixels on the pitch, widened by `tolerance_px` (uint8, 255 = line)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
    pitch = cv2.dilate(grass_mask(frame), np.ones((15, 15), np.uint8))  # lines at the grass edge count
    white &= pitch
    k = 2 * tolerance_px + 1
    return cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def line_alignment(frame: np.ndarray, calib: PitchCalibration, step_m: float = 0.5,
                   min_samples: int = 30, tolerance_px: int = 3) -> float | None:
    """Fraction (0..1) of visible projected line samples that hit a white line pixel.

    None if too little of the pitch's markings is in view to judge.
    """
    h, w = frame.shape[:2]
    samples = np.vstack([_densify(line, step_m) for line in calib.pitch.lines()])
    img = calib.to_image(samples)
    ok = np.isfinite(img).all(axis=1)
    img = img[ok]
    xy = np.round(img).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy = xy[inside]
    if len(xy) == 0:
        return None
    on_grass = cv2.dilate(grass_mask(frame), np.ones((15, 15), np.uint8))[xy[:, 1], xy[:, 0]] > 0
    xy = xy[on_grass]  # only judge where the pitch is actually visible
    if len(xy) < min_samples:
        return None
    hits = white_line_mask(frame, tolerance_px)[xy[:, 1], xy[:, 0]] > 0
    return float(hits.mean())
