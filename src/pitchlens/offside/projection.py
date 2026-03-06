"""Project an offside line (a straight line across the pitch) into the camera view.

The line runs from touchline to touchline at a fixed pitch x. The pitch is flat,
so it stays a straight line in the image: projecting its ends through the
calibration is enough. It is then clipped to the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration


def _touchline_ends(calib: PitchCalibration, x: float, samples: int = 64) -> tuple[np.ndarray, np.ndarray] | None:
    """Image points of the line at pitch x, from the top to the bottom touchline (unclipped).

    Sampled along the line so that a part behind the camera (no image position) is skipped.
    """
    ys = np.linspace(0.0, calib.pitch.width, samples)
    img = calib.to_image(np.c_[np.full(samples, x), ys])
    ok = np.isfinite(img).all(axis=1) & (np.abs(img) < 1e6).all(axis=1)
    if ok.sum() < 2:
        return None
    pts = img[ok]
    return pts[0], pts[-1]


def project_pitch_line(calib: PitchCalibration, x: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Visible segment (two image points) of the cross-pitch line at pitch x, or None if not in view."""
    ends = _touchline_ends(calib, x)
    if ends is None:
        return None
    w, h = calib.image_size
    a, b = (tuple(int(round(v)) for v in p) for p in ends)
    visible, p1, p2 = cv2.clipLine((0, 0, w, h), a, b)
    if not visible or p1 == p2:
        return None
    return np.array(p1, float), np.array(p2, float)


def project_band(calib: PitchCalibration, x: float, half_width_m: float) -> np.ndarray | None:
    """Image polygon (4, 2) of a band on the pitch from x - half_width to x + half_width.

    Not clipped: fill it with cv2.fillPoly, which clips to the image itself.
    """
    near = _touchline_ends(calib, x - half_width_m)
    far = _touchline_ends(calib, x + half_width_m)
    if near is None or far is None:
        return None
    return np.array([near[0], near[1], far[1], far[0]], float)
