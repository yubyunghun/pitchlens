"""Keep the pitch calibration valid while the camera moves.

You calibrate one frame (image -> pitch homography H0). For each following frame,
camera_motion gives M = (previous image -> current image). A pixel in the current
frame maps back to the previous frame with M^-1, and from there to the pitch, so:

    H_current = H_previous @ M^-1

If motion estimation fails for a frame (blur, a player filling the screen), or
returns a motion no real camera could make (a cut to another camera), the
camera most likely kept moving the way it was, so the last motion is reused as a
prediction. After too many failures in a row the calibration is marked lost
until it is re-anchored with a fresh calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.calibration.camera_motion import MotionResult
from pitchlens.calibration.homography import PitchCalibration, apply_homography


def is_plausible_motion(M: np.ndarray, image_size: tuple[int, int],
                        max_shift: float = 0.3, max_zoom: float = 0.15) -> bool:
    """Could a broadcast camera really move like this between two frames?

    Rejects camera cuts and bad estimates: frame corners jumping more than
    `max_shift` of the frame width, zoom changing more than `max_zoom`, or the
    frame folding over itself.
    """
    w, h = image_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    moved = apply_homography(M, corners)
    if not np.isfinite(moved).all():
        return False
    if np.linalg.norm(moved - corners, axis=1).max() > max_shift * w:
        return False
    area = cv2.contourArea(moved.astype(np.float32))
    if not (1 / (1 + max_zoom) ** 2 <= area / (w * h) <= (1 + max_zoom) ** 2):
        return False
    return bool(cv2.isContourConvex(moved.astype(np.float32).reshape(-1, 1, 2)))


class CalibrationTracker:
    def __init__(self, base: PitchCalibration, max_lost_frames: int = 10):
        self.max_lost_frames = max_lost_frames
        self.reset(base)

    def reset(self, calib: PitchCalibration) -> None:
        """Re-anchor to a fresh calibration of the current frame."""
        self._base = calib
        self._H = calib.H.copy()
        self._last_motion = np.eye(3)
        self.lost_frames = 0

    @property
    def is_lost(self) -> bool:
        return self.lost_frames > self.max_lost_frames

    def update(self, motion: MotionResult | None) -> PitchCalibration | None:
        """Calibration for the current frame, or None once tracking is lost."""
        if self.is_lost:
            return None
        if motion is not None and not is_plausible_motion(motion.H, self._base.image_size):
            motion = None  # e.g. a camera cut; treat like a failed estimate
        if motion is not None:
            M = motion.H
            self.lost_frames = 0
        else:
            M = self._last_motion  # predict: camera keeps moving the same way
            self.lost_frames += 1
            if self.is_lost:
                return None
        self._last_motion = M
        self._H = self._H @ np.linalg.inv(M)
        self._H /= self._H[2, 2]  # keep the scale normalized so values don't drift numerically
        return self.current

    @property
    def current(self) -> PitchCalibration:
        return PitchCalibration(self._H, self._base.image_size, self._base.pitch)
