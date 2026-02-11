"""Keep the pitch calibration valid while the camera moves.

You calibrate one frame (image -> pitch homography H0). For each following frame,
camera_motion gives M = (previous image -> current image). A pixel in the current
frame maps back to the previous frame with M^-1, and from there to the pitch, so:

    H_current = H_previous @ M^-1

If motion estimation fails for a frame (blur, a player filling the screen), the
camera most likely kept moving the way it was, so the last motion is reused as a
prediction. After too many failures in a row the calibration is marked lost
until it is re-anchored with a fresh calibration.
"""

from __future__ import annotations

import numpy as np

from pitchlens.calibration.camera_motion import MotionResult
from pitchlens.calibration.homography import PitchCalibration


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
