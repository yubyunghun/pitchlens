"""Calibrate a frame automatically from detected pitch landmarks.

A keypoint model (trained with training/keypoints.py) predicts, for each of the
33 named pitch landmarks (calibration/pitch.py), where it is in the image and how
sure it is. This module turns those predictions into a PitchCalibration:

  1. keep landmarks above `min_confidence`;
  2. fit the image -> pitch homography with RANSAC, so a few wrong landmarks
     are ignored instead of bending the whole calibration;
  3. accept the fit only if enough landmarks agree, their error is small, and
     they are spread over enough of the pitch (points nearly on one line give
     a wildly wrong homography).

Accepted frames become calibration keyframes automatically, so manual clicking
is only needed when the model isn't sure.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.pitch import Pitch

KEYPOINT_NAMES: list[str] = list(Pitch().keypoints())   # model keypoint i <-> landmark name i


@dataclass
class AutoCalibration:
    calib: PitchCalibration | None     # None when the fit was rejected
    used: int                          # landmarks above the confidence threshold
    inliers: int                       # landmarks consistent with the fit
    mean_error_m: float                # mean error of the inliers (meters)
    spread_m2: float                   # area covered by the inlier landmarks on the pitch
    reason: str = ""                   # why it was rejected

    @property
    def ok(self) -> bool:
        return self.calib is not None


def calibration_from_keypoints(keypoints: np.ndarray, image_size: tuple[int, int], pitch: Pitch = Pitch(),
                               min_confidence: float = 0.5, min_inliers: int = 5, max_error_m: float = 1.0,
                               min_spread_m2: float = 150.0, ransac_threshold_m: float = 1.5) -> AutoCalibration:
    """keypoints: (33, 3) array of (x, y, confidence) in image pixels, in KEYPOINT_NAMES order."""
    kp = np.asarray(keypoints, dtype=float).reshape(-1, 3)
    landmarks = pitch.keypoints()
    names = [n for n in KEYPOINT_NAMES][:len(kp)]
    sure = [i for i in range(len(names)) if kp[i, 2] >= min_confidence and np.isfinite(kp[i, :2]).all()]
    if len(sure) < max(4, min_inliers):
        return AutoCalibration(None, len(sure), 0, float("inf"), 0.0, f"only {len(sure)} confident landmarks")

    img = kp[sure, :2]
    pit = np.array([landmarks[names[i]] for i in sure], dtype=float)
    H, mask = cv2.findHomography(img, pit, cv2.RANSAC, ransac_threshold_m)
    if H is None:
        return AutoCalibration(None, len(sure), 0, float("inf"), 0.0, "no homography fits")
    inl = mask.ravel().astype(bool)
    n_in = int(inl.sum())
    if n_in < min_inliers:
        return AutoCalibration(None, len(sure), n_in, float("inf"), 0.0, f"only {n_in} landmarks agree")

    # refit on the inliers only (least squares) for the final calibration
    sel = [sure[k] for k in np.flatnonzero(inl)]
    calib = PitchCalibration.from_correspondences(kp[sel, :2], pit[inl], image_size, pitch,
                                                  names=[names[i] for i in sel])
    err = float(calib.point_errors().mean())
    hull = cv2.convexHull(pit[inl].astype(np.float32))
    spread = float(cv2.contourArea(hull))
    if spread < min_spread_m2:
        return AutoCalibration(None, len(sure), n_in, err, spread,
                               f"landmarks cover only {spread:.0f} m2 (nearly on one line)")
    if err > max_error_m:
        return AutoCalibration(None, len(sure), n_in, err, spread, f"mean error {err:.2f} m")
    return AutoCalibration(calib, len(sure), n_in, err, spread)
