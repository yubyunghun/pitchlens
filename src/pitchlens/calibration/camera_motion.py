"""Estimate how the camera moved between consecutive frames.

A broadcast camera pans, tilts and zooms from a fixed position, so the motion
between two frames is exactly a homography (3x3) over the whole image.

Per frame pair:
  1. Pick corner-like points on the static pitch (see pitch_mask) in the previous frame.
  2. Track them into the current frame with pyramidal Lucas-Kanade optical flow.
  3. Track them back again and drop points that don't return to where they started
     (forward-backward check; removes tracks that slipped along a line or onto a player).
  4. Fit the homography with RANSAC so remaining outliers are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchlens.calibration.pitch_mask import background_mask


@dataclass
class MotionResult:
    H: np.ndarray        # previous-frame pixels -> current-frame pixels
    n_tracked: int       # points that survived optical flow + forward-backward check
    n_inliers: int       # points consistent with H

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_tracked if self.n_tracked else 0.0


class CameraMotionEstimator:
    def __init__(self, max_corners: int = 1000, quality: float = 0.005, min_distance: int = 8,
                 win_size: int = 21, pyramid_levels: int = 4, fb_threshold: float = 1.0,
                 ransac_threshold: float = 2.0, min_inliers: int = 25, min_inlier_ratio: float = 0.5):
        self.feature_params = dict(maxCorners=max_corners, qualityLevel=quality,
                                   minDistance=min_distance, blockSize=7)
        self.lk_params = dict(winSize=(win_size, win_size), maxLevel=pyramid_levels,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.fb_threshold = fb_threshold
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio

    def estimate(self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_mask: np.ndarray,
                 curr_mask: np.ndarray | None = None, guess: np.ndarray | None = None) -> MotionResult | None:
        """Homography prev -> curr, or None if there isn't enough reliable texture.

        guess: expected motion (e.g. the previous frame's H). Optical flow starts its
        search there, which lets it follow fast pans it would otherwise lose.
        """
        pts = cv2.goodFeaturesToTrack(prev_gray, mask=prev_mask, **self.feature_params)
        if pts is None or len(pts) < self.min_inliers:
            return None

        if guess is not None:
            init = cv2.perspectiveTransform(pts, guess).astype(np.float32)
            fwd, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, init,
                                                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
            init_back = cv2.perspectiveTransform(fwd, np.linalg.inv(guess)).astype(np.float32)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, fwd, init_back,
                                                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
        else:
            fwd, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **self.lk_params)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, fwd, None, **self.lk_params)
        fb_error = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_error < self.fb_threshold)

        if curr_mask is not None:  # the point must also land on static pitch in the new frame
            xy = np.round(fwd.reshape(-1, 2)).astype(int)
            h, w = curr_mask.shape
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            on_pitch = np.zeros(len(xy), bool)
            on_pitch[inside] = curr_mask[xy[inside, 1], xy[inside, 0]] > 0
            good &= on_pitch

        src, dst = pts.reshape(-1, 2)[good], fwd.reshape(-1, 2)[good]
        if len(src) < self.min_inliers:
            return None
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)
        if H is None:
            return None
        result = MotionResult(H, len(src), int(inliers.sum()))
        if result.n_inliers < self.min_inliers or result.inlier_ratio < self.min_inlier_ratio:
            return None
        return result


class CameraMotionTracker:
    """Feeds consecutive frames to the estimator; remembers the previous frame."""

    def __init__(self, estimator: CameraMotionEstimator | None = None):
        self.estimator = estimator or CameraMotionEstimator()
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._last_H: np.ndarray | None = None  # last successful motion, used as the next guess

    def update(self, frame: np.ndarray, exclude_boxes: np.ndarray | None = None) -> MotionResult | None:
        """Motion from the previous frame to this one (None on the first frame or on failure)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = background_mask(frame, exclude_boxes)
        result = None
        if self._prev_gray is not None:
            result = None
            if self._last_H is not None:  # camera usually keeps moving the way it was
                result = self.estimator.estimate(self._prev_gray, gray, self._prev_mask, mask, self._last_H)
            if result is None:
                result = self.estimator.estimate(self._prev_gray, gray, self._prev_mask, mask)
            self._last_H = result.H if result is not None else None
        self._prev_gray, self._prev_mask = gray, mask
        return result
