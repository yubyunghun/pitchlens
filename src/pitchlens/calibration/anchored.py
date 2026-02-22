"""Solve a calibration for every frame of a clip from a few keyframes.

Inputs: the camera motion between every pair of consecutive frames, and a set of
calibrated keyframes. Each frame is calibrated from its *nearest* keyframe,
chaining motions forward (after the keyframe) or backward (before it):

    forward:  H_t = H_(t-1) @ M_t^-1
    backward: H_t = H_(t+1) @ M_(t+1)

where M_t maps frame t-1 pixels to frame t pixels. Drift grows with the number
of chained motions, so the nearest keyframe gives the most accurate result.

The chain is cut where the footage is discontinuous:
  * an explicit break (a detected camera cut),
  * an implausible motion (see is_plausible_motion),
  * more than `max_gap` failed motion estimates in a row.
Shorter runs of failures are bridged with the last good motion. Frames with no
keyframe in their continuous stretch of footage get no calibration (None).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from pitchlens.calibration.calibration_tracker import is_plausible_motion
from pitchlens.calibration.camera_motion import MotionResult
from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.keyframes import CalibrationSet


@dataclass
class FrameCalibration:
    calib: PitchCalibration
    keyframe: int    # which keyframe it was chained from
    distance: int    # how many motions were chained (0 = the keyframe itself)


def _clean_motions(motions: Sequence[MotionResult | np.ndarray | None], image_size, breaks: set[int],
                   max_gap: int) -> list[np.ndarray | None]:
    """Per frame t: usable motion from t-1 to t, or None where the chain is broken."""
    n = len(motions)
    raw: list[np.ndarray | None] = [None] * n
    hard_break = [False] * n
    for t in range(1, n):
        m = motions[t]
        H = m.H if isinstance(m, MotionResult) else m
        if t in breaks:
            hard_break[t] = True
        elif H is not None:
            if is_plausible_motion(H, image_size):
                raw[t] = np.asarray(H, dtype=float)
            else:
                hard_break[t] = True

    out = list(raw)
    t = 1
    while t < n:
        if raw[t] is not None or hard_break[t]:
            t += 1
            continue
        end = t
        while end < n and raw[end] is None and not hard_break[end]:
            end += 1
        gap = range(t, end)                                   # failed estimates
        before = raw[t - 1] if t > 1 else None                # None if the gap follows a break
        after = raw[end] if end < n else None
        fill = before if before is not None else after
        if len(gap) <= max_gap and fill is not None:
            for i in gap:
                out[i] = fill                                 # camera kept moving the same way
        t = end
    return out


def solve_calibrations(motions: Sequence[MotionResult | np.ndarray | None], keyframes: CalibrationSet,
                       image_size: tuple[int, int], breaks: Iterable[int] = (),
                       max_gap: int = 10) -> list[FrameCalibration | None]:
    """motions[t] = camera motion from frame t-1 to t (motions[0] is ignored)."""
    n = len(motions)
    M = _clean_motions(motions, tuple(image_size), set(breaks), max_gap)
    result: list[FrameCalibration | None] = [None] * n
    key_frames = [k for k in keyframes.frames if 0 <= k < n]

    def offer(t: int, H: np.ndarray, base: PitchCalibration, k: int, d: int) -> None:
        cur = result[t]
        if cur is None or d < cur.distance:  # nearest keyframe wins; ties go to the earlier one
            result[t] = FrameCalibration(PitchCalibration(H / H[2, 2], image_size, base.pitch), k, d)

    keys = set(key_frames)
    for k in key_frames:
        base = keyframes.keyframes[k]
        offer(k, base.H.copy(), base, k, 0)
        # forward until the chain breaks or the next keyframe takes over
        H, t = base.H.copy(), k + 1
        while t < n and M[t] is not None and t not in keys:
            H = H @ np.linalg.inv(M[t])
            offer(t, H, base, k, t - k)
            t += 1
        # backward
        H, t = base.H.copy(), k - 1
        while t >= 0 and M[t + 1] is not None and t not in keys:
            H = H @ M[t + 1]
            offer(t, H, base, k, k - t)
            t -= 1
    return result
