"""Clean up per-frame ball positions into a usable ball path.

Raw ball detections are messy: the ball is small and fast, so the detector
misses it in many frames, sometimes mistakes a shoe or a head for it, and its
position jitters by a few pixels. Before passes can be detected:

  1. remove impossible detections: a detection that would need the ball to
     travel faster than `max_speed` to reach the detections before AND after it
     (the fastest shots are around 35 m/s);
  2. fill short gaps (up to `max_gap_s`) by straight-line interpolation;
  3. smooth lightly (a short moving average) so velocity is not dominated by
     jitter, while kicks stay sharp.

Positions are on the pitch in meters. A ball in the air is projected to where
it appears on the grass, so lofted balls look faster/further than they are;
ground passes are measured well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BallPath:
    positions: np.ndarray     # (n, 2) pitch meters, NaN where unknown
    observed: np.ndarray      # (n,) True where a (kept) detection was used
    fps: float

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def known(self) -> np.ndarray:
        return np.isfinite(self.positions).all(axis=1)

    def velocity(self) -> np.ndarray:
        """(n, 2) meters per second, NaN where unknown or on isolated single frames."""
        vel = np.full_like(self.positions, np.nan)
        for start, end in _segments(self.known):
            if end - start >= 2:
                vel[start:end] = np.gradient(self.positions[start:end], axis=0) * self.fps
        return vel

    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity(), axis=1)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) ranges where mask is True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _remove_outliers(pos: np.ndarray, fps: float, max_speed: float) -> np.ndarray:
    """Mask of detections to keep."""
    idx = np.flatnonzero(np.isfinite(pos).all(axis=1))
    keep = np.isfinite(pos).all(axis=1)
    for k, t in enumerate(idx):
        speeds = []
        for n in (idx[k - 1] if k > 0 else None, idx[k + 1] if k + 1 < len(idx) else None):
            if n is not None:
                speeds.append(np.linalg.norm(pos[t] - pos[n]) / (abs(t - n) / fps))
        if speeds and min(speeds) > max_speed:    # can't be reached from either neighbor
            keep[t] = False
    return keep


def _fill_gaps(pos: np.ndarray, max_gap: int) -> np.ndarray:
    out = pos.copy()
    idx = np.flatnonzero(np.isfinite(pos).all(axis=1))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = pos[a] + t[:, None] * (pos[b] - pos[a])
    return out


def _smooth(pos: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pos
    out = pos.copy()
    half = window // 2
    for start, end in _segments(np.isfinite(pos).all(axis=1)):
        for t in range(start + half, end - half):          # segment ends keep their raw value
            out[t] = pos[t - half:t + half + 1].mean(axis=0)
    return out


def build_ball_path(raw: Sequence[np.ndarray | None], fps: float, max_speed: float = 40.0,
                    max_gap_s: float = 0.5, smooth_window: int = 3) -> BallPath:
    """raw[t]: ball pitch position (2,) at frame t, or None / NaN if not detected or not calibrated."""
    pos = np.full((len(raw), 2), np.nan)
    for t, p in enumerate(raw):
        if p is not None:
            pos[t] = np.asarray(p, dtype=float).reshape(2)
    keep = _remove_outliers(pos, fps, max_speed)
    pos[~keep] = np.nan
    filled = _fill_gaps(pos, int(round(max_gap_s * fps)))
    return BallPath(_smooth(filled, smooth_window), keep, fps)
