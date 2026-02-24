"""Detect camera cuts (switches between broadcast cameras or to replays).

A calibration chain must never cross a cut: the new shot is a different camera.

Two signals per frame:
  * appearance change: distance between the color histograms of this frame and
    the previous one (0 = identical colors, 1 = nothing in common). A cut from
    the wide camera to a close-up changes the colors a lot.
  * camera motion: if a reliable motion was measured, the footage is continuous.

A huge appearance change is always a cut. A moderate change is a cut only when
camera motion also failed. That catches cuts between two similar-looking wide
cameras, without treating a single blurred frame as a cut.

Typical values on synthetic broadcast frames: continuous pan ~0.02, heavy
motion blur ~0.15, cut between two wide shots ~0.55, cut to a close-up ~0.97.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

SMALL = (160, 90)


def color_histogram(frame: np.ndarray) -> np.ndarray:
    """Normalized hue-saturation histogram of a downscaled frame."""
    hsv = cv2.cvtColor(cv2.resize(frame, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).astype(np.float32)


class AppearanceChange:
    """Feeds consecutive frames; returns how much the picture changed (0..1)."""

    def __init__(self):
        self._prev: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> float:
        hist = color_histogram(frame)
        change = 0.0
        if self._prev is not None:
            change = float(cv2.compareHist(self._prev, hist, cv2.HISTCMP_BHATTACHARYYA))
        self._prev = hist
        return change


def find_cuts(changes: Sequence[float], motion_ok: Sequence[bool],
              certain: float = 0.5, possible: float = 0.25) -> list[int]:
    """Frames where a new shot starts.

    changes[t]:   appearance change from frame t-1 to t
    motion_ok[t]: whether camera motion from t-1 to t was measured
    certain:      change above which it's a cut no matter what
    possible:     change above which it's a cut if motion also failed
    """
    cuts = []
    for t in range(1, len(changes)):
        if changes[t] >= certain or (changes[t] >= possible and not motion_ok[t]):
            cuts.append(t)
    return cuts
