"""Split players into two teams by jersey color.

For each player box we crop the torso, mask out grass-green pixels, and take
the mean remaining color (in Lab space, where distances match perception).
KMeans with k=2 over many player samples gives the two team colors. Each
track ID then gets a majority vote over time so labels don't flicker.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from sklearn.cluster import KMeans

TEAM_A, TEAM_B, UNKNOWN = 0, 1, -1


def jersey_color(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Mean Lab color of the torso region, ignoring grass. None if no usable pixels."""
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 5:
        return None
    # Torso: upper-middle part of the box, trimmed at the sides.
    tx1, tx2 = x1 + w // 4, x2 - w // 4
    ty1, ty2 = y1 + h // 6, y1 + h // 2
    crop = frame[max(ty1, 0):max(ty2, 0), max(tx1, 0):max(tx2, 0)]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)) > 0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass.reshape(-1)]
    if len(lab) < 10:
        return None
    return lab.mean(axis=0).astype(np.float32)


class TeamAssigner:
    def __init__(self, vote_window: int = 50):
        self._kmeans: KMeans | None = None
        self._samples: list[np.ndarray] = []
        self._votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))
        self.team_colors_bgr: dict[int, tuple[int, int, int]] = {}

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    def collect(self, frame: np.ndarray, boxes: np.ndarray) -> None:
        """Gather jersey color samples before fitting."""
        for box in boxes:
            c = jersey_color(frame, box)
            if c is not None:
                self._samples.append(c)

    def fit(self) -> None:
        if len(self._samples) < 10:
            raise RuntimeError("Not enough player samples to fit team colors")
        self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.stack(self._samples))
        for team, center in enumerate(self._kmeans.cluster_centers_):
            lab = np.clip(center, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
            self.team_colors_bgr[team] = (int(b), int(g), int(r))
        self._samples.clear()

    def assign(self, frame: np.ndarray, boxes: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
        """Team id per box (TEAM_A / TEAM_B / UNKNOWN), smoothed per track."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before assign()")
        teams = np.full(len(boxes), UNKNOWN, dtype=int)
        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
            c = jersey_color(frame, box)
            if c is not None:
                self._votes[int(tid)].append(int(self._kmeans.predict(c[None])[0]))
            votes = self._votes.get(int(tid))
            if votes:
                teams[i] = Counter(votes).most_common(1)[0][0]
        return teams
