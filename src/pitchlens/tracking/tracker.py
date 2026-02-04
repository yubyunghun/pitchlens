"""ByteTrack-style multi-object tracker so each player keeps a stable ID.

Per frame:
  1. Predict every track's box forward using its smoothed velocity.
  2. Match tracks to high-confidence detections (Hungarian on IoU).
  3. Match leftover tracks to low-confidence detections. These are often
     partly occluded players, and this second pass is the core ByteTrack idea.
  4. Start new tracks from unmatched high-confidence detections.
  5. Drop tracks that have been missing for longer than `lost_buffer` frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _match(tracks: np.ndarray, dets: np.ndarray, min_iou: float):
    """Returns (matches [(track_i, det_j)], unmatched_tracks, unmatched_dets)."""
    iou = iou_matrix(tracks, dets)
    if iou.size == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))
    rows, cols = linear_sum_assignment(-iou)
    matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    mt, md = {r for r, _ in matches}, {c for _, c in matches}
    return (matches, [i for i in range(len(tracks)) if i not in mt],
            [j for j in range(len(dets)) if j not in md])


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    missed: int = 0

    def predict(self) -> None:
        self.box = self.box + self.velocity

    def update(self, box: np.ndarray) -> None:
        # box has already been advanced by `missed + 1` predictions; spread the error over them
        step = (box - (self.box - self.velocity * (self.missed + 1))) / (self.missed + 1)
        self.velocity = 0.6 * self.velocity + 0.4 * step
        self.box = box
        self.hits += 1
        self.missed = 0


class PlayerTracker:
    def __init__(self, fps: float = 25.0, high_conf: float = 0.5, new_track_conf: float = 0.5,
                 min_iou_high: float = 0.2, min_iou_low: float = 0.4, min_hits: int = 2,
                 lost_seconds: float = 2.0):
        self.high_conf = high_conf
        self.new_track_conf = new_track_conf
        self.min_iou_high = min_iou_high
        self.min_iou_low = min_iou_low
        self.min_hits = min_hits
        self.lost_buffer = int(fps * lost_seconds)  # keep IDs alive through short occlusions
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Returns the tracked subset of `detections` with `tracker_id` filled in."""
        for t in self._tracks:
            t.predict()

        conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        high = np.where(conf >= self.high_conf)[0]
        low = np.where(conf < self.high_conf)[0]
        assigned: dict[int, int] = {}  # detection index -> track id

        track_boxes = np.array([t.box for t in self._tracks]).reshape(-1, 4)
        m1, left_tracks, left_high = _match(track_boxes, detections.xyxy[high], self.min_iou_high)
        for ti, dj in m1:
            self._tracks[ti].update(detections.xyxy[high[dj]])
            assigned[high[dj]] = self._tracks[ti].track_id

        m2, still_left, _ = _match(track_boxes[left_tracks], detections.xyxy[low], self.min_iou_low)
        for ti, dj in m2:
            t = self._tracks[left_tracks[ti]]
            t.update(detections.xyxy[low[dj]])
            assigned[low[dj]] = t.track_id

        for ti in still_left:
            self._tracks[left_tracks[ti]].missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.lost_buffer]

        for dj in left_high:
            idx = high[dj]
            if conf[idx] >= self.new_track_conf:
                self._tracks.append(_Track(self._next_id, detections.xyxy[idx].astype(float)))
                assigned[idx] = self._next_id
                self._next_id += 1

        hits = {t.track_id: t.hits for t in self._tracks}
        keep = sorted(i for i, tid in assigned.items() if hits.get(tid, 0) >= self.min_hits)
        out = detections[np.array(keep, dtype=int)]
        out.tracker_id = np.array([assigned[i] for i in keep], dtype=int)
        return out
