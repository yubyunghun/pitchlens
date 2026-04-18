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
    class_id: int = 0             # last detected class (player / goalkeeper / referee)
    confidence: float = 1.0       # last detection confidence

    def predict(self) -> None:
        self.box = self.box + self.velocity

    def update(self, box: np.ndarray) -> None:
        # box has already been advanced by `missed + 1` predictions; spread the error over them
        step = (box - (self.box - self.velocity * (self.missed + 1))) / (self.missed + 1)
        # first measured motion is taken as is (the starting velocity of 0 is just a placeholder)
        self.velocity = step if self.hits == 1 else 0.6 * self.velocity + 0.4 * step
        self.box = box
        self.hits += 1
        self.missed = 0


class PlayerTracker:
    def __init__(self, fps: float = 25.0, high_conf: float = 0.5, new_track_conf: float = 0.5,
                 min_iou_high: float = 0.2, min_iou_low: float = 0.4, min_hits: int = 2,
                 lost_seconds: float = 2.0, max_center_shift: float = 1.0):
        self.max_center_shift = max_center_shift   # in box heights, for the distance-based 3rd match
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
        cls = detections.class_id if detections.class_id is not None else np.zeros(len(detections), int)
        high = np.where(conf >= self.high_conf)[0]
        low = np.where(conf < self.high_conf)[0]
        assigned: dict[int, int] = {}  # detection index -> track id

        def take(track: _Track, idx: int) -> None:
            track.update(detections.xyxy[idx])
            track.class_id, track.confidence = int(cls[idx]), float(conf[idx])
            assigned[idx] = track.track_id

        track_boxes = np.array([t.box for t in self._tracks]).reshape(-1, 4)
        m1, left_tracks, left_high = _match(track_boxes, detections.xyxy[high], self.min_iou_high)
        for ti, dj in m1:
            take(self._tracks[ti], high[dj])

        m2, still_left, _ = _match(track_boxes[left_tracks], detections.xyxy[low], self.min_iou_low)
        for ti, dj in m2:
            take(self._tracks[left_tracks[ti]], low[dj])

        # 3rd chance for small, fast boxes (quick pans, skipped frames) that no longer overlap:
        # pair the rest by center distance, at most `max_center_shift` box heights apart
        rest_tracks = [left_tracks[ti] for ti in still_left]
        if rest_tracks and left_high:
            tb = track_boxes[rest_tracks]
            db = detections.xyxy[[high[dj] for dj in left_high]]
            tc = (tb[:, :2] + tb[:, 2:]) / 2
            dc = (db[:, :2] + db[:, 2:]) / 2
            heights = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
            cost = np.linalg.norm(tc[:, None] - dc[None], axis=2) / heights[:, None]
            rows, cols = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_center_shift:
                    take(self._tracks[rest_tracks[r]], high[left_high[c]])
                    matched_t.add(r)
                    matched_d.add(c)
            still_left = [still_left[k] for k in range(len(still_left)) if k not in matched_t]
            left_high = [left_high[k] for k in range(len(left_high)) if k not in matched_d]

        for ti in still_left:
            self._tracks[left_tracks[ti]].missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.lost_buffer]

        for dj in left_high:
            idx = high[dj]
            if conf[idx] >= self.new_track_conf:
                self._tracks.append(_Track(self._next_id, detections.xyxy[idx].astype(float),
                                           class_id=int(cls[idx]), confidence=float(conf[idx])))
                assigned[idx] = self._next_id
                self._next_id += 1

        hits = {t.track_id: t.hits for t in self._tracks}
        keep = sorted(i for i, tid in assigned.items() if hits.get(tid, 0) >= self.min_hits)
        out = detections[np.array(keep, dtype=int)]
        out.tracker_id = np.array([assigned[i] for i in keep], dtype=int)
        return out

    def coast(self, since_detection: int) -> sv.Detections:
        """A frame without detections (detection skipped for speed): move every track by its velocity.

        Reports the confirmed tracks that were matched at the last detected frame, i.e. missed for
        at most `since_detection` frames; players already lost before then stay hidden.
        """
        for t in self._tracks:
            t.predict()
            t.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.lost_buffer]
        shown = [t for t in self._tracks if t.hits >= self.min_hits and t.missed <= since_detection]
        if not shown:
            return sv.Detections.empty()
        return sv.Detections(xyxy=np.array([t.box for t in shown], dtype=float),
                             confidence=np.array([t.confidence for t in shown]),
                             class_id=np.array([t.class_id for t in shown], dtype=int),
                             tracker_id=np.array([t.track_id for t in shown], dtype=int))
