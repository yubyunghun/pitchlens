"""Body keypoints for each tracked player, from a YOLO pose model.

Broadcast players are often only 40-80 px tall: too small for a pose model
looking at the whole frame. So the model runs on a padded crop around each
tracked player (all crops in one batch), and the keypoints are shifted back
to frame coordinates. Each crop belongs to a known player, so no matching of
poses to players is needed; when a crop holds several people (a challenge),
the pose whose box is closest to the crop's center is used.
"""

from __future__ import annotations

import numpy as np


def crop_boxes(boxes: np.ndarray, frame_shape: tuple[int, int], padding: float = 0.25) -> np.ndarray:
    """Padded, frame-clipped integer crops (x1, y1, x2, y2) around each box."""
    h, w = frame_shape[:2]
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    size = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
    pad = np.hstack([size, size]) * padding
    out = b + pad * np.array([-1, -1, 1, 1])
    return np.clip(np.round(out), 0, [w, h, w, h]).astype(int)


def pick_centered(pose_boxes: np.ndarray, crop_size: tuple[int, int]) -> int | None:
    """Index of the pose box whose center is nearest the crop center (the player the crop is for)."""
    if len(pose_boxes) == 0:
        return None
    cw, ch = crop_size
    centers = np.stack([(pose_boxes[:, 0] + pose_boxes[:, 2]) / 2, (pose_boxes[:, 1] + pose_boxes[:, 3]) / 2], 1)
    return int(np.argmin(np.linalg.norm(centers - [cw / 2, ch / 2], axis=1)))


class PoseDetector:
    def __init__(self, model_path: str = "yolov8m-pose.pt", device: str | None = None, padding: float = 0.25,
                 min_box_confidence: float = 0.25):
        from ultralytics import YOLO  # imported here: loads PyTorch

        self.model = YOLO(model_path)
        self.device = device
        self.padding = padding
        self.min_box_confidence = min_box_confidence

    def _run(self, crops: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per crop: (pose boxes (n, 4), keypoints (n, 17, 3)) in crop coordinates."""
        def to_numpy(x):
            return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=float)

        out = []
        for r in self.model(crops, conf=self.min_box_confidence, device=self.device, verbose=False):
            kps = getattr(r, "keypoints", None)
            if kps is None or kps.data is None or r.boxes is None:
                out.append((np.zeros((0, 4)), np.zeros((0, 17, 3))))
                continue
            out.append((to_numpy(r.boxes.xyxy).reshape(-1, 4), to_numpy(kps.data).reshape(-1, 17, 3)))
        return out

    def predict_players(self, frame: np.ndarray, boxes: np.ndarray) -> list[np.ndarray | None]:
        """(17, 3) frame-coordinate keypoints per player box, or None where no pose was found."""
        if len(boxes) == 0:
            return []
        crops_xyxy = crop_boxes(boxes, frame.shape, self.padding)
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in crops_xyxy]
        out: list[np.ndarray | None] = []
        for (x1, y1, x2, y2), (pboxes, pkps) in zip(crops_xyxy, self._run(crops)):
            k = pick_centered(pboxes, (x2 - x1, y2 - y1))
            if k is None:
                out.append(None)
                continue
            kp = pkps[k].copy()
            kp[:, 0] += x1
            kp[:, 1] += y1
            out.append(kp)
        return out
