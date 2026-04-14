"""Run a pitch-landmark (YOLO pose) model and turn its output into keyframes.

The model (trained with `pitchlens-train --task pose`) predicts one object,
"pitch", with 33 keypoints in KEYPOINT_NAMES order, each (x, y, confidence).
"""

from __future__ import annotations

import numpy as np

from pitchlens.calibration.auto_calib import KEYPOINT_NAMES, calibration_from_keypoints
from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.pitch import Pitch


class KeypointDetector:
    def __init__(self, model_path: str, device: str | None = None, min_box_confidence: float = 0.25):
        from ultralytics import YOLO  # imported here: loads PyTorch

        self.model = YOLO(model_path)
        self.device = device
        self.min_box_confidence = min_box_confidence

    def _run(self, frames: list[np.ndarray]) -> list[np.ndarray | None]:
        """(33, 3) keypoints of the most confident 'pitch' per frame, or None."""
        def to_numpy(x):
            return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=float)

        out = []
        for r in self.model(frames, conf=self.min_box_confidence, device=self.device, verbose=False):
            kps = getattr(r, "keypoints", None)
            data = None if kps is None or kps.data is None else to_numpy(kps.data)
            if data is None or data.size == 0:
                out.append(None)
                continue
            best = int(np.argmax(to_numpy(r.boxes.conf)))
            out.append(data[best].reshape(-1, 3)[:len(KEYPOINT_NAMES)])
        return out

    def predict(self, frame: np.ndarray) -> np.ndarray | None:
        return self._run([frame])[0]


def keyframes_from_predictions(predictions: dict[int, np.ndarray], image_size: tuple[int, int],
                               pitch: Pitch = Pitch(), **quality) -> tuple[CalibrationSet, dict[int, str]]:
    """(automatic keyframes, frame -> why rejected). `quality` goes to calibration_from_keypoints."""
    accepted, rejected = {}, {}
    for frame, kp in sorted(predictions.items()):
        if kp is None:
            rejected[frame] = "no pitch found"
            continue
        result = calibration_from_keypoints(kp, image_size, pitch, **quality)
        if result.ok:
            accepted[frame] = result.calib
        else:
            rejected[frame] = result.reason
    return CalibrationSet(accepted), rejected
