"""YOLO-based detection of players, goalkeepers, referees and the ball."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO

from pitchlens.detection.classes import BALL, GOALKEEPER, PLAYER, REFEREE

# How model class names map onto PitchLens classes. Covers COCO models
# ("person", "sports ball") and football-finetuned models.
_NAME_MAP = {
    "ball": BALL,
    "sports ball": BALL,
    "football": BALL,
    "goalkeeper": GOALKEEPER,
    "player": PLAYER,
    "person": PLAYER,
    "referee": REFEREE,
}


@dataclass
class FrameDetections:
    players: sv.Detections  # players + goalkeepers + referees
    ball: sv.Detections


class Detector:
    def __init__(self, model_path: str = "yolov8x.pt", confidence: float = 0.3,
                 ball_confidence: float = 0.15, device: str | None = None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.ball_confidence = ball_confidence
        self.device = device
        # model class id -> PitchLens class id (unknown classes are dropped)
        self._class_lut = {
            idx: _NAME_MAP[name.lower()]
            for idx, name in self.model.names.items()
            if name.lower() in _NAME_MAP
        }
        if not self._class_lut:
            raise ValueError(f"Model {model_path} has no football-relevant classes: "
                             f"{list(self.model.names.values())}")

    def detect(self, frame: np.ndarray) -> FrameDetections:
        # Run at the lower ball threshold; players are filtered afterwards.
        result = self.model(frame, conf=min(self.confidence, self.ball_confidence),
                            device=self.device, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(result)

        keep = np.array([c in self._class_lut for c in dets.class_id], dtype=bool)
        dets = dets[keep]
        dets.class_id = np.array([self._class_lut[c] for c in dets.class_id], dtype=int)

        ball = dets[dets.class_id == BALL]
        if len(ball) > 1:  # keep only the most confident ball
            ball = ball[[int(np.argmax(ball.confidence))]]

        people = dets[(dets.class_id != BALL) & (dets.confidence >= self.confidence)]
        people = people.with_nms(threshold=0.5, class_agnostic=True)
        return FrameDetections(players=people, ball=ball)
