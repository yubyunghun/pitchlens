"""YOLO-based detection of players, goalkeepers, referees and the ball."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO

from pitchlens.detection.classes import BALL, map_class_names
from pitchlens.detection.tiling import tiled_detections


@dataclass
class FrameDetections:
    players: sv.Detections  # players + goalkeepers + referees
    ball: sv.Detections


class Detector:
    def __init__(self, model_path: str = "yolov8x.pt", confidence: float = 0.3,
                 ball_confidence: float = 0.15, device: str | None = None,
                 ball_tiles: bool = False, tile_size: int = 640, tile_overlap: float = 0.2):
        """ball_tiles: also look for the ball on overlapping full-resolution tiles (slower, finds small balls)."""
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.ball_confidence = ball_confidence
        self.device = device
        self.ball_tiles = ball_tiles
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        # model class id -> PitchLens class id (unknown classes are dropped)
        self._class_lut = map_class_names(self.model.names)
        if not self._class_lut:
            raise ValueError(f"Model {model_path} has no football-relevant classes: "
                             f"{list(self.model.names.values())}")
        self._ball_ids = [i for i, c in self._class_lut.items() if c == BALL]

    def _run(self, images, conf: float, classes: list[int] | None = None,
             imgsz: int | None = None) -> list[sv.Detections]:
        """One model call on one image or a batch; detections in model class ids."""
        kwargs = dict(conf=conf, device=self.device, verbose=False)
        if classes is not None:
            kwargs["classes"] = classes
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        return [sv.Detections.from_ultralytics(r) for r in self.model(images, **kwargs)]

    def _to_pitchlens(self, dets: sv.Detections) -> sv.Detections:
        keep = np.array([c in self._class_lut for c in dets.class_id], dtype=bool)
        dets = dets[keep]
        dets.class_id = np.array([self._class_lut[c] for c in dets.class_id], dtype=int)
        return dets

    def detect(self, frame: np.ndarray) -> FrameDetections:
        # Run at the lower ball threshold; players are filtered afterwards.
        dets = self._to_pitchlens(self._run(frame, min(self.confidence, self.ball_confidence))[0])
        ball = dets[dets.class_id == BALL]

        if self.ball_tiles and self._ball_ids:
            def predict(crops):
                return [self._to_pitchlens(d) for d in
                        self._run(crops, self.ball_confidence, self._ball_ids, self.tile_size)]
            tiled = tiled_detections(frame, predict, self.tile_size, self.tile_overlap)
            ball = sv.Detections.merge([b for b in (ball, tiled) if len(b)]) if len(tiled) else ball

        if len(ball) > 1:  # keep only the most confident ball
            ball = ball[[int(np.argmax(ball.confidence))]]

        people = dets[(dets.class_id != BALL) & (dets.confidence >= self.confidence)]
        people = people.with_nms(threshold=0.5, class_agnostic=True)
        return FrameDetections(players=people, ball=ball)
