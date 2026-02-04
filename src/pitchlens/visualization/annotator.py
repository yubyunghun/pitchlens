"""Broadcast-style overlays: ellipses at players' feet, IDs, ball marker."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from pitchlens.detection.classes import GOALKEEPER, REFEREE
from pitchlens.tracking.team_assigner import UNKNOWN

REFEREE_COLOR = (0, 215, 255)   # yellow (BGR)
UNKNOWN_COLOR = (200, 200, 200)
BALL_COLOR = (255, 255, 255)


def _draw_player(frame, box, color, label):
    x1, y1, x2, y2 = box.astype(int)
    cx, w = (x1 + x2) // 2, x2 - x1
    cv2.ellipse(frame, (cx, y2), (max(w // 2, 8), max(w // 6, 3)), 0, -45, 235, color, 2, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1, by1 = cx - tw // 2 - 4, y2 + 6
        cv2.rectangle(frame, (bx1, by1), (bx1 + tw + 8, by1 + th + 6), color, -1)
        cv2.putText(frame, label, (bx1 + 4, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, players: sv.Detections, teams: np.ndarray,
             team_colors: dict[int, tuple[int, int, int]], ball: sv.Detections) -> np.ndarray:
    out = frame.copy()
    for box, cls, tid, team in zip(players.xyxy, players.class_id, players.tracker_id, teams):
        if cls == REFEREE:
            color, label = REFEREE_COLOR, "REF"
        else:
            color = team_colors.get(int(team), UNKNOWN_COLOR) if team != UNKNOWN else UNKNOWN_COLOR
            label = f"{'GK ' if cls == GOALKEEPER else ''}{int(tid)}"
        _draw_player(out, box, color, label)

    for box in ball.xyxy:
        x1, y1, x2, _ = box.astype(int)
        cx = (x1 + x2) // 2
        tri = np.array([[cx, y1 - 4], [cx - 8, y1 - 18], [cx + 8, y1 - 18]])
        cv2.fillPoly(out, [tri], BALL_COLOR)
        cv2.polylines(out, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
    return out
