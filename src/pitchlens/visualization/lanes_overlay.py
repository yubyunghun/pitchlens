"""Draw passing lanes: green open, amber risky, red blocked; the best option thicker."""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.tactics.lane_options import FrameOptions
from pitchlens.tactics.passing_lanes import BLOCKED, OPEN, RISKY
from pitchlens.visualization.minimap import Minimap

LANE_COLORS = {OPEN: (80, 200, 80), RISKY: (0, 200, 255), BLOCKED: (60, 60, 230)}   # BGR


def draw_lanes_minimap(img: np.ndarray, minimap: Minimap, positions: np.ndarray, options: FrameOptions,
                       show_blocked: bool = True) -> np.ndarray:
    """Lanes from the ball to each teammate on a rendered minimap (in place)."""
    start = tuple(int(v) for v in minimap.to_px(options.start))
    base = max(1, int(round(minimap.scale * 0.25)))
    for lane in sorted(options.lanes, key=lambda ln: ln is options.best):   # best drawn last, on top
        if lane.status == BLOCKED and not show_blocked:
            continue
        end = tuple(int(v) for v in minimap.to_px(positions[lane.receiver_index]))
        thick = base * 3 if lane is options.best else base
        cv2.line(img, start, end, LANE_COLORS[lane.status], thick, cv2.LINE_AA)
    return img


def draw_lanes_video(frame: np.ndarray, calib: PitchCalibration, positions: np.ndarray,
                     options: FrameOptions, occluders: np.ndarray | None = None, alpha: float = 0.6,
                     show_blocked: bool = True) -> np.ndarray:
    """Lanes on the grass in the camera view, behind the players (in place).

    The pitch is flat, so each lane is a straight line in the image between the projected ends.
    """
    h, w = frame.shape[:2]
    base = max(2, int(round(h / 400)))
    layer = frame.copy()
    mask = np.zeros((h, w), np.uint8)
    start = calib.to_image(options.start[None])[0]
    if not np.isfinite(start).all():
        return frame
    for lane in sorted(options.lanes, key=lambda ln: ln is options.best):
        if lane.status == BLOCKED and not show_blocked:
            continue
        end = calib.to_image(positions[lane.receiver_index][None])[0]
        if not np.isfinite(end).all():
            continue
        p1, p2 = tuple(int(round(v)) for v in start), tuple(int(round(v)) for v in end)
        thick = base * 3 if lane is options.best else base
        cv2.line(layer, p1, p2, LANE_COLORS[lane.status], thick, cv2.LINE_AA)
        cv2.line(mask, p1, p2, 255, thick, cv2.LINE_AA)
    if occluders is not None:                           # players stay in front of the lanes
        for x1, y1, x2, y2 in np.round(np.asarray(occluders).reshape(-1, 4)).astype(int):
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    m = mask > 0
    frame[m] = (alpha * layer[m] + (1 - alpha) * frame[m]).astype(np.uint8)
    return frame
