"""VAR-style offside line drawn in the camera view.

A translucent band with a solid center line, in the defending team's color,
running across the pitch at the offside line. Player boxes are cut out of the
overlay so players appear in front of the line, as in broadcast graphics. The
defender who sets the line gets a ring at their feet.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.offside.line import BALL, HALFWAY, OffsideLine
from pitchlens.offside.position import OFFSIDE, TOO_CLOSE, PositionCheck
from pitchlens.offside.projection import project_band, project_pitch_line


FLAG_STYLE = {
    OFFSIDE: ("OFFSIDE POS", (40, 40, 230)),     # red
    TOO_CLOSE: ("TOO CLOSE", (0, 200, 255)),     # amber
}
FLAG_TEXT = {OFFSIDE: (255, 255, 255), TOO_CLOSE: (0, 0, 0)}  # readable on each background


def flag_color(status: str) -> tuple[int, int, int] | None:
    style = FLAG_STYLE.get(status)
    return None if style is None else style[1]


def draw_position_flags(frame: np.ndarray, boxes: np.ndarray, checks: dict[int, PositionCheck]) -> np.ndarray:
    """Tag above each attacker that is in (or too close to call for) an offside position."""
    h = frame.shape[0]
    scale = max(0.4, h / 1800)
    for i, check in checks.items():
        style = FLAG_STYLE.get(check.status)
        if style is None:
            continue
        text, color = style
        label = f"{text} {check.margin_m:+.1f}m"
        x1, y1, x2, _ = boxes[i]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cx = int(round((x1 + x2) / 2))
        bx = int(np.clip(cx - tw // 2 - 4, 0, frame.shape[1] - tw - 8))
        by = int(max(th + 8, y1 - 6))
        cv2.rectangle(frame, (bx, by - th - 6), (bx + tw + 8, by + 3), color, -1)
        cv2.putText(frame, label, (bx + 4, by - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    FLAG_TEXT[check.status], 1, cv2.LINE_AA)
    return frame


def _blend(frame: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> None:
    m = mask > 0
    frame[m] = (alpha * layer[m] + (1 - alpha) * frame[m]).astype(np.uint8)


def draw_offside_overlay(frame: np.ndarray, calib: PitchCalibration, line: OffsideLine,
                         color: tuple[int, int, int], occluders: np.ndarray | None = None,
                         defender_foot: np.ndarray | None = None, band_m: float = 0.5,
                         band_alpha: float = 0.35, line_alpha: float = 0.9) -> np.ndarray:
    """Draws the offside line into `frame` (in place) and returns it.

    occluders:     (N, 4) xyxy boxes drawn in front of the line (players).
    defender_foot: image point of the defender who sets the line.
    """
    seg = project_pitch_line(calib, line.x)
    if seg is None:
        return frame
    h, w = frame.shape[:2]
    band = project_band(calib, line.x, band_m / 2)
    thick = max(2, int(round(h / 360)))

    band_mask = np.zeros((h, w), np.uint8)
    line_mask = np.zeros((h, w), np.uint8)
    if band is not None:
        cv2.fillPoly(band_mask, [np.round(band).astype(np.int32)], 255)
    p1, p2 = (tuple(int(v) for v in p) for p in seg)
    cv2.line(line_mask, p1, p2, 255, thick, cv2.LINE_AA)
    if occluders is not None:
        for x1, y1, x2, y2 in np.round(np.asarray(occluders).reshape(-1, 4)).astype(int):
            band_mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
            line_mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0

    layer = np.empty_like(frame)
    layer[:] = color
    _blend(frame, layer, band_mask, band_alpha)
    _blend(frame, layer, line_mask, line_alpha)

    if defender_foot is not None and np.isfinite(defender_foot).all():
        cx, cy = (int(round(v)) for v in defender_foot)
        r = max(10, int(round(h / 54)))
        cv2.ellipse(frame, (cx, cy), (r, r // 3), 0, 0, 360, (255, 255, 255), thick + 2, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (r, r // 3), 0, 0, 360, color, thick, cv2.LINE_AA)

    label = "offside line"
    if line.source == BALL:
        label += " (ball)"
    elif line.source == HALFWAY:
        label += " (halfway)"
    if line.keeper_assumed:
        label += " - keeper assumed"
    top = min((p1, p2), key=lambda p: p[1])
    scale = h / 1200
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(top[0] - tw // 2, 4, w - tw - 4))
    y = int(np.clip(top[1] - 8, th + 8, h - 4))
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), (20, 20, 20), -1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return frame
