"""Football pitch model in real-world meters.

Coordinate system (top-down view):
    origin (0, 0) = top-left corner flag
    x runs along the length (0 -> left goal line, `length` -> right goal line)
    y runs across the width (0 -> top touchline, `width` -> bottom touchline)

Box, circle and spot sizes are fixed by the Laws of the Game; only the overall
length/width vary between stadiums (105 x 68 m is the common standard).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt

import numpy as np

PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CIRCLE_RADIUS = 9.15


@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0

    @property
    def center(self) -> tuple[float, float]:
        return self.length / 2, self.width / 2

    def keypoints(self) -> dict[str, tuple[float, float]]:
        """Named landmarks that are easy to spot on broadcast footage."""
        L, W = self.length, self.width
        cx, cy = self.center
        pa, ga = PENALTY_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2
        # where the penalty arc meets the edge of the penalty area
        arc_dy = sqrt(CIRCLE_RADIUS**2 - (PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE) ** 2)

        kp = {
            "corner_top_left": (0.0, 0.0),
            "corner_bottom_left": (0.0, W),
            "corner_top_right": (L, 0.0),
            "corner_bottom_right": (L, W),
            "halfway_top": (cx, 0.0),
            "halfway_bottom": (cx, W),
            "center_spot": (cx, cy),
            "center_circle_top": (cx, cy - CIRCLE_RADIUS),
            "center_circle_bottom": (cx, cy + CIRCLE_RADIUS),
            "center_circle_left": (cx - CIRCLE_RADIUS, cy),
            "center_circle_right": (cx + CIRCLE_RADIUS, cy),
        }
        for side, x0, d in (("left", 0.0, 1.0), ("right", L, -1.0)):
            def at(depth: float) -> float:
                return x0 + d * depth

            kp |= {
                f"{side}_penalty_area_goalline_top": (x0, cy - pa),
                f"{side}_penalty_area_corner_top": (at(PENALTY_AREA_DEPTH), cy - pa),
                f"{side}_penalty_area_corner_bottom": (at(PENALTY_AREA_DEPTH), cy + pa),
                f"{side}_penalty_area_goalline_bottom": (x0, cy + pa),
                f"{side}_goal_area_goalline_top": (x0, cy - ga),
                f"{side}_goal_area_corner_top": (at(GOAL_AREA_DEPTH), cy - ga),
                f"{side}_goal_area_corner_bottom": (at(GOAL_AREA_DEPTH), cy + ga),
                f"{side}_goal_area_goalline_bottom": (x0, cy + ga),
                f"{side}_penalty_spot": (at(PENALTY_SPOT_DISTANCE), cy),
                f"{side}_arc_top": (at(PENALTY_AREA_DEPTH), cy - arc_dy),
                f"{side}_arc_bottom": (at(PENALTY_AREA_DEPTH), cy + arc_dy),
            }
        return kp

    def lines(self, arc_steps: int = 64) -> list[np.ndarray]:
        """All pitch markings as polylines (each an (N, 2) array in meters)."""
        L, W = self.length, self.width
        cx, cy = self.center
        pa, ga = PENALTY_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2

        def rect(x1, y1, x2, y2):
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=float)

        def arc(center, radius, a0, a1):
            t = np.linspace(a0, a1, arc_steps)
            return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)

        out = [
            rect(0, 0, L, W),
            np.array([[cx, 0], [cx, W]], dtype=float),
            arc((cx, cy), CIRCLE_RADIUS, 0, 2 * np.pi),
        ]
        # half-angle of the penalty arc portion that lies outside the box
        half = acos((PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE) / CIRCLE_RADIUS)
        out += [
            rect(0, cy - pa, PENALTY_AREA_DEPTH, cy + pa),
            rect(0, cy - ga, GOAL_AREA_DEPTH, cy + ga),
            arc((PENALTY_SPOT_DISTANCE, cy), CIRCLE_RADIUS, -half, half),
            rect(L - PENALTY_AREA_DEPTH, cy - pa, L, cy + pa),
            rect(L - GOAL_AREA_DEPTH, cy - ga, L, cy + ga),
            arc((L - PENALTY_SPOT_DISTANCE, cy), CIRCLE_RADIUS, np.pi - half, np.pi + half),
        ]
        return out

    def spots(self) -> list[tuple[float, float]]:
        cx, cy = self.center
        return [(cx, cy), (PENALTY_SPOT_DISTANCE, cy), (self.length - PENALTY_SPOT_DISTANCE, cy)]
