"""Top-down 2D pitch ("minimap") with players drawn at their real positions."""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.pitch import Pitch

GRASS_DARK = (38, 110, 52)
GRASS_LIGHT = (46, 126, 60)
LINE_COLOR = (235, 235, 235)


class Minimap:
    def __init__(self, pitch: Pitch = Pitch(), width_px: int = 420, margin_m: float = 3.0):
        self.pitch = pitch
        self.margin = margin_m
        self.scale = width_px / (pitch.length + 2 * margin_m)  # pixels per meter
        self.width = width_px
        self.height = int(round((pitch.width + 2 * margin_m) * self.scale))
        self._base = self._render_base()

    def to_px(self, xy_m) -> np.ndarray:
        return np.round((np.asarray(xy_m, dtype=float) + self.margin) * self.scale).astype(int)

    def from_px(self, xy_px) -> np.ndarray:
        return np.asarray(xy_px, dtype=float) / self.scale - self.margin

    def _render_base(self) -> np.ndarray:
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        img[:] = GRASS_DARK
        stripe = self.pitch.length / 12  # mowing stripes
        for i in range(0, 12, 2):
            x1, _ = self.to_px((i * stripe, 0))
            x2, _ = self.to_px(((i + 1) * stripe, 0))
            img[:, x1:x2] = GRASS_LIGHT
        thick = max(1, int(round(self.scale * 0.2)))
        for line in self.pitch.lines():
            cv2.polylines(img, [self.to_px(line).reshape(-1, 1, 2)], False, LINE_COLOR, thick, cv2.LINE_AA)
        for spot in self.pitch.spots():
            cv2.circle(img, tuple(self.to_px(spot)), max(2, thick + 1), LINE_COLOR, -1, cv2.LINE_AA)
        return img

    def render(self, positions: np.ndarray, colors: list[tuple[int, int, int]],
               labels: list[str] | None = None, ball: np.ndarray | None = None) -> np.ndarray:
        """positions: (N, 2) pitch meters (NaN rows are skipped)."""
        img = self._base.copy()
        r = max(4, int(round(self.scale * 0.9)))
        lim = np.array([self.pitch.length, self.pitch.width]) + self.margin
        for i, (pos, color) in enumerate(zip(np.asarray(positions).reshape(-1, 2), colors)):
            if np.isnan(pos).any() or (pos < -self.margin).any() or (pos > lim).any():
                continue
            p = tuple(self.to_px(pos))
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, p, r, (20, 20, 20), 1, cv2.LINE_AA)
            if labels and labels[i]:
                cv2.putText(img, labels[i], (p[0] - r, p[1] - r - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1, cv2.LINE_AA)
        if ball is not None and len(ball) and not np.isnan(ball).any():
            p = tuple(self.to_px(np.asarray(ball).reshape(2)))
            cv2.circle(img, p, max(3, r // 2), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, max(3, r // 2), (0, 0, 0), 1, cv2.LINE_AA)
        return img


def draw_attack_arrows(img: np.ndarray, minimap: Minimap,
                       arrows: list[tuple[tuple[int, int, int], int]]) -> np.ndarray:
    """Arrows in the top margin of a rendered minimap: (team color, +1 = attacks right / -1 = left)."""
    y = max(4, int(round(minimap.margin * minimap.scale / 2)))
    half = 4.0  # arrow half-length in meters
    for i, (color, direction) in enumerate(arrows):
        cx = minimap.pitch.length * (0.3 + 0.4 * i)        # first team left of center, second right
        x1, _ = minimap.to_px((cx - half * direction, 0))
        x2, _ = minimap.to_px((cx + half * direction, 0))
        cv2.arrowedLine(img, (int(x1), y), (int(x2), y), (20, 20, 20), 4, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(img, (int(x1), y), (int(x2), y), color, 2, cv2.LINE_AA, tipLength=0.35)
    return img


def draw_offside_lines(img: np.ndarray, minimap: Minimap,
                       lines: list[tuple[float, tuple[int, int, int], bool]]) -> np.ndarray:
    """Vertical offside lines across the pitch: (pitch x, defending team color, dashed).

    Dashed means less certain (the goalkeeper was out of view and assumed).
    """
    thick = max(2, int(round(minimap.scale * 0.35)))
    for x, color, dashed in lines:
        (px, y_top), (_, y_bot) = minimap.to_px((x, 0)), minimap.to_px((x, minimap.pitch.width))
        if not dashed:
            cv2.line(img, (int(px), int(y_top)), (int(px), int(y_bot)), color, thick, cv2.LINE_AA)
            continue
        dash = max(4, int(round(minimap.scale * 2)))
        for y in range(int(y_top), int(y_bot), 2 * dash):
            cv2.line(img, (int(px), y), (int(px), min(y + dash, int(y_bot))), color, thick, cv2.LINE_AA)
    return img


def draw_player_rings(img: np.ndarray, minimap: Minimap, positions: np.ndarray,
                      colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Rings around player dots (e.g. attackers in an offside position)."""
    r = max(4, int(round(minimap.scale * 0.9))) + 3
    for pos, color in zip(np.asarray(positions, dtype=float).reshape(-1, 2), colors):
        if np.isfinite(pos).all():
            cv2.circle(img, tuple(minimap.to_px(pos)), r, color, 2, cv2.LINE_AA)
    return img


def paste_minimap(frame: np.ndarray, minimap: np.ndarray, pad: int = 16) -> np.ndarray:
    """Places the minimap (with a thin border) at the bottom-center of the frame."""
    h, w = minimap.shape[:2]
    H, W = frame.shape[:2]
    if h + pad + 2 > H or w + 4 > W:
        return frame
    x, y = (W - w) // 2, H - h - pad
    cv2.rectangle(frame, (x - 2, y - 2), (x + w + 1, y + h + 1), (20, 20, 20), -1)
    frame[y:y + h, x:x + w] = minimap
    return frame


def draw_projected_pitch(frame: np.ndarray, calib: PitchCalibration,
                         color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Draws the pitch lines through the calibration so you can check it lines up."""
    for line in calib.pitch.lines():
        pts = calib.to_image(line)
        for a, b in zip(pts[:-1], pts[1:]):
            if np.isfinite(a).all() and np.isfinite(b).all() and np.abs(np.r_[a, b]).max() < 1e5:
                cv2.line(frame, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                         color, thickness, cv2.LINE_AA)
    return frame
