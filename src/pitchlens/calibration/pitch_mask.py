"""Which pixels belong to the (static) pitch surface?

Camera-motion estimation tracks texture between frames. Only the pitch itself
is reliable for that:
  * players and the ball move on their own, so they would look like camera motion
  * the crowd moves, and stands/ad boards are not on the pitch plane
  * broadcast graphics (score bug, clock) stay fixed on screen and would suggest
    the camera never moves

So the mask is: the grass region (with its white lines filled in), minus
padded boxes around every detected object, eroded away from the grass edge.
"""

from __future__ import annotations

import cv2
import numpy as np

# HSV range for grass (OpenCV hue is 0-179). Low V threshold keeps shadowed grass.
GRASS_LOW = (35, 40, 25)
GRASS_HIGH = (85, 255, 255)


def grass_mask(frame: np.ndarray, min_area_fraction: float = 0.02, line_width: int = 15) -> np.ndarray:
    """uint8 mask (255 = pitch). Lines and players inside the grass are filled in.

    line_width: widest white line (pixels) to bridge. Lines that cross the whole
    frame would otherwise split the grass into separate regions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((line_width, line_width), np.uint8))

    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_fraction * frame.shape[0] * frame.shape[1]
    mask = np.zeros(frame.shape[:2], np.uint8)
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(mask, big, -1, 255, thickness=cv2.FILLED)
    return mask


def background_mask(frame: np.ndarray, exclude_boxes: np.ndarray | None = None,
                    box_padding: float = 0.2, edge_margin: int = 15) -> np.ndarray:
    """uint8 mask of pitch pixels that are safe to track camera motion on.

    exclude_boxes: (N, 4) xyxy boxes of players/ball/referees to cut out.
    box_padding:   grow each box by this fraction of its size (shadows, blur).
    edge_margin:   pixels to shrink the pitch region away from stands/graphics.
    """
    mask = grass_mask(frame)
    if edge_margin > 0:
        k = 2 * edge_margin + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    if exclude_boxes is not None and len(exclude_boxes):
        h, w = mask.shape
        boxes = np.asarray(exclude_boxes, dtype=float).reshape(-1, 4)
        size = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        pad = np.hstack([size, size]) * box_padding
        grown = boxes + pad * np.array([-1, -1, 1, 1])
        grown = np.clip(np.round(grown), 0, [w, h, w, h]).astype(int)
        for x1, y1, x2, y2 in grown:
            mask[y1:y2, x1:x2] = 0
    return mask
