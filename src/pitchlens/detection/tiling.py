"""Find small objects (the ball) by running the detector on overlapping tiles.

YOLO shrinks the whole frame to its input size (e.g. 1920 px wide -> 640), so a
ball 15 px wide in a broadcast frame becomes ~5 px and is easily missed.
Running the model on tiles of its own input size keeps full resolution:

  1. cut the frame into tiles of `tile` px that overlap by `overlap` (so a ball
     on a tile border is fully inside at least one tile);
  2. detect in every tile (in one batch);
  3. shift the boxes back to frame coordinates and merge duplicates from the
     overlaps with non-maximum suppression.

It costs one model run per tile, so it's meant for the ball only.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import supervision as sv


def _starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)                 # last tile flush with the edge
    return starts


def make_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """(x1, y1, x2, y2) tiles covering the frame, each at most `tile` px square."""
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in _starts(height, tile, overlap) for x in _starts(width, tile, overlap)]


def merge_tile_detections(per_tile: Sequence[sv.Detections], tiles: Sequence[tuple[int, int, int, int]],
                          iou: float = 0.5) -> sv.Detections:
    """Shifts each tile's detections to frame coordinates and removes duplicates."""
    shifted = []
    for dets, (x1, y1, _, _) in zip(per_tile, tiles):
        if len(dets) == 0:
            continue
        d = dets[np.arange(len(dets))]           # copy
        d.xyxy = d.xyxy + np.array([x1, y1, x1, y1], dtype=float)
        shifted.append(d)
    if not shifted:
        return sv.Detections.empty()
    merged = sv.Detections.merge(shifted)
    return merged.with_nms(threshold=iou, class_agnostic=False)


def tiled_detections(frame: np.ndarray, predict: Callable[[list[np.ndarray]], list[sv.Detections]],
                     tile: int = 640, overlap: float = 0.2, iou: float = 0.5) -> sv.Detections:
    """predict: list of image crops -> detections per crop (in crop coordinates)."""
    h, w = frame.shape[:2]
    tiles = make_tiles(w, h, tile, overlap)
    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
    return merge_tile_detections(predict(crops), tiles, iou)
