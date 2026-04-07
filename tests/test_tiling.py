import numpy as np
import supervision as sv

from pitchlens.detection.tiling import make_tiles, merge_tile_detections, tiled_detections


def test_tiles_cover_the_frame_with_overlap():
    tiles = make_tiles(1920, 1080, tile=640, overlap=0.2)
    covered = np.zeros((1080, 1920), bool)
    for x1, y1, x2, y2 in tiles:
        assert x2 - x1 <= 640 and y2 - y1 <= 640
        covered[y1:y2, x1:x2] = True
    assert covered.all()
    xs = sorted({t[0] for t in tiles})
    assert all(b - a <= 512 for a, b in zip(xs, xs[1:]))          # 20% overlap: step 512
    assert make_tiles(500, 300, tile=640) == [(0, 0, 500, 300)]   # small frame: one tile


def _dets(boxes, conf=0.9, cls=0):
    boxes = np.array(boxes, float).reshape(-1, 4)
    return sv.Detections(xyxy=boxes, confidence=np.full(len(boxes), conf), class_id=np.full(len(boxes), cls))


def test_boxes_shifted_and_duplicates_merged():
    tiles = [(0, 0, 640, 640), (512, 0, 1152, 640)]
    # the same ball at frame x 600..615 seen by both tiles (in each tile's own coordinates)
    per_tile = [_dets([[600, 100, 615, 115]], 0.8), _dets([[88, 100, 103, 115]], 0.6)]
    merged = merge_tile_detections(per_tile, tiles)
    assert len(merged) == 1
    assert np.allclose(merged.xyxy[0], [600, 100, 615, 115]) and merged.confidence[0] == 0.8
    assert len(merge_tile_detections([sv.Detections.empty(), sv.Detections.empty()], tiles)) == 0


def test_small_ball_found_that_a_downscaled_pass_would_miss():
    """A fake detector that only 'sees' balls at least 10 px wide in the image it gets."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[700:714, 1500:1514] = 255                               # 14 px ball

    def predict(crops):
        out = []
        for crop in crops:
            ys, xs = np.nonzero(crop[..., 0])
            if len(xs) and xs.max() - xs.min() + 1 >= 10:
                out.append(_dets([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]))
            else:
                out.append(sv.Detections.empty())
        return out

    small = frame[::3, ::3]                                       # what a 640-wide full-frame pass sees
    assert len(predict([small])[0]) == 0                          # ball now ~5 px: missed
    found = tiled_detections(frame, predict, tile=640, overlap=0.2)
    assert len(found) == 1 and np.allclose(found.xyxy[0], [1500, 700, 1514, 714])
