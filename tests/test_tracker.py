import numpy as np
import supervision as sv

from pitchlens.tracking.tracker import PlayerTracker, iou_matrix


def _dets(boxes, conf=None):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    conf = np.full(len(boxes), 0.9) if conf is None else np.array(conf, dtype=float)
    return sv.Detections(xyxy=boxes, confidence=conf, class_id=np.full(len(boxes), 2))


def test_iou():
    iou = iou_matrix(np.array([[0, 0, 10, 10]], float), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float))
    assert np.allclose(iou, [[1.0, 1 / 3, 0.0]], atol=1e-6)


def test_ids_stable_while_players_move():
    tracker = PlayerTracker(fps=25)
    seen = []
    for f in range(20):
        a = [10 + 3 * f, 50, 40 + 3 * f, 130]     # moving right
        b = [300 - 3 * f, 50, 330 - 3 * f, 130]   # moving left
        out = tracker.update(_dets([a, b]))
        if len(out) == 2:
            seen.append(tuple(out.tracker_id))
    assert len(seen) >= 18 and len(set(seen)) == 1


def test_id_survives_short_occlusion_and_low_conf():
    tracker = PlayerTracker(fps=25)
    box = lambda f: [100 + 4 * f, 50, 130 + 4 * f, 130]
    for f in range(5):
        tid = tracker.update(_dets([box(f)])).tracker_id
    for f in range(5, 10):                       # player hidden for 5 frames
        tracker.update(_dets([]))
    out = tracker.update(_dets([box(10)], conf=[0.3]))  # reappears, low confidence
    assert list(out.tracker_id) == list(tid)


def test_empty_frame():
    assert len(PlayerTracker().update(_dets([]))) == 0


def test_coast_moves_tracks_between_detections():
    tracker = PlayerTracker(fps=25)
    box = lambda f: [100 + 4 * f, 50, 130 + 4 * f, 130]
    for f in range(4):                                      # detected every frame: velocity learnt
        out = tracker.update(_dets([box(f)]))
    tid = out.tracker_id[0]
    coasted = tracker.coast(since_detection=1)              # frame 4 not detected
    assert list(coasted.tracker_id) == [tid] and coasted.class_id[0] == 2
    assert np.allclose(coasted.xyxy[0], box(4), atol=1.0)   # moved by the tracked velocity
    out = tracker.update(_dets([box(5)]))                   # frame 5 detected again: same id
    assert list(out.tracker_id) == [tid]


def test_coast_hides_tracks_already_lost():
    tracker = PlayerTracker(fps=25)
    for f in range(3):
        tracker.update(_dets([[100, 50, 130, 130]]))
    tracker.update(_dets([]))                               # player missed on a detected frame
    assert len(tracker.coast(since_detection=1)) == 0        # not shown while coasting either
