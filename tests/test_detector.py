"""Detector logic with a fake model (ultralytics / PyTorch not needed)."""

import sys
import types

import numpy as np
import pytest
import supervision as sv

from pitchlens.detection.classes import BALL, GOALKEEPER, PLAYER, REFEREE


class FakeYOLO:
    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee", 4: "advert"}

    def __init__(self, path):
        self.path = path


@pytest.fixture
def detector_cls(monkeypatch):
    """Import Detector with a stand-in 'ultralytics' module."""
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.delitem(sys.modules, "pitchlens.detection.detector", raising=False)
    from pitchlens.detection.detector import Detector
    return Detector


def _dets(rows):
    """rows: (x1, y1, x2, y2, conf, model class)"""
    a = np.array(rows, float).reshape(-1, 6)
    return sv.Detections(xyxy=a[:, :4], confidence=a[:, 4], class_id=a[:, 5].astype(int))


def test_classes_mapped_and_filtered(detector_cls):
    det = detector_cls("football.pt")
    frame_dets = _dets([[100, 100, 130, 180, 0.9, 2],     # player
                        [300, 100, 330, 180, 0.2, 2],     # player below confidence
                        [500, 100, 530, 180, 0.8, 1],     # goalkeeper
                        [700, 100, 730, 180, 0.7, 3],     # referee
                        [900, 100, 910, 110, 0.3, 0],     # ball
                        [950, 100, 960, 110, 0.2, 0],     # weaker second ball
                        [10, 10, 200, 40, 0.9, 4]])       # advert board: not a PitchLens class
    det._run = lambda images, conf, classes=None, imgsz=None: [frame_dets]
    out = det.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert sorted(out.players.class_id) == [GOALKEEPER, PLAYER, REFEREE]
    assert len(out.ball) == 1 and out.ball.confidence[0] == 0.3 and out.ball.class_id[0] == BALL


def test_tiles_find_the_ball_the_full_frame_missed(detector_cls):
    det = detector_cls("football.pt", ball_tiles=True, tile_size=640)
    calls = []

    def fake_run(images, conf, classes=None, imgsz=None):
        if isinstance(images, list):                       # tiles: ball found in the one at x 1280
            calls.append((len(images), classes, imgsz))
            return [_dets([[220, 600, 234, 614, 0.4, 0]]) if i == 2 else sv.Detections.empty()
                    for i in range(len(images))]
        return [_dets([[100, 100, 130, 180, 0.9, 2]])]     # full frame: a player, no ball

    det._run = fake_run
    out = det.detect(np.zeros((1080, 1920, 3), np.uint8))
    n_tiles, classes, imgsz = calls[0]
    assert classes == [0] and imgsz == 640 and n_tiles >= 4
    assert len(out.ball) == 1 and out.ball.class_id[0] == BALL
    x1 = out.ball.xyxy[0, 0]
    assert x1 > 1000                                      # shifted from tile to frame coordinates
    assert len(out.players) == 1


def test_no_tiling_by_default(detector_cls):
    det = detector_cls("football.pt")
    seen = []
    det._run = lambda images, conf, classes=None, imgsz=None: seen.append(images) or [sv.Detections.empty()]
    det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert len(seen) == 1 and not isinstance(seen[0], list)


def test_model_without_football_classes(detector_cls, monkeypatch):
    monkeypatch.setattr(FakeYOLO, "names", {0: "car", 1: "bus"})
    with pytest.raises(ValueError):
        detector_cls("cars.pt")
