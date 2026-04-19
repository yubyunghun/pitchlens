"""Speed options: detecting only every Nth frame."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import supervision as sv

from pitchlens.detection.classes import PLAYER

sys.path.insert(0, str(Path(__file__).parent))
import test_pipeline as tp  # noqa: E402


class EveryNthDetector:
    """Fake detector for the synthetic clip, called only on frames 0, n, 2n, ..."""

    def __init__(self, every):
        self.every, self.calls = every, 0

    def detect(self, frame):
        boxes = tp._boxes(self.calls * self.every)
        self.calls += 1
        players = sv.Detections(xyxy=boxes, confidence=np.full(len(boxes), 0.9), class_id=np.full(len(boxes), PLAYER))
        return SimpleNamespace(players=players, ball=sv.Detections.empty())


def _run(tmp_path, every):
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.pipeline import analyze, offside_inputs, solve
    from pitchlens.tracking.team_assigner import TeamAssigner

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    assigner.collect(first, tp._boxes(0))
    assigner.collect(first, tp._boxes(0))
    assigner.fit()
    detector = EveryNthDetector(every)
    records = analyze(str(video), detector, assigner, fps=25, detect_every=every)
    calibs = solve(records, CalibrationSet({0: tp._calib_for(tp._cam(0))}), tp.SIZE)
    return detector.calls, records, offside_inputs(records, calibs)[0]


def _position_errors(players):
    errs = []
    for t, fp in enumerate(players):
        truth = np.array([p for _, p in tp._truth(t)])
        for pos in fp.positions:
            if np.isfinite(pos).all():
                errs.append(np.linalg.norm(truth - pos, axis=1).min())
    return np.array(errs)


def test_detecting_every_other_frame(tmp_path):
    calls, records, players = _run(tmp_path, every=2)
    assert calls == tp.N_FRAMES // 2                              # half the detector work
    assert len(records) == tp.N_FRAMES
    shown = [len(r.players) for r in records[4:]]
    assert min(shown) == 6                                         # players present on skipped frames too
    ids = {tuple(sorted(r.players.tracker_id)) for r in records[4:]}
    assert len(ids) == 1                                           # same ids throughout
    errs = _position_errors(players[4:])
    assert errs.max() < 0.5, f"worst {errs.max():.2f} m"


def test_stage_timer():
    from pitchlens.timing import StageTimer
    now = iter([0.0, 2.0, 2.0, 2.5, 3.0, 4.0])
    timer = StageTimer(clock=lambda: next(now))
    with timer("detection"):
        pass                                                      # 2.0 s
    with timer("tracking"):
        pass                                                      # 0.5 s
    with timer("detection"):
        pass                                                      # 1.0 s
    timer.add("rendering", 0.5)
    assert timer.seconds == {"detection": 3.0, "tracking": 0.5, "rendering": 0.5}
    text = timer.summary(frames=100)
    assert "4.0 s, 25.0 frames/s overall" in text
    assert text.splitlines()[1].split()[0] == "detection"        # biggest stage first
    assert "nothing measured" in StageTimer().summary()


def test_batched_detection_is_one_model_call(monkeypatch):
    import types
    import test_detector as td
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=td.FakeYOLO))
    monkeypatch.delitem(sys.modules, "pitchlens.detection.detector", raising=False)
    from pitchlens.detection.detector import Detector
    det = Detector("football.pt")
    calls = []

    def fake_run(images, conf, classes=None, imgsz=None):
        calls.append(len(images) if isinstance(images, list) else 1)
        n = len(images) if isinstance(images, list) else 1
        return [td._dets([[100 + k, 100, 130 + k, 180, 0.9, 2], [900, 100, 910, 110, 0.3, 0]]) for k in range(n)]

    det._run = fake_run
    frames = [np.zeros((720, 1280, 3), np.uint8)] * 4
    out = det.detect_batch(frames)
    assert calls == [4] and len(out) == 4
    assert all(len(o.players) == 1 and len(o.ball) == 1 for o in out)
    assert out[3].players.xyxy[0, 0] == 103                       # each frame keeps its own results
    assert det.detect_batch([]) == []


class BatchingDetector(EveryNthDetector):
    """Same fake detector, but also offers detect_batch (records batch sizes)."""

    def __init__(self, every):
        super().__init__(every)
        self.batches = []

    def detect_batch(self, frames):
        self.batches.append(len(frames))
        return [self.detect(f) for f in frames]


def test_batched_analysis_gives_the_same_result(tmp_path):
    from pitchlens.pipeline import analyze
    from pitchlens.timing import StageTimer
    from pitchlens.tracking.team_assigner import TeamAssigner

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    assigner.collect(first, tp._boxes(0))
    assigner.collect(first, tp._boxes(0))
    assigner.fit()
    one = analyze(str(video), EveryNthDetector(2), assigner, fps=25, detect_every=2)
    det, timer = BatchingDetector(2), StageTimer()
    batched = analyze(str(video), det, assigner, fps=25, detect_every=2, batch=4, timer=timer)
    assert det.batches[0] == 2 and sum(det.batches) == tp.N_FRAMES // 2     # 4 frames read, 2 detected
    assert len(one) == len(batched)
    def ids(rec):                                               # empty frames have no tracker ids
        return [] if rec.players.tracker_id is None else list(rec.players.tracker_id)

    for a, b in zip(one, batched):
        assert np.allclose(a.players.xyxy, b.players.xyxy) and ids(a) == ids(b)
    assert {"detection", "tracking + teams", "camera motion + cuts"} <= set(timer.seconds)
