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
