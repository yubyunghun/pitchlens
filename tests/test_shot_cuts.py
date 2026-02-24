import sys
from pathlib import Path

import cv2
import numpy as np

from pitchlens.calibration.shot_cuts import AppearanceChange, find_cuts
from pitchlens.pipeline import detect_cuts

sys.path.insert(0, str(Path(__file__).parent))
import test_pipeline as tp  # noqa: E402
from test_camera_motion import _camera, _pitch_texture, _view  # noqa: E402


def test_find_cuts_rules():
    changes = [0.0, 0.02, 0.9, 0.3, 0.3, 0.1, 0.6]
    motion = [False, True, True, False, True, False, True]
    # 2: huge change -> cut even though motion was measured
    # 3: moderate change + motion failed -> cut
    # 4: moderate change but motion measured -> continuous
    # 5: motion failed but tiny change (blur) -> not a cut
    # 6: huge change -> cut
    assert find_cuts(changes, motion) == [2, 3, 6]


def test_appearance_change_pan_vs_cut():
    world = _pitch_texture(5)
    meter = AppearanceChange()
    assert meter.update(_view(world, _camera(150, 120))) == 0.0           # first frame
    pan = [meter.update(_view(world, _camera(150 + 12 * t, 120))) for t in range(1, 6)]
    assert max(pan) < 0.1
    assert meter.update(tp._close_up()) > 0.5
    blurred = AppearanceChange()
    f = _view(world, _camera(200, 120))
    blurred.update(f)
    assert blurred.update(cv2.blur(f, (25, 1))) < 0.25                  # heavy blur is not a cut


def test_pipeline_cut_to_close_up_and_back(tmp_path, monkeypatch):
    def cam(t):   # wide shot, close-up for frames 15-19, then the wide camera somewhere else
        if t < 15:
            return _camera(150 + 12 * t, 120 + 2 * t)
        return _camera(60 + 10 * t, 90, zoom=0.9)

    monkeypatch.setattr(tp, "_cam", cam)
    monkeypatch.setattr(tp, "CLOSE_UP_FRAMES", range(15, 20))

    # only the first shot calibrated: nothing after the cut may get (wrong) coordinates
    records, rows = tp._run(tmp_path / "a", track_camera=True, key_frames=(0,))
    cuts = detect_cuts(records)
    assert 15 in cuts and 20 in cuts
    assert all(r["x_m"] == "" for r in rows if int(r["frame"]) >= 15)
    assert tp._position_errors([r for r in rows if int(r["frame"]) < 15]).max() < 0.3

    # with a keyframe in the third shot, its frames are calibrated again (backward to frame 20)
    _, rows = tp._run(tmp_path / "b", track_camera=True, key_frames=(0, 30))
    after = [r for r in rows if int(r["frame"]) >= 20]
    assert after and all(r["x_m"] != "" for r in after)
    assert tp._position_errors(after).max() < 0.3
    assert all(r["x_m"] == "" for r in rows if 15 <= int(r["frame"]) < 20)   # close-up has no pitch


def test_no_cuts_in_continuous_clip(tmp_path):
    records, _ = tp._run(tmp_path, track_camera=True)
    assert detect_cuts(records) == []
