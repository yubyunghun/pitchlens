import sys
from pathlib import Path

import numpy as np

from pitchlens.calibration.anchored import solve_calibrations
from pitchlens.calibration.camera_motion import CameraMotionTracker
from pitchlens.calibration.keyframes import CalibrationSet

sys.path.insert(0, str(Path(__file__).parent))
from test_calibration_tracker import PLAYERS, _calib_for, _error_m  # noqa: E402
from test_camera_motion import _camera, _pitch_texture, _view  # noqa: E402

SIZE = (960, 540)


def _cams(n, speed=12):
    return [_camera(150 + speed * t, 120 + 2 * t, zoom=1 + 0.002 * t) for t in range(n)]


def _exact_motions(cams):
    return [None] + [c @ np.linalg.inv(p) for p, c in zip(cams[:-1], cams[1:])]


def _keys(cams, *frames):
    return CalibrationSet({f: _calib_for(cams[f]) for f in frames})


def _assert_exact(res, cams, frames):
    for t in frames:
        assert res[t] is not None, f"frame {t} has no calibration"
        assert _error_m(res[t].calib, cams[t], PLAYERS) < 1e-6, f"frame {t} is off"


def test_nearest_keyframe_forward_and_backward():
    cams = _cams(41)
    res = solve_calibrations(_exact_motions(cams), _keys(cams, 0, 30), SIZE)
    _assert_exact(res, cams, range(41))
    assert (res[10].keyframe, res[10].distance) == (0, 10)
    assert (res[20].keyframe, res[20].distance) == (30, 10)   # chained backward
    assert (res[40].keyframe, res[40].distance) == (30, 10)
    assert res[15].keyframe == 0                               # tie -> earlier keyframe


def test_frames_before_first_keyframe():
    cams = _cams(25)
    res = solve_calibrations(_exact_motions(cams), _keys(cams, 20), SIZE)
    _assert_exact(res, cams, range(25))
    assert res[0].distance == 20


def test_cut_breaks_the_chain():
    cams = _cams(30)
    motions = _exact_motions(cams)
    res = solve_calibrations(motions, _keys(cams, 0), SIZE, breaks=[15])
    _assert_exact(res, cams, range(15))
    assert all(r is None for r in res[15:])                    # nothing known after the cut
    res = solve_calibrations(motions, _keys(cams, 0, 25), SIZE, breaks=[15])
    _assert_exact(res, cams, range(30))                        # second keyframe covers it
    assert res[15].keyframe == 25 and res[14].keyframe == 0


def test_implausible_motion_is_a_break():
    cams = _cams(20)
    motions = _exact_motions(cams)
    motions[10] = np.array([[2.0, 0.0, -500], [0.0, 2.0, -300], [0, 0, 1]])
    res = solve_calibrations(motions, _keys(cams, 0), SIZE)
    assert res[9] is not None and res[10] is None


def test_short_gap_bridged_long_gap_breaks():
    cams = [_camera(150 + 20 * t, 120) for t in range(40)]    # constant speed
    motions = _exact_motions(cams)
    for t in (5, 6, 7):
        motions[t] = None                                      # 3 failed estimates
    res = solve_calibrations(motions, _keys(cams, 0), SIZE, max_gap=5)
    _assert_exact(res, cams, range(40))

    for t in range(20, 30):
        motions[t] = None                                      # 10 failures in a row
    res = solve_calibrations(motions, _keys(cams, 0), SIZE, max_gap=5)
    assert res[19] is not None and all(r is None for r in res[20:])


def test_keyframes_reduce_drift_with_real_optical_flow():
    world = _pitch_texture(4)
    xs = np.concatenate([np.linspace(0, 600, 60), np.full(30, 600), np.linspace(600, 300, 60)])
    cams = [_camera(150 + x, 120 + 0.1 * x, zoom=1 + 0.0003 * x) for x in xs]
    tracker = CameraMotionTracker()
    motions = [tracker.update(_view(world, c)) for c in cams]
    pts = np.array([[60.0, 30.0], [80.0, 40.0], [90.0, 55.0]])

    def worst(keys):
        res = solve_calibrations(motions, _keys(cams, *keys), SIZE)
        return max(_error_m(r.calib, c, pts) for r, c in zip(res, cams))

    one, three = worst([0]), worst([0, 75, 149])
    assert three < one / 2 and three < 0.1
