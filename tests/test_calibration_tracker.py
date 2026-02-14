import sys
from pathlib import Path

import numpy as np

from pitchlens.calibration.camera_motion import CameraMotionTracker, MotionResult
from pitchlens.calibration.calibration_tracker import CalibrationTracker, is_plausible_motion
from pitchlens.calibration.homography import PitchCalibration

sys.path.insert(0, str(Path(__file__).parent))
from test_camera_motion import _camera, _pitch_texture, _view  # noqa: E402

PX_PER_M = 10.0  # the synthetic world texture is drawn at 10 px per pitch meter
TO_METERS = np.diag([1 / PX_PER_M, 1 / PX_PER_M, 1.0])


def _calib_for(cam):
    """Exact image -> pitch calibration for a camera (world px -> image px)."""
    return PitchCalibration(TO_METERS @ np.linalg.inv(cam), (960, 540))


def _motion(cam_prev, cam_curr):
    return MotionResult(cam_curr @ np.linalg.inv(cam_prev), 100, 100)


def _error_m(calib, cam, pitch_pts):
    img = (cam @ np.c_[pitch_pts * PX_PER_M, np.ones(len(pitch_pts))].T).T
    img = img[:, :2] / img[:, 2:]
    return np.linalg.norm(calib.to_pitch(img) - pitch_pts, axis=1).max()


PLAYERS = np.array([[30.0, 25.0], [40.0, 35.0], [55.0, 30.0]])


def test_exact_motions_keep_calibration_exact():
    cams = [_camera(200 + 15 * t, 150 + 2 * t, zoom=1 + 0.004 * t) for t in range(30)]
    tracker = CalibrationTracker(_calib_for(cams[0]))
    for prev, curr in zip(cams[:-1], cams[1:]):
        calib = tracker.update(_motion(prev, curr))
    assert _error_m(calib, cams[-1], PLAYERS) < 1e-6


def test_missing_frame_predicted_from_last_motion():
    cams = [_camera(200 + 20 * t, 150) for t in range(6)]   # constant-speed pan
    tracker = CalibrationTracker(_calib_for(cams[0]))
    tracker.update(_motion(cams[0], cams[1]))
    tracker.update(_motion(cams[1], cams[2]))
    calib = tracker.update(None)                            # motion failed on frame 3
    assert tracker.lost_frames == 1
    assert _error_m(calib, cams[3], PLAYERS) < 1e-6
    tracker.update(_motion(cams[3], cams[4]))
    assert tracker.lost_frames == 0


def test_lost_after_too_many_failures_and_reset():
    cam = _camera(200, 150)
    tracker = CalibrationTracker(_calib_for(cam), max_lost_frames=3)
    results = [tracker.update(None) for _ in range(4)]
    assert results[2] is not None and results[3] is None and tracker.is_lost
    assert tracker.update(_motion(cam, cam)) is None        # stays lost until re-anchored
    tracker.reset(_calib_for(cam))
    assert tracker.update(_motion(cam, cam)) is not None


def test_plausibility_check():
    size = (960, 540)
    shift = lambda dx: np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], float)
    zoom = lambda z: np.array([[z, 0, 480 * (1 - z)], [0, z, 270 * (1 - z)], [0, 0, 1]], float)
    assert is_plausible_motion(np.eye(3), size)
    assert is_plausible_motion(shift(150), size)           # fast pan
    assert is_plausible_motion(zoom(1.05), size)           # zooming in
    assert not is_plausible_motion(shift(500), size)       # jump of half the frame
    assert not is_plausible_motion(zoom(1.5), size)        # sudden 50% zoom
    assert not is_plausible_motion(np.diag([-1.0, 1, 1]), size)  # mirrored


def test_camera_cut_does_not_corrupt_calibration():
    cams = [_camera(200 + 10 * t, 150) for t in range(5)]
    tracker = CalibrationTracker(_calib_for(cams[0]))
    tracker.update(_motion(cams[0], cams[1]))
    tracker.update(_motion(cams[1], cams[2]))
    cut = MotionResult(np.array([[2.0, 0.3, -400], [0.1, 1.8, -200], [0, 0, 1]]), 100, 90)
    calib = tracker.update(cut)                            # rejected -> predicted instead
    assert tracker.lost_frames == 1
    assert _error_m(calib, cams[3], PLAYERS) < 1e-6


def test_end_to_end_with_optical_flow_drift():
    """Real motion estimation on rendered frames: how far does the calibration drift?"""
    world = _pitch_texture(4)
    cams = [_camera(150 + 12 * t, 120 + 3 * t, zoom=1 + 0.002 * t) for t in range(40)]
    motion = CameraMotionTracker()
    motion.update(_view(world, cams[0]))
    tracker = CalibrationTracker(_calib_for(cams[0]))
    for cam in cams[1:]:
        calib = tracker.update(motion.update(_view(world, cam)))
        assert calib is not None
    # 40 frames of pan + zoom: player positions still within 10 cm
    pts = np.array([[60.0, 30.0], [80.0, 40.0], [90.0, 55.0]])
    assert _error_m(calib, cams[-1], pts) < 0.1
