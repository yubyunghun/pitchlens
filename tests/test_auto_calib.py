import numpy as np
import pytest

from pitchlens.calibration.auto_calib import KEYPOINT_NAMES, calibration_from_keypoints
from pitchlens.calibration.homography import apply_homography
from pitchlens.calibration.pitch import Pitch

SIZE = (1280, 720)
# broadcast-like camera on the left half: pitch meters -> image pixels
CAM = np.array([[14.0, -6.0, 300.0], [0.4, 5.0, 250.0], [0.0, -0.004, 1.0]])


def _predicted(noise_px=0.0, seed=0):
    """What a keypoint model would output: true image positions, confidence 0.9 if on screen."""
    rng = np.random.default_rng(seed)
    pts = np.array([Pitch().keypoints()[n] for n in KEYPOINT_NAMES])
    img = apply_homography(CAM, pts) + rng.normal(0, noise_px, (len(pts), 2))
    on_screen = (img[:, 0] >= 0) & (img[:, 0] < SIZE[0]) & (img[:, 1] >= 0) & (img[:, 1] < SIZE[1])
    conf = np.where(on_screen, 0.9, 0.05)
    return np.c_[img, conf]


def _error_m(calib, pitch_pts):
    return np.linalg.norm(calib.to_pitch(apply_homography(CAM, pitch_pts)) - pitch_pts, axis=1).max()


def test_keypoint_order_is_the_pitch_landmarks():
    assert len(KEYPOINT_NAMES) == 33 and KEYPOINT_NAMES[0] == "corner_top_left"


def test_accurate_calibration_from_noisy_landmarks():
    kp = _predicted(noise_px=2.0)
    assert (kp[:, 2] > 0.5).sum() >= 8                            # several landmarks in view
    result = calibration_from_keypoints(kp, SIZE)
    assert result.ok, result.reason
    assert result.mean_error_m < 0.5
    assert _error_m(result.calib, np.array([[20.0, 30.0], [35.0, 10.0]])) < 0.5


def test_wrong_landmarks_are_ignored():
    kp = _predicted(noise_px=1.0)
    visible = np.flatnonzero(kp[:, 2] > 0.5)
    kp[visible[:2], :2] += [[150, -80], [-120, 200]]              # two confidently wrong predictions
    result = calibration_from_keypoints(kp, SIZE)
    assert result.ok and result.inliers == len(visible) - 2
    assert _error_m(result.calib, np.array([[20.0, 30.0]])) < 0.5


def test_rejections():
    kp = _predicted()
    low = kp.copy()
    low[:, 2] = 0.3                                               # model unsure everywhere
    assert calibration_from_keypoints(low, SIZE).reason.startswith("only 0 confident")

    # six landmarks all on the left goal line: they cover no area at all
    collinear = kp.copy()
    collinear[:, 2] = 0.0
    for name in ("corner_top_left", "left_penalty_area_goalline_top", "left_goal_area_goalline_top",
                 "left_goal_area_goalline_bottom", "left_penalty_area_goalline_bottom", "corner_bottom_left"):
        collinear[KEYPOINT_NAMES.index(name), 2] = 0.9
    result = calibration_from_keypoints(collinear, SIZE)
    assert not result.ok and ("one line" in result.reason or "homography" in result.reason
                              or "agree" in result.reason)


@pytest.mark.parametrize("noise", [1.0, 3.0])
def test_quality_numbers_reported(noise):
    result = calibration_from_keypoints(_predicted(noise_px=noise, seed=2), SIZE)
    assert result.ok and result.used >= result.inliers >= 5 and result.spread_m2 > 150
