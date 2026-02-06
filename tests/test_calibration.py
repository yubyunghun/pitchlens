import numpy as np
import pytest

from pitchlens.calibration.homography import PitchCalibration, apply_homography
from pitchlens.calibration.pitch import Pitch
from pitchlens.visualization.minimap import Minimap

# A plausible broadcast camera: pitch meters -> 1920x1080 pixels, with perspective.
TRUE_PITCH_TO_IMAGE = np.array([
    [14.0, -6.0, 300.0],
    [0.4, 5.0, 250.0],
    [0.0, -0.004, 1.0],
])


def _image_points(pitch_pts):
    return apply_homography(TRUE_PITCH_TO_IMAGE, pitch_pts)


def test_keypoints_inside_pitch_and_symmetric():
    p = Pitch()
    kp = p.keypoints()
    pts = np.array(list(kp.values()))
    assert (pts >= 0).all() and (pts[:, 0] <= p.length).all() and (pts[:, 1] <= p.width).all()
    for name, (x, y) in kp.items():
        if name.startswith("left_"):
            rx, ry = kp["right_" + name[5:]]
            assert rx == pytest.approx(p.length - x) and ry == pytest.approx(y)
    # penalty arc meets the box edge 9.15 m from the spot
    assert np.hypot(*(np.array(kp["left_arc_top"]) - kp["left_penalty_spot"])) == pytest.approx(9.15)


def test_calibration_recovers_positions():
    kp = Pitch().keypoints()
    names = ["left_penalty_area_corner_top", "left_penalty_area_corner_bottom",
             "halfway_top", "halfway_bottom", "center_spot", "left_goal_area_corner_top"]
    pitch_pts = np.array([kp[n] for n in names])
    rng = np.random.default_rng(0)
    clicks = _image_points(pitch_pts) + rng.normal(0, 1.5, (len(names), 2))  # ~1.5 px click noise

    calib = PitchCalibration.from_correspondences(clicks, pitch_pts, (1920, 1080), names=names)
    assert calib.point_errors().max() < 0.5

    # A player standing somewhere not used for calibration
    player = np.array([[30.0, 40.0]])
    assert np.linalg.norm(calib.to_pitch(_image_points(player)) - player) < 0.5
    assert np.allclose(calib.to_pitch(calib.to_image(player)), player, atol=1e-6)


def test_needs_four_points():
    with pytest.raises(ValueError):
        PitchCalibration.from_correspondences(np.zeros((3, 2)), np.zeros((3, 2)), (1920, 1080))


def test_resolution_change_and_save_load(tmp_path):
    pitch_pts = np.array([[0, 0], [52.5, 0], [52.5, 68], [16.5, 54.16], [16.5, 13.84]], dtype=float)
    calib = PitchCalibration.from_correspondences(_image_points(pitch_pts), pitch_pts, (1920, 1080))

    path = tmp_path / "calib.json"
    calib.save(path, video="x.mp4")
    loaded = PitchCalibration.load(path)
    assert np.allclose(loaded.H, calib.H)

    half = loaded.for_image_size(960, 540)
    img_full = _image_points(np.array([[40.0, 20.0]]))
    assert np.allclose(half.to_pitch(img_full / 2), [[40.0, 20.0]], atol=1e-6)
    assert half.point_errors().max() < 1e-3  # sub-millimeter


def test_minimap_coordinates():
    mm = Minimap(Pitch(), width_px=555, margin_m=3.0)  # 5 px per meter
    assert mm.scale == pytest.approx(5.0)
    assert tuple(mm.to_px((0, 0))) == (15, 15)
    assert np.allclose(mm.from_px(mm.to_px((52.5, 34))), (52.5, 34), atol=0.2)
    img = mm.render(np.array([[52.5, 34.0], [np.nan, np.nan], [500.0, 0.0]]),
                    [(0, 0, 255)] * 3)
    cx, cy = mm.to_px((52.5, 34))
    assert tuple(img[cy, cx]) == (0, 0, 255)          # red dot at center
    assert img.shape == (mm.height, mm.width, 3)
