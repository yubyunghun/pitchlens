import cv2
import numpy as np

from pitchlens.calibration.camera_motion import CameraMotionEstimator, CameraMotionTracker
from pitchlens.calibration.homography import apply_homography

W, H = 960, 540


def _pitch_texture(seed=0):
    """Large grass image with mottled texture and some white lines."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H * 2, W * 2)).astype(np.float32), (0, 0), 3)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    img = np.zeros((H * 2, W * 2, 3), np.uint8)
    img[..., 0] = 35 + 20 * noise
    img[..., 1] = 105 + 45 * noise
    img[..., 2] = 40 + 20 * noise
    for x in range(150, W * 2, 400):
        cv2.line(img, (x, 0), (x + 200, H * 2), (235, 235, 235), 3)
    cv2.circle(img, (W, H), 180, (235, 235, 235), 3)
    return img


def _view(world, H_world_to_view):
    return cv2.warpPerspective(world, H_world_to_view, (W, H))


def _max_point_error(H_est, H_true):
    grid = np.stack(np.meshgrid(np.linspace(50, W - 50, 8), np.linspace(50, H - 50, 5)), -1).reshape(-1, 2)
    return np.linalg.norm(apply_homography(H_est, grid) - apply_homography(H_true, grid), axis=1).max()


def _camera(tx, ty, zoom=1.0):
    """World -> view homography for a camera centered at (tx, ty) with some perspective."""
    return np.array([[zoom, 0.0, -tx], [0.0, zoom, -ty], [0.0, 0.0, 1.0]]) @ np.array(
        [[1.0, 0.15, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0002, 1.0]])


def _full_mask():
    return np.full((H, W), 255, np.uint8)


def test_recovers_pan_and_zoom():
    world = _pitch_texture()
    cam1, cam2 = _camera(200, 150), _camera(230, 158, zoom=1.03)   # 30 px pan + 3% zoom
    f1, f2 = _view(world, cam1), _view(world, cam2)
    true = cam2 @ np.linalg.inv(cam1)                                # frame1 px -> frame2 px

    est = CameraMotionEstimator()
    res = est.estimate(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), _full_mask())
    assert res is not None and res.inlier_ratio > 0.8
    assert _max_point_error(res.H, true) < 1.0


def test_ignores_moving_players_via_mask():
    world = _pitch_texture(1)
    cam1, cam2 = _camera(300, 100), _camera(320, 100)
    f1, f2 = _view(world, cam1), _view(world, cam2)
    boxes = []
    for i, x in enumerate(range(80, W - 80, 110)):     # players running against the pan
        y = 150 + 40 * (i % 5)
        cv2.rectangle(f1, (x, y), (x + 25, y + 60), (0, 0, 220), -1)
        cv2.rectangle(f2, (x - 35, y + 10), (x - 10, y + 70), (0, 0, 220), -1)
        boxes += [[x, y, x + 25, y + 60]]

    tracker = CameraMotionTracker()
    assert tracker.update(f1, np.array(boxes, float)) is None        # first frame: nothing yet
    moved = [[x1 - 35, y1 + 10, x2 - 35, y2 + 10] for x1, y1, x2, y2 in boxes]
    res = tracker.update(f2, np.array(moved, float))
    assert res is not None
    assert _max_point_error(res.H, cam2 @ np.linalg.inv(cam1)) < 1.0


def test_static_camera_is_identity():
    f = _view(_pitch_texture(2), _camera(100, 100))
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    res = CameraMotionEstimator().estimate(g, g, _full_mask())
    assert res is not None and _max_point_error(res.H, np.eye(3)) < 0.1


def test_textureless_frame_returns_none():
    g = np.full((H, W), 90, np.uint8)
    assert CameraMotionEstimator().estimate(g, g, _full_mask()) is None
