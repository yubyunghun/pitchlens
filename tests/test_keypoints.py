import sys
from pathlib import Path

import numpy as np
import yaml

from pitchlens.calibration.auto_calib import KEYPOINT_NAMES, calibration_from_keypoints
from pitchlens.calibration.homography import PitchCalibration, apply_homography
from pitchlens.calibration.pitch import Pitch
from pitchlens.training.dataset import inspect_dataset
from pitchlens.training.keypoints import export_keypoint_dataset, flip_index, keypoint_label
from pitchlens.training.train import TrainSettings, train

sys.path.insert(0, str(Path(__file__).parent))
from test_auto_calib import CAM, SIZE  # noqa: E402
from test_train import FakeYOLO  # noqa: E402


def _calib():
    return PitchCalibration(np.linalg.inv(CAM), SIZE)


def test_flip_index_swaps_left_and_right():
    flip = flip_index()
    idx = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    assert flip[idx["corner_top_left"]] == idx["corner_top_right"]
    assert flip[idx["left_penalty_spot"]] == idx["right_penalty_spot"]
    assert flip[idx["center_spot"]] == idx["center_spot"]
    assert flip[idx["center_circle_left"]] == idx["center_circle_right"]
    assert [flip[f] for f in flip] == list(range(len(flip)))   # flipping twice is the identity


def test_label_matches_the_true_landmark_positions():
    label = keypoint_label(_calib(), SIZE)
    values = [float(v) for v in label.split()]
    assert len(values) == 5 + 33 * 3 and values[0] == 0
    kp = np.array(values[5:]).reshape(33, 3)
    truth = apply_homography(CAM, np.array([Pitch().keypoints()[n] for n in KEYPOINT_NAMES]))
    visible = kp[:, 2] == 2
    assert visible.sum() >= 8 and set(np.unique(kp[:, 2])) <= {0.0, 2.0}
    assert np.allclose(kp[visible, :2] * SIZE, truth[visible], atol=0.01)
    assert (kp[~visible, :2] == 0).all()
    # a model that predicted exactly these labels would calibrate the frame
    pred = np.c_[kp[:, :2] * SIZE, np.where(visible, 0.9, 0.0)]
    assert calibration_from_keypoints(pred, SIZE).ok


def test_too_few_landmarks_in_view():
    # close-up of open grass: 60 px per meter, showing x 64-85 m, y 40-52 m (no landmarks there)
    zoomed = PitchCalibration(np.linalg.inv(np.diag([60.0, 60.0, 1.0]) @ np.array(
        [[1, 0, -64], [0, 1, -40], [0, 0, 1]], float)), SIZE)
    assert keypoint_label(zoomed, SIZE) is None


def test_export_then_check_then_train(tmp_path):
    n = 60
    frames = (np.zeros((SIZE[1], SIZE[0], 3), np.uint8) for _ in range(n))     # a generator, like a video
    calibs = [_calib() if t < 50 else None for t in range(n)]
    counts = export_keypoint_dataset(frames, calibs, tmp_path / "kp", every=5, val_every=5)
    assert counts == {"train": 8, "valid": 2, "skipped": 0}           # frames 0, 5, ..., 45
    data = yaml.safe_load((tmp_path / "kp" / "data.yaml").read_text())
    assert data["kpt_shape"] == [33, 3] and data["names"] == {0: "pitch"} and len(data["flip_idx"]) == 33

    report = inspect_dataset(tmp_path / "kp")
    assert report.is_pose and report.problems == []
    assert report.splits["train"].images == 8 and report.splits["val"].images == 2

    FakeYOLO.calls.clear()
    settings = TrainSettings(task="pose", project=str(tmp_path / "runs"), output=str(tmp_path / "kp.pt"))
    assert train(tmp_path / "kp", settings, yolo_cls=FakeYOLO) is not None
    weights, kwargs = FakeYOLO.calls[0]
    assert weights == "yolov8m-pose.pt" and kwargs["imgsz"] == 640
    # wrong task for the dataset is refused
    assert train(tmp_path / "kp", TrainSettings(task="detect", project=str(tmp_path / "r2")),
                 force=True, yolo_cls=FakeYOLO) is None


def test_calibrations_for_a_panning_video(tmp_path):
    """One keyframe calibrates the whole synthetic clip through camera tracking."""
    import test_pipeline as tp
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.training.keypoints import calibrations_for_video

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    CalibrationSet({0: tp._calib_for(tp._cam(0))}).save(tmp_path / "calibration.json")
    calibs = calibrations_for_video(str(video), str(tmp_path / "calibration.json"))
    assert len(calibs) == tp.N_FRAMES and sum(c is not None for c in calibs) >= tp.N_FRAMES - 2
    on_screen = np.array([[150, 112], [480, 270], [750, 375], [360, 338]], float)   # pixels in the 960x540 view
    for t in (10, tp.N_FRAMES - 1):
        true = tp._calib_for(tp._cam(t))
        err = np.linalg.norm(calibs[t].to_pitch(on_screen) - true.to_pitch(on_screen), axis=1)
        assert err.max() < 0.3, f"frame {t}: {err.max():.2f} m"
