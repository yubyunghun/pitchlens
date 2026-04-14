import sys
import types
from pathlib import Path

import numpy as np

from pitchlens.calibration.auto_calib import KEYPOINT_NAMES
from pitchlens.calibration.homography import apply_homography
from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.keypoint_model import keyframes_from_predictions
from pitchlens.calibration.pitch import Pitch

sys.path.insert(0, str(Path(__file__).parent))
import test_pipeline as tp  # noqa: E402
from test_auto_calib import SIZE, _predicted  # noqa: E402

LANDMARKS = np.array([Pitch().keypoints()[n] for n in KEYPOINT_NAMES])


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakePoseYOLO:
    """Mimics ultralytics results: r.keypoints.data (n, 33, 3), r.boxes.conf (n,)."""
    outputs: list = []

    def __init__(self, path):
        self.path = path

    def __call__(self, frames, **kwargs):
        res = []
        for _ in frames:
            kp = FakePoseYOLO.outputs.pop(0)
            if kp is None:
                res.append(types.SimpleNamespace(keypoints=None, boxes=None))
            else:
                data = np.stack([kp * 0.5, kp])                     # a weak and a strong candidate
                res.append(types.SimpleNamespace(keypoints=types.SimpleNamespace(data=_Arr(data)),
                                                 boxes=types.SimpleNamespace(conf=_Arr([0.3, 0.8]))))
        return res


def test_keypoint_detector_picks_the_most_confident_pitch(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakePoseYOLO))
    from pitchlens.calibration.keypoint_model import KeypointDetector
    det = KeypointDetector("pitch.pt")
    kp = _predicted(noise_px=1.0)
    FakePoseYOLO.outputs = [kp, None]
    out = det._run([np.zeros((SIZE[1], SIZE[0], 3), np.uint8)] * 2)
    assert np.allclose(out[0], kp) and out[1] is None


def test_predictions_become_keyframes():
    good = _predicted(noise_px=1.0)
    unsure = good.copy()
    unsure[:, 2] = 0.2
    keys, rejected = keyframes_from_predictions({0: good, 25: unsure, 50: None, 75: np.zeros((0, 3))}, SIZE)
    assert keys.frames == [0]
    assert rejected[25].startswith("only 0 confident") and rejected[50] == "no pitch found"
    assert rejected[75].startswith("only 0 confident")


def test_manual_keyframes_win_when_merged():
    a, b = _predicted(), _predicted(noise_px=3.0)
    manual = CalibrationSet({10: keyframes_from_predictions({0: a}, SIZE)[0].keyframes[0]})
    auto = CalibrationSet({10: keyframes_from_predictions({0: b}, SIZE)[0].keyframes[0],
                           40: keyframes_from_predictions({0: b}, SIZE)[0].keyframes[0]})
    merged = manual.merged_with(auto)
    assert merged.frames == [10, 40] and merged.keyframes[10] is manual.keyframes[10]
    assert manual.merged_with(None).frames == [10]


class FakeLandmarkModel:
    """Knows the synthetic clip's true camera: returns noisy landmarks visible in each frame it is shown.

    It is called on frames 0, every, 2*every, ... so its n-th call is frame n * every.
    """

    def __init__(self, every):
        self.every, self.calls = every, 0
        self.rng = np.random.default_rng(0)

    def predict(self, frame):
        t = self.calls * self.every
        self.calls += 1
        h, w = frame.shape[:2]
        cam_px = tp._cam(t) @ np.diag([tp.PX_PER_M, tp.PX_PER_M, 1.0])        # pitch meters -> pixels
        img = apply_homography(cam_px, LANDMARKS) + self.rng.normal(0, 1.5, (len(LANDMARKS), 2))
        vis = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
        return np.c_[img, np.where(vis, 0.9, 0.05)]


def test_pipeline_calibrates_a_clip_without_any_clicking(tmp_path):
    import supervision as sv
    from pitchlens.pipeline import analyze, auto_keyframes, solve
    from pitchlens.tracking.team_assigner import TeamAssigner

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    assigner.collect(first, tp._boxes(0))
    assigner.collect(first, tp._boxes(0))
    assigner.fit()

    records = analyze(str(video), tp.FakeDetector(), assigner, fps=25,
                      keypoint_detector=FakeLandmarkModel(every=10), keypoint_every=10)
    assert [t for t, r in enumerate(records) if r.pitch_keypoints is not None] == [0, 10, 20, 30]
    keys, rejected = auto_keyframes(records, tp.SIZE)
    assert keys.frames == [0, 10, 20, 30], rejected         # no manual keyframes at all
    calibs = solve(records, keys, tp.SIZE)
    on_screen = np.array([[150, 112], [480, 270], [750, 375]], float)
    for t in (5, 25, tp.N_FRAMES - 1):
        true = tp._calib_for(tp._cam(t))
        err = np.linalg.norm(calibs[t].calib.to_pitch(on_screen) - true.to_pitch(on_screen), axis=1)
        assert err.max() < 0.5, f"frame {t}: {err.max():.2f} m"
