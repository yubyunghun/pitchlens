import cv2
import numpy as np

from pitchlens.calibration.drift import line_alignment
from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.pitch import Pitch
from pitchlens.visualization.minimap import Minimap

SIZE = (1280, 720)


def _broadcast_view():
    """Pitch markings seen from a broadcast-style camera, plus the true camera (meters -> px)."""
    cam = cv2.getPerspectiveTransform(np.float32([[0, 0], [60, 0], [60, 68], [0, 68]]),
                                      np.float32([[130, 200], [1150, 200], [1400, 700], [-120, 700]]))
    mm = Minimap(Pitch(), width_px=20 * 111, margin_m=3)             # top-down pitch at 20 px/m
    to_top = np.array([[mm.scale, 0, 3 * mm.scale], [0, mm.scale, 3 * mm.scale], [0, 0, 1]])
    frame = cv2.warpPerspective(mm.render(np.empty((0, 2)), []), cam @ np.linalg.inv(to_top), SIZE,
                                borderValue=(90, 80, 100))
    return frame, cam


def _shifted(cam, dx_m, dy_m=0.0):
    shift = np.array([[1, 0, dx_m], [0, 1, dy_m], [0, 0, 1]], float)
    return PitchCalibration(shift @ np.linalg.inv(cam), SIZE)


def test_correct_calibration_scores_high_and_drift_scores_low():
    frame, cam = _broadcast_view()
    good = line_alignment(frame, _shifted(cam, 0))
    assert good > 0.85
    assert line_alignment(frame, _shifted(cam, 1.0)) < good - 0.3
    assert line_alignment(frame, _shifted(cam, 3.0, 2.0)) < 0.3


def test_score_drops_steadily_with_drift():
    frame, cam = _broadcast_view()
    scores = [line_alignment(frame, _shifted(cam, d)) for d in (0, 0.25, 0.5, 1.0, 2.0)]
    assert all(a >= b for a, b in zip(scores, scores[1:]))


def test_not_enough_pitch_visible():
    frame = np.full((720, 1280, 3), (90, 80, 100), np.uint8)       # stands only
    _, cam = _broadcast_view()
    assert line_alignment(frame, _shifted(cam, 0)) is None


def test_summary_text():
    from pitchlens.pipeline import summarize_alignment
    assert "no frames could be checked" in summarize_alignment([None, None], 25)
    text = summarize_alignment([0.9, None, 0.95, 0.3, 0.92], 25)
    assert "median 91%" in text and "1 frames below 50%" in text and "frame 3" in text
    assert "below" not in summarize_alignment([0.9, 0.8], 25)
