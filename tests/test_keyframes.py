import json

import numpy as np
import pytest

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.tool import parse_time

PITCH_PTS = np.array([[0, 0], [52.5, 0], [52.5, 68], [16.5, 54.16], [16.5, 13.84]], dtype=float)


def _calib(shift=0.0):
    img = PITCH_PTS * 12 + [200 + shift, 150]
    return PitchCalibration.from_correspondences(img, PITCH_PTS, (1920, 1080))


def test_add_replace_remove_and_nearest():
    s = CalibrationSet()
    s.add(250, _calib(10))
    s.add(0, _calib())
    s.add(900, _calib(20))
    assert s.frames == [0, 250, 900]                   # kept sorted
    s.add(250, _calib(30))                             # same frame replaces
    assert len(s) == 3
    assert s.nearest(100) == 0 and s.nearest(200) == 250 and s.nearest(5000) == 900
    assert s.nearest(125) == 0                         # tie -> earlier keyframe
    s.remove(250)
    assert s.frames == [0, 900]
    assert CalibrationSet().nearest(10) is None


def test_save_load_roundtrip(tmp_path):
    s = CalibrationSet({0: _calib(), 400: _calib(15)}, video="clip.mp4")
    path = tmp_path / "calibration.json"
    s.save(path)
    loaded = CalibrationSet.load(path)
    assert loaded.frames == [0, 400] and loaded.video == "clip.mp4"
    assert np.allclose(loaded.keyframes[400].H, s.keyframes[400].H)
    assert json.loads(path.read_text())["version"] == 2


def test_loads_old_single_calibration_file(tmp_path):
    path = tmp_path / "old.json"
    _calib().save(path, video="clip.mp4", frame=75)    # format written before keyframes existed
    loaded = CalibrationSet.load(path)
    assert loaded.frames == [75] and loaded.video == "clip.mp4"


def test_for_image_size_scales_every_keyframe():
    s = CalibrationSet({0: _calib(), 10: _calib(5)}).for_image_size(960, 540)
    assert all(c.image_size == (960, 540) for c in s.keyframes.values())


@pytest.mark.parametrize("text, seconds", [("83", 83), ("83.5", 83.5), ("1:23", 83), ("0:01:23", 83), ("2:00.5", 120.5)])
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)
