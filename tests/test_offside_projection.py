import sys
from pathlib import Path

import numpy as np

from pitchlens.offside.projection import project_band, project_pitch_line

sys.path.insert(0, str(Path(__file__).parent))
from test_offside_line import _calib  # noqa: E402  (camera on the left third, 1280x720)


def test_projected_line_lies_on_the_pitch_line():
    calib = _calib(True)
    seg = project_pitch_line(calib, 20.0)
    assert seg is not None
    p1, p2 = seg
    for t in np.linspace(0, 1, 7):                    # every point of the segment maps back to x = 20
        x, y = calib.to_pitch((p1 + t * (p2 - p1))[None])[0]
        assert abs(x - 20.0) < 0.1 and -0.5 <= y <= 68.5
    w, h = calib.image_size
    for p in (p1, p2):
        assert 0 <= p[0] <= w and 0 <= p[1] <= h


def test_touchline_to_touchline_when_fully_visible():
    calib = _calib(True)
    p1, p2 = project_pitch_line(calib, 20.0)
    ys = sorted(calib.to_pitch(np.array([p1, p2]))[:, 1])
    assert ys[0] < 0.5 and ys[1] > 67.5                 # spans the whole width


def test_line_partly_out_of_frame_is_clipped():
    calib = _calib(True)
    p1, p2 = project_pitch_line(calib, 0.5)            # bottom end at image x = -52: off-screen
    w, h = calib.image_size
    assert all(0 <= p[0] <= w and 0 <= p[1] <= h for p in (p1, p2))
    ys = sorted(calib.to_pitch(np.array([p1, p2]))[:, 1])
    assert ys[1] - ys[0] < 68.0 - 1.0                  # only part of the width is visible


def test_line_out_of_view():
    assert project_pitch_line(_calib(True), 90.0) is None      # camera shows x 0-40


def test_band_polygon():
    calib = _calib(True)
    band = project_band(calib, 20.0, 0.5)
    assert band.shape == (4, 2)
    xs = calib.to_pitch(band)[:, 0]
    assert np.allclose(sorted(xs), [19.5, 19.5, 20.5, 20.5], atol=1e-6)

