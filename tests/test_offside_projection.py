import sys
from pathlib import Path

import cv2
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


def _scene():
    """Broadcast-like frame of the left third: real markings, red defenders, a blue attacker."""
    from test_drift import _broadcast_view
    frame, cam = _broadcast_view()
    from pitchlens.calibration.homography import PitchCalibration
    calib = PitchCalibration(np.linalg.inv(cam), (1280, 720))
    players = {(40, 40, 210): [(2, 34), (14, 15), (11.5, 28), (12, 40), (20, 52)],   # red, defends left
               (210, 90, 30): [(10, 24), (25, 40)]}                                  # blue attackers
    boxes, pos, teams = [], [], []
    for team, (color, pts) in enumerate(players.items()):
        for p in pts:
            fx, fy = calib.to_image(np.array([p], float))[0]
            hgt = 0.06 * fy + 20
            box = [fx - hgt * 0.2, fy - hgt, fx + hgt * 0.2, fy]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, -1)
            boxes.append(box), pos.append(p), teams.append(team)
    return frame, calib, np.array(boxes), np.array(pos, float), np.array(teams)



def test_overlay_draws_band_behind_players():
    from pitchlens.offside.line import offside_line
    from pitchlens.visualization.offside_overlay import draw_offside_overlay
    frame, calib, boxes, pos, teams = _scene()
    line = offside_line(pos, teams, 0, True)
    assert line.x == 11.5 and line.defender_index == 2
    color = (40, 40, 210)
    out = draw_offside_overlay(frame.copy(), calib, line, color, occluders=boxes,
                               defender_foot=calib.to_image(pos[2:3])[0])
    changed = (out != frame).any(axis=2)
    # the line is drawn on the pitch, far from players...
    mid = calib.to_image(np.array([[11.5, 60.0]]))[0].round().astype(int)
    assert changed[mid[1] - 3:mid[1] + 4, mid[0] - 3:mid[0] + 4].any()
    # ...but not over the players' boxes (players stay in front), except the defender's ring at the feet
    for i, (x1, y1, x2, y2) in enumerate(boxes.round().astype(int)):
        upper = changed[y1:(y1 + y2) // 2, x1:x2]
        assert not upper.any(), f"overlay drawn over player {i}"


def test_active_team_is_the_one_defending_near_the_ball():
    from pitchlens.offside.direction import ATTACKS_LEFT, ATTACKS_RIGHT, DirectionEstimate
    from pitchlens.offside.line import active_defending_team, frame_offside_lines
    _, calib, _, pos, teams = _scene()
    lines = {0: frame_offside_lines(pos, teams, DirectionEstimate({0: ATTACKS_RIGHT, 1: ATTACKS_LEFT}, 1, 1),
                                    calib)[0]}
    lines[1] = lines[0].__class__(1, 80.0, 25.0, "defender", None, True, False)
    assert active_defending_team(lines, np.array([20.0, 30.0]), pos) == 0     # ball near the left goal
    assert active_defending_team(lines, np.array([90.0, 30.0]), pos) == 1     # ball near the right goal
    assert active_defending_team(lines, None, pos) == 0                       # players are on the left
    assert active_defending_team({}, None, pos) is None
