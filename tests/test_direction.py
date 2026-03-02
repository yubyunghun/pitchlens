import csv

import numpy as np

from pitchlens.offside.direction import (ATTACKS_LEFT, ATTACKS_RIGHT, frames_from_tracks_csv,
                                         infer_attack_directions)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B, UNKNOWN

# 4-4-2 for a team defending the left goal, in pitch meters (goalkeeper first)
SHAPE_442 = np.array([[5, 34], [20, 10], [18, 27], [18, 41], [20, 58],
                      [35, 12], [33, 28], [33, 40], [35, 56], [47, 28], [47, 40]], float)
PITCH_LENGTH = 105.0


def _mirror(shape):
    return np.c_[PITCH_LENGTH - shape[:, 0], shape[:, 1]]


def _frame(a_shape, b_shape, rng, visible=None):
    xy = np.vstack([a_shape, b_shape]) + rng.normal(0, 1.5, (len(a_shape) + len(b_shape), 2))
    teams = np.array([TEAM_A] * len(a_shape) + [TEAM_B] * len(b_shape))
    if visible is not None:                         # camera shows only part of the pitch
        xy[(xy[:, 0] < visible[0]) | (xy[:, 0] > visible[1])] = np.nan
    return xy, teams


def _phases(a_defends_left, rng, n=60):
    """Mix of game phases: both teams in shape, A pressing high, B pressing high."""
    a = SHAPE_442 if a_defends_left else _mirror(SHAPE_442)
    b = _mirror(SHAPE_442) if a_defends_left else SHAPE_442
    toward_b_goal = 1 if a_defends_left else -1
    frames = []
    for t in range(n):
        shift = [0, 25, -25][t % 3] * toward_b_goal    # whole game moves up/down the pitch
        frames.append(_frame(a + [shift, 0], b + [shift, 0], rng))
    return frames


def test_detects_both_orientations():
    rng = np.random.default_rng(0)
    est = infer_attack_directions(_phases(True, rng))
    assert est.directions == {TEAM_A: ATTACKS_RIGHT, TEAM_B: ATTACKS_LEFT}
    assert est.confidence > 0.9 and est.frames_used == 60
    assert est.describe(TEAM_A) == "left -> right"

    est = infer_attack_directions(_phases(False, rng))
    assert est.directions == {TEAM_A: ATTACKS_LEFT, TEAM_B: ATTACKS_RIGHT}


def test_sustained_attack_in_one_half():
    """B camps in A's half: B's attackers are deep, yet A's defenders are still closer to A's goal."""
    rng = np.random.default_rng(1)
    a_def = np.array([[4, 34], [9, 20], [8, 30], [8, 40], [9, 50], [16, 25], [16, 43], [22, 34]], float)
    b_att = np.array([[12, 18], [11, 33], [12, 48], [20, 30], [24, 40], [30, 20], [35, 50], [60, 34]], float)
    frames = [_frame(a_def, b_att, rng, visible=(0, 50)) for _ in range(30)]
    est = infer_attack_directions(frames)
    assert est.directions[TEAM_A] == ATTACKS_RIGHT


def test_partial_view_and_noise():
    rng = np.random.default_rng(2)
    frames = [_frame(SHAPE_442 + [s, 0], _mirror(SHAPE_442) + [s, 0], rng, visible=(25 + s, 75 + s))
              for s in rng.uniform(-15, 15, 80)]
    # a few garbage frames (e.g. wrong team colors for a moment)
    frames += [_frame(_mirror(SHAPE_442), SHAPE_442, rng) for _ in range(8)]
    est = infer_attack_directions(frames)
    assert est.directions[TEAM_A] == ATTACKS_RIGHT
    assert 0.6 < est.confidence < 1.0


def test_not_enough_players():
    rng = np.random.default_rng(3)
    xy, teams = _frame(SHAPE_442[:2], _mirror(SHAPE_442)[:2], rng)
    assert infer_attack_directions([(xy, teams)]) is None
    assert infer_attack_directions([]) is None


def test_reads_tracks_csv(tmp_path):
    path = tmp_path / "tracks.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class", "team", "x_px", "y_px", "x_m", "y_m", "calib_distance"])
        for frame in range(3):
            for i, (x, y) in enumerate(SHAPE_442[:5]):
                w.writerow([frame, i, "player", TEAM_A, 0, 0, x, y, 0])
            for i, (x, y) in enumerate(_mirror(SHAPE_442)[:5]):
                w.writerow([frame, 20 + i, "player", TEAM_B, 0, 0, x, y, 0])
            w.writerow([frame, 99, "referee", UNKNOWN, 0, 0, 50, 30, 0])
            w.writerow([frame, 98, "player", TEAM_A, 0, 0, "", "", ""])   # uncalibrated row
    frames = frames_from_tracks_csv(path)
    assert len(frames) == 3 and len(frames[0][0]) == 11
    assert infer_attack_directions(frames).directions[TEAM_A] == ATTACKS_RIGHT


def test_minimap_arrows_point_the_right_way():
    from pitchlens.calibration.pitch import Pitch
    from pitchlens.visualization.minimap import Minimap, draw_attack_arrows
    mm = Minimap(Pitch(), width_px=555, margin_m=3.0)
    red, blue = (0, 0, 255), (255, 0, 0)
    img = draw_attack_arrows(mm.render(np.empty((0, 2)), []), mm, [(red, ATTACKS_RIGHT), (blue, ATTACKS_LEFT)])
    top = img[: int(mm.margin * mm.scale)]

    def tip_x(color):   # arrow tip is the far end of the colored pixels in the arrow's direction
        xs = np.where((top == color).all(axis=2))[1]
        return xs.min(), xs.max()

    r_min, r_max = tip_x(red)
    b_min, b_max = tip_x(blue)
    assert r_max < b_min                         # red arrow on the left, blue on the right
    # the arrowhead is wider than the shaft, so most colored pixels sit near the tip
    red_cols = np.where((top == red).all(axis=2))[1]
    blue_cols = np.where((top == blue).all(axis=2))[1]
    assert np.median(red_cols) > (r_min + r_max) / 2     # red points right
    assert np.median(blue_cols) < (b_min + b_max) / 2    # blue points left
