import sys
from pathlib import Path

import numpy as np

from pitchlens.offside.passes import FramePlayers, Touch
from pitchlens.tactics.formations import classify_formation
from pitchlens.tactics.possession import (ALL, IN, NO_POSSESSION, OUT, phase_frames, phase_shapes,
                                          possession_by_frame)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_average_shape import _layout  # noqa: E402

FPS = 25.0


def _touch(frame, team):
    return Touch(frame, None if team is None else 1, team, None, np.zeros(2), 10.0)


def test_possession_timeline():
    touches = [_touch(10, TEAM_A), _touch(30, TEAM_A), _touch(50, TEAM_B), _touch(70, None), _touch(90, TEAM_A)]
    pos = possession_by_frame(touches, n_frames=400, fps=FPS, max_hold_s=4.0)
    assert (pos[:10] == NO_POSSESSION).all()                # before the first touch
    assert (pos[10:50] == TEAM_A).all()
    assert (pos[50:70] == TEAM_B).all()
    assert (pos[70:90] == NO_POSSESSION).all()              # after an uncredited touch
    assert (pos[90:190] == TEAM_A).all()                    # held for at most 4 s
    assert (pos[190:] == NO_POSSESSION).all()
    phases = phase_frames(pos, TEAM_B)
    assert list(phases[IN]) == list(range(50, 70))
    assert len(phases[OUT]) == 40 + 100 and len(phases[ALL]) == 400


def test_no_touches():
    assert (possession_by_frame([], 50, FPS) == NO_POSSESSION).all()


def test_different_shapes_with_and_without_the_ball():
    """A: 4-3-3 with the ball, drops into a 4-1-4-1 without it. B: 4-4-2 throughout."""
    rng = np.random.default_rng(0)
    a_in, a_out = _layout("4-3-3", (30, 70)), _layout("4-1-4-1", (15, 50))
    b_lay = _layout("4-4-2", (20, 55))
    b = np.c_[105 - b_lay[:, 0], 68 - b_lay[:, 1]]
    n = 300
    touches = [_touch(t, TEAM_A if (t // 50) % 2 == 0 else TEAM_B) for t in range(0, n, 25)]
    possession = possession_by_frame(touches, n, FPS)
    frames = []
    for t in range(n):
        a = (a_in if possession[t] == TEAM_A else a_out) + rng.normal(0, 1.5, (10, 2))
        bb = b + rng.normal(0, 1.5, (10, 2))
        frames.append(FramePlayers(np.vstack([a, bb]), np.array([TEAM_A] * 10 + [TEAM_B] * 10),
                                   np.r_[np.arange(1, 11), np.arange(21, 31)]))
    shapes = phase_shapes(frames, {TEAM_A: 1, TEAM_B: -1}, possession)
    assert classify_formation(shapes[TEAM_A][IN].positions, True).name == "4-3-3"
    assert classify_formation(shapes[TEAM_A][OUT].positions, True).name == "4-1-4-1"
    assert classify_formation(shapes[TEAM_B][IN].positions, False).name == "4-4-2"
    assert classify_formation(shapes[TEAM_B][OUT].positions, False).name == "4-4-2"
    # out of possession A sits deeper: its average is further back
    assert shapes[TEAM_A][OUT].positions[:, 0].mean() < shapes[TEAM_A][IN].positions[:, 0].mean() - 10
