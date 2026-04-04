import csv
import sys
from pathlib import Path

import numpy as np

from pitchlens.offside.passes import FramePlayers, Pass, Touch
from pitchlens.tactics.lane_options import frame_options
from pitchlens.tactics.pass_choice import (analyze_passes, options_before_kick, summarize_passing, team_passing,
                                           write_pass_choices_csv)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_passing_lanes import _attack  # noqa: E402

DIRS = {TEAM_A: 1, TEAM_B: -1}
IDS = np.arange(100, 112)           # player index i has track id 100 + i


def _setup(n=20):
    pos, teams = _attack()           # A's carrier (100) at x=40; best option is 102 between the lines
    frames = [FramePlayers(pos.copy(), teams, IDS) for _ in range(n)]
    ball = np.tile(pos[0] + [0.5, 0.0], (n, 1))
    options = frame_options(frames, ball, np.full(n, TEAM_A), DIRS)
    return options


def _pass(kick, receiver, receiver_team=TEAM_A, passer=100, team=TEAM_A):
    return Pass(Touch(kick, passer, team, 0, np.zeros(2), 12.0),
                Touch(kick + 20, receiver, receiver_team, None, np.zeros(2), 12.0))


def test_best_option_taken_or_not():
    options = _setup()
    choices = analyze_passes([_pass(10, 102), _pass(15, 101), _pass(18, 107, receiver_team=TEAM_B)], options)
    best, safe, lost = choices
    assert best.chose_best and best.forward and best.line_breaking and best.used.status == "open"
    assert best.best_receiver_track_id == 102
    assert safe.chose_best is False and safe.forward is False and safe.line_breaking is False
    assert lost.completed is False and lost.used is None and lost.chose_best is False


def test_options_come_from_just_before_the_kick():
    options = _setup()
    options[12] = None                                        # ball already left the carrier's feet
    assert options_before_kick(_pass(12, 102), options).frame == 11
    other = list(options)
    for t in range(0, 13):
        other[t] = None                                       # no options in the lookback window
    assert options_before_kick(_pass(12, 102), other) is None
    choice = analyze_passes([_pass(12, 102)], other)[0]
    assert choice.options_frame is None and choice.chose_best is None and choice.forward is None


def test_team_profile_and_summary(tmp_path):
    options = _setup()
    # B has the ball in the last 5 frames (mirror the scene, B attacks left)
    pos, teams = _attack()
    flipped = np.where(teams == TEAM_A, TEAM_B, TEAM_A)
    frames_b = [FramePlayers(np.c_[105 - pos[:, 0], 68 - pos[:, 1]], flipped, IDS) for _ in range(5)]
    ball_b = np.tile(np.array([105 - pos[0, 0] - 0.5, 68 - pos[0, 1]]), (5, 1))
    options += frame_options(frames_b, ball_b, np.full(5, TEAM_B), DIRS)
    choices = analyze_passes([_pass(10, 102), _pass(15, 101), _pass(18, 107, receiver_team=TEAM_B)], options)
    stats = {s.team: s for s in team_passing(choices, options, [TEAM_A, TEAM_B])}
    a = stats[TEAM_A]
    assert a.passes == 3 and abs(a.completed - 2 / 3) < 1e-9
    assert a.forward == 0.5 and a.line_breaking == 0.5      # of the 2 passes to visible teammates
    assert abs(a.chose_best - 1 / 3) < 1e-9
    assert a.open_lanes_on_ball >= 1
    assert a.open_forward_lanes_allowed is not None          # B had the ball in 5 frames
    assert stats[TEAM_B].passes == 0
    text = summarize_passing(list(stats.values()), {TEAM_A: "team 0 (red)"})
    assert "team 0 (red): 3 passes:" in text and "team 1: no passes detected" in text

    out = tmp_path / "pass_choices.csv"
    write_pass_choices_csv(out, choices, fps=25.0)
    rows = list(csv.DictReader(out.open()))
    assert [r["chose_best"] for r in rows] == ["True", "False", "False"]
    assert rows[0]["lane_status"] == "open" and rows[0]["bypassed"] == "3"
    assert rows[2]["lane_status"] == "" and rows[2]["completed"] == "False"
