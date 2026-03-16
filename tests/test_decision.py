import sys
from pathlib import Path

import numpy as np

from pitchlens.offside.decision import (NO_DECISION, NO_OFFENCE, OFFSIDE, ONSIDE, TOO_CLOSE, decide,
                                        decide_all)
from pitchlens.offside.freeze import freeze_pass
from pitchlens.offside.passes import Pass, Touch

sys.path.insert(0, str(Path(__file__).parent))
from test_freeze import KICK, LATER, SCENE, _frames, _lines, _pass  # noqa: E402


def _decide(p, lines=None):
    frames = _frames()
    return decide(p, freeze_pass(p, frames, lines if lines is not None else _lines(frames)))


def test_receiver_in_offside_position_is_an_offence():
    d = _decide(_pass(receiver=11))
    assert d.verdict == OFFSIDE and d.is_offence
    assert d.receiver_check.margin_m == 2.0
    assert "+2.00 m beyond" in d.reason
    assert d.uninvolved_offside == []


def test_onside_receiver_with_an_offside_teammate_not_involved():
    d = _decide(_pass(receiver=12))        # 12 was onside at the kick (ran offside only afterwards)
    assert d.verdict == ONSIDE and not d.is_offence
    assert d.uninvolved_offside == [11]    # 11 was in an offside position but didn't get the ball


def test_level_receiver_is_too_close_to_call():
    d = _decide(_pass(receiver=13))
    assert d.verdict == TOO_CLOSE and abs(d.receiver_check.margin_m) < 0.3


def test_ball_to_an_opponent_is_no_offence():
    d = _decide(_pass(receiver=21))
    assert d.verdict == NO_OFFENCE
    assert d.uninvolved_offside == [11]


def test_cases_that_cannot_be_judged():
    frames = _frames()
    lines = _lines(frames)
    lines[KICK] = {}
    assert _decide(_pass(), lines).verdict == NO_DECISION                        # no line at the kick

    kick = Touch(KICK, 10, 0, 3, np.array([70.0, 34.0]), 12.0)
    assert _decide(Pass(kick, None)).verdict == NO_DECISION                      # clip ends
    unknown = Touch(LATER, None, None, None, np.array([95.0, 30.0]), 14.0)
    assert _decide(Pass(kick, unknown)).reason.startswith("receiver not identified")
    stranger = Touch(LATER, 99, 0, None, np.array([95.0, 30.0]), 14.0)       # teammate not tracked at kick
    assert _decide(Pass(kick, stranger)).reason == "receiver's position unknown at the kick"


def test_decide_all_keeps_order():
    frames = _frames()
    passes = [_pass(11), _pass(12), _pass(21)]
    frozen = [freeze_pass(p, frames, _lines(frames)) for p in passes]
    assert [d.verdict for d in decide_all(passes, frozen)] == [OFFSIDE, ONSIDE, NO_OFFENCE]
    assert SCENE[11][0] == SCENE[12][0]


def _through_ball(forward_x, seed=0):
    """A passer at x=70 plays the ball to a forward; B defends the right goal (line at 92)."""
    from pitchlens.offside.ball_path import build_ball_path
    from pitchlens.offside.line import offside_line
    from pitchlens.offside.passes import FramePlayers, detect_touches, find_passes
    from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

    fps = 25.0
    ids = np.array([20, 21, 22, 10, 11])
    teams = np.array([TEAM_B, TEAM_B, TEAM_B, TEAM_A, TEAM_A])
    pos = np.array([[104, 34], [92, 20], [90, 50], [70, 34], [forward_x, 34]], float)
    passer, forward = pos[3], pos[4]
    n = int(round(np.linalg.norm(forward - passer) / 15.0 * fps))
    ball = [passer] * 15 + [passer + (forward - passer) * k / n for k in range(1, n + 1)] + [forward] * 15
    rng = np.random.default_rng(seed)
    raw = [b + rng.normal(0, 0.15, 2) for b in ball]                 # noisy detections

    frames = [FramePlayers(pos, teams, ids) for _ in raw]
    path = build_ball_path(raw, fps)
    passes = find_passes(detect_touches(path, frames))
    lines = [{TEAM_B: offside_line(pos, teams, TEAM_B, False, ball=path.positions[t])} for t in range(len(raw))]
    return [decide(p, freeze_pass(p, frames, lines)) for p in passes]


def test_end_to_end_through_ball():
    decisions = _through_ball(forward_x=95.0)
    assert decisions[0].pass_.kick.track_id == 10 and decisions[0].pass_.receive.track_id == 11
    assert decisions[0].verdict == OFFSIDE
    assert abs(decisions[0].receiver_check.margin_m - 3.0) < 0.5
    assert _through_ball(forward_x=88.0)[0].verdict == ONSIDE
