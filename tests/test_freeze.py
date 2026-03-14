import numpy as np

from pitchlens.offside.freeze import freeze_pass
from pitchlens.offside.line import offside_line
from pitchlens.offside.passes import FramePlayers, Pass, Touch
from pitchlens.offside.position import OFFSIDE, ONSIDE, TOO_CLOSE
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

# A attacks right; B defends the right goal (x = 105).
# track id -> (team, x at the kick, x one second later)
SCENE = {
    20: (TEAM_B, 104.0, 104.0),   # B keeper
    21: (TEAM_B, 92.0, 95.0),     # B second-last defender -> line at 92 at the kick
    22: (TEAM_B, 88.0, 90.0),
    10: (TEAM_A, 70.0, 72.0),     # passer, ball at their feet
    11: (TEAM_A, 94.0, 91.0),     # past the line at the kick, back onside later
    12: (TEAM_A, 85.0, 97.0),     # onside at the kick, runs past the line after it
    13: (TEAM_A, 92.1, 93.0),     # level with the line: too close to call
}
KICK, LATER = 10, 35


def _frames(n=40):
    ids = np.array(list(SCENE))
    teams = np.array([v[0] for v in SCENE.values()])
    out = []
    for t in range(n):
        x = np.array([v[1] if t <= KICK else v[2] for v in SCENE.values()])
        out.append(FramePlayers(np.c_[x, np.linspace(10, 58, len(x))], teams, ids))
    return out


def _lines(frames):
    return [{TEAM_B: offside_line(f.positions, f.teams, TEAM_B, defends_left=False,
                                  ball=np.array([70.0, 34.0]))} for f in frames]


def _pass(receiver=11, kick_team=TEAM_A):
    kick = Touch(KICK, 10, kick_team, 3, np.array([70.0, 34.0]), 12.0)
    receive = Touch(LATER, receiver, SCENE[receiver][0], None, np.array([90.0, 30.0]), 12.0)
    return Pass(kick, receive)


def test_checks_at_the_kick_frame_only():
    frames = _frames()
    frozen = freeze_pass(_pass(), frames, _lines(frames))
    assert frozen.frame == KICK and frozen.attacking_team == TEAM_A
    assert frozen.line.x == 92.0
    status = {tid: c.status for tid, c in frozen.checks.items()}
    assert status == {11: OFFSIDE, 12: ONSIDE, 13: TOO_CLOSE}     # passer 10 not judged
    assert frozen.offside_track_ids() == [11]
    assert frozen.checks[11].margin_m == 2.0


def test_line_unknown_at_the_kick():
    frames = _frames()
    lines = _lines(frames)
    lines[KICK] = {}                                              # too few defenders visible then
    frozen = freeze_pass(_pass(), frames, lines)
    assert frozen.line is None and frozen.checks == {}


def test_passer_team_unknown():
    frames = _frames()
    frozen = freeze_pass(_pass(kick_team=None), frames, _lines(frames))
    assert frozen.line is None
