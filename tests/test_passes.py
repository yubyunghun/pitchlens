import numpy as np

from pitchlens.offside.ball_path import build_ball_path
from pitchlens.offside.passes import FramePlayers, detect_touches, find_passes, group_possessions
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

FPS = 25.0
# track id -> (team, position)
PLAYERS = {1: (TEAM_A, (30.0, 30.0)), 2: (TEAM_A, (50.0, 30.0)), 3: (TEAM_A, (60.0, 45.0)),
           7: (TEAM_B, (55.0, 20.0)), 8: (TEAM_B, (40.0, 50.0))}


def _players(n):
    ids = np.array(list(PLAYERS))
    pos = np.array([p for _, p in PLAYERS.values()])
    teams = np.array([t for t, _ in PLAYERS.values()])
    return [FramePlayers(pos.copy(), teams, ids) for _ in range(n)]


def _ball(legs, rest_frames=15):
    """legs: [(from_id, to_id, speed m/s)]; ball rests at each receiver for rest_frames."""
    pts = []
    for src, dst, speed in legs:
        a, b = np.array(PLAYERS[src][1]), np.array(PLAYERS[dst][1])
        if not pts:
            pts += [a] * rest_frames
        n = max(2, int(round(np.linalg.norm(b - a) / speed * FPS)))
        pts += [a + (b - a) * (k / n) for k in range(1, n + 1)]
        pts += [b] * rest_frames
    return pts


def test_pass_sequence():
    raw = _ball([(1, 2, 15.0), (2, 3, 12.0), (3, 7, 15.0)])
    path = build_ball_path(raw, FPS)
    players = _players(len(raw))
    touches = detect_touches(path, players)
    ids = [t.track_id for t in touches]
    # 1 kicks, 2 controls + kicks, 3 controls + kicks, 7 receives
    assert ids == [1, 2, 2, 3, 3, 7]
    poss = group_possessions(touches)
    assert [p.track_id for p in poss] == [1, 2, 3, 7]
    passes = find_passes(touches)
    assert [(p.kick.track_id, p.receive.track_id if p.receive else None) for p in passes] == \
        [(1, 2), (2, 3), (3, 7), (7, None)]
    assert [p.completed for p in passes[:3]] == [True, True, False]   # last one intercepted by B
    # the kick moment is when the ball leaves the passer
    assert abs(passes[0].kick.frame - 15) <= 2


def test_missed_ball_frames_and_jitter():
    rng = np.random.default_rng(1)
    raw = [p + rng.normal(0, 0.05, 2) for p in _ball([(1, 2, 15.0), (2, 3, 12.0)])]
    for t in list(range(20, 24)) + list(range(50, 53)):     # detector misses the ball here and there
        raw[t] = None
    touches = detect_touches(build_ball_path(raw, FPS), _players(len(raw)))
    assert [p.track_id for p in group_possessions(touches)] == [1, 2, 3]


def test_touch_with_nobody_near():
    raw = _ball([(1, 2, 15.0)])
    players = _players(len(raw))
    for fp in players:                                       # receiver not tracked (off-screen)
        fp.positions[1] = np.nan
    touches = detect_touches(build_ball_path(raw, FPS), players)
    assert touches[0].track_id == 1 and touches[-1].track_id is None
    passes = find_passes(touches)
    assert len(passes) == 1 and passes[0].receive.track_id is None and not passes[0].completed


def test_ball_at_rest_has_no_touches():
    raw = [np.array([30.0, 30.0])] * 40
    assert detect_touches(build_ball_path(raw, FPS), _players(40)) == []


def test_noisy_detections_still_give_the_right_passes():
    """0.2 m jitter (typical broadcast ball detection) and 20% missed frames."""
    exact = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        raw = [p + rng.normal(0, 0.2, 2) for p in _ball([(1, 2, 15.0), (2, 3, 12.0), (3, 7, 15.0)])]
        raw = [None if rng.random() < 0.2 else p for p in raw]
        touches = detect_touches(build_ball_path(raw, FPS), _players(len(raw)))
        exact += [p.track_id for p in group_possessions(touches)] == [1, 2, 3, 7]
    assert exact >= 8
