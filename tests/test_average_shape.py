import numpy as np

from pitchlens.detection.classes import GOALKEEPER, PLAYER
from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.average_shape import (PlayerAverage, collect_averages, find_goalkeeper,
                                             merge_fragments, team_shapes)
from pitchlens.tactics.formations import TEMPLATES, classify_formation
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

DIRS = {TEAM_A: 1, TEAM_B: -1}


def _layout(name, depth, width=(6.0, 62.0)):
    return np.array([[depth[0] + s.depth * (depth[1] - depth[0]), width[0] + s.width * (width[1] - width[0])]
                     for s in TEMPLATES[name]])


def _match(n=200, seed=0, fragment=False, keeper=True, gk_class=False):
    """A plays 4-3-3 attacking right, B a 4-4-2 attacking left; the game moves up and down."""
    rng = np.random.default_rng(seed)
    a_home = _layout("4-3-3", (22, 58))
    b_layout = _layout("4-4-2", (22, 58))
    b_home = np.c_[105 - b_layout[:, 0], 68 - b_layout[:, 1]]       # same shape, attacking left
    frames, classes = [], []
    for t in range(n):
        shift = 12 * np.sin(t / 30)                          # both teams move up/down the pitch together
        a = a_home + [shift, 0] + rng.normal(0, 1.5, a_home.shape)
        b = b_home + [shift, 0] + rng.normal(0, 1.5, b_home.shape)
        pos, teams, ids, cls = [a, b], [TEAM_A] * 10 + [TEAM_B] * 10, list(range(1, 11)) + list(range(21, 31)), \
            [PLAYER] * 20
        if keeper:
            pos += [np.array([[4 + shift / 4, 34]]), np.array([[101 + shift / 4, 34]])]
            teams += [TEAM_A, TEAM_B]
            ids += [0, 20]
            cls += [GOALKEEPER if gk_class else PLAYER] * 2
        ids = np.array(ids)
        if fragment and t >= n // 2:
            ids[ids == 3] = 103                             # player 3's track is lost and re-found
        visible = rng.random(len(ids)) > 0.15               # some players off-screen in each frame
        p = np.vstack(pos)
        p[~visible] = np.nan
        frames.append(FramePlayers(p, np.array(teams), ids))
        classes.append(np.array(cls))
    return frames, classes


def test_average_positions_give_the_right_formations():
    frames, classes = _match()
    shapes = team_shapes(frames, DIRS, classes)
    assert len(shapes[TEAM_A].outfield) == 10 and len(shapes[TEAM_B].outfield) == 10
    assert shapes[TEAM_A].goalkeeper.track_id == 0 and shapes[TEAM_B].goalkeeper.track_id == 20
    assert classify_formation(shapes[TEAM_A].positions, attacks_right=True).name == "4-3-3"
    assert classify_formation(shapes[TEAM_B].positions, attacks_right=False).name == "4-4-2"


def test_track_fragments_are_merged():
    frames, classes = _match(fragment=True)
    raw = collect_averages(frames, classes)
    assert 3 in raw and 103 in raw and not (raw[3].frames & raw[103].frames)
    shapes = team_shapes(frames, DIRS, classes)
    ids = [p.track_ids for p in shapes[TEAM_A].outfield]
    assert sorted([3, 103]) in [sorted(i) for i in ids]          # one player, two fragments
    assert len(shapes[TEAM_A].outfield) == 10


def test_players_seen_together_are_never_merged():
    a = PlayerAverage([1], TEAM_A, np.array([30.0, 30.0]), {1, 2, 3})
    b = PlayerAverage([2], TEAM_A, np.array([31.0, 30.0]), {3, 4})     # same frame 3: two people
    c = PlayerAverage([3], TEAM_B, np.array([30.5, 30.0]), {9})        # other team
    assert len(merge_fragments([a, b, c])) == 3


def test_goalkeeper_detection():
    frames, classes = _match(gk_class=True)
    shapes = team_shapes(frames, DIRS, classes)
    assert shapes[TEAM_A].goalkeeper.track_id == 0                     # by detected class
    frames, classes = _match(keeper=False)                             # keeper never tracked
    shapes = team_shapes(frames, DIRS, classes)
    assert shapes[TEAM_A].goalkeeper is None and len(shapes[TEAM_A].outfield) == 10
    # deepest player close to the back line is a defender, not a keeper
    back = [PlayerAverage([i], TEAM_A, np.array([x, 30.0]), set(range(50))) for i, x in enumerate([18, 20, 30])]
    assert find_goalkeeper(back, attacks_right=True) is None


def test_frame_selection_and_minimum_frames():
    frames, classes = _match(n=100)
    early = team_shapes(frames, DIRS, classes, frames=range(0, 20), min_frames=25)
    assert early[TEAM_A].outfield == []                                # 20 frames: nobody seen 25 times
    early = team_shapes(frames, DIRS, classes, frames=range(0, 20), min_frames=10)
    assert len(early[TEAM_A].outfield) == 10
