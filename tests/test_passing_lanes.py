import numpy as np
import pytest

from pitchlens.tactics.passing_lanes import (BLOCKED, OPEN, RISKY, LaneModel, ball_carrier, lane_margin,
                                             openness_from_margin, passing_lanes, status_of)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

MODEL = LaneModel()


def _lane(defender):
    """Ball at (30, 34), receiver 20 m ahead at (50, 34), one defender."""
    return lane_margin(np.array([30.0, 34.0]), np.array([50.0, 34.0]), np.array([defender], float))


def test_defender_on_the_line_blocks_it():
    margin, who = _lane([40.0, 34.0])               # standing in the middle of the lane
    assert margin < 0 and who == 0
    assert openness_from_margin(margin) < 0.3


def test_defender_far_away_leaves_it_open():
    margin, _ = _lane([40.0, 5.0])                  # 29 m to the side
    assert margin > 2.0 and openness_from_margin(margin) > 0.99


def test_margin_matches_the_physics():
    # defender 6 m beside the lane's midpoint. At a point x along the lane the ball needs
    # (x - 30) / 15 s; the defender needs 0.3 s + (distance - 1 m reach) / 7 m/s.
    # Only points the defender reaches before the receiver (at x=50) count.
    margin, _ = _lane([40.0, 40.0])
    expected = min((0.3 + max(0.0, np.hypot(40 - x, 6.0) - 1.0) / 7.0) - (x - 30) / 15.0
                   for x in np.linspace(30, 50, 25)
                   if 0.3 + max(0.0, np.hypot(40 - x, 6.0) - 1.0) / 7.0 < max(0.0, 50 - x - 1.0) / 7.0)
    assert margin == pytest.approx(expected)
    # e.g. at the midpoint alone the margin would be 0.3 + 5/7 - 10/15 = 0.35 s; the tightest
    # point is further on, where the ball is later but the defender barely further away
    assert margin < (0.3 + 5.0 / 7.0) - 10.0 / 15.0


def test_defender_in_front_of_the_receiver_cuts_the_lane():
    margin, _ = _lane([47.0, 34.0])                 # 3 m in front of the receiver, on the ball's side
    assert margin < 0


def test_defender_behind_the_receiver_cannot_intercept():
    margin, who = _lane([51.5, 34.0])               # marking from behind: can tackle, not intercept
    assert margin > 0.5 and openness_from_margin(margin) > 0.95


def test_no_defenders():
    margin, who = lane_margin(np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.empty((0, 2)))
    assert margin == float("inf") and who is None
    assert openness_from_margin(margin) == 1.0


def test_lanes_for_every_teammate():
    pos = np.array([[30, 34], [50, 34], [45, 10], [40, 55], [np.nan, np.nan],   # A: carrier + 3 + unknown
                    [40, 34], [60, 20], [30, 60]], float)                          # B defenders
    teams = np.array([TEAM_A] * 5 + [TEAM_B] * 3)
    lanes = {ln.receiver_index: ln for ln in passing_lanes(pos, teams, carrier=0)}
    assert set(lanes) == {1, 2, 3}                                  # carrier and unknown excluded
    assert lanes[1].status == BLOCKED and lanes[1].blocker_index == 5
    assert lanes[2].status == OPEN
    assert lanes[1].length_m == pytest.approx(20.0)
    assert all(0.0 <= ln.openness <= 1.0 for ln in lanes.values())


def test_status_bands():
    m = LaneModel()
    margins = np.linspace(-1, 1, 41)
    statuses = [status_of(openness_from_margin(x, m), m) for x in margins]
    assert statuses[0] == BLOCKED and statuses[-1] == OPEN and RISKY in statuses
    # statuses only ever get better as the margin grows
    rank = {BLOCKED: 0, RISKY: 1, OPEN: 2}
    assert all(rank[a] <= rank[b] for a, b in zip(statuses, statuses[1:]))


def test_ball_carrier():
    pos = np.array([[30, 34], [31, 35], [50, 20]], float)
    teams = np.array([TEAM_A, TEAM_B, TEAM_A])
    ball = np.array([30.5, 34.2])
    assert ball_carrier(pos, teams, ball) == 0
    assert ball_carrier(pos, teams, ball, team=TEAM_B) == 1
    assert ball_carrier(pos, teams, np.array([70.0, 50.0])) is None     # nobody within 3 m
    assert ball_carrier(pos, teams, None) is None


def _attack():
    """A attacks right. Carrier at x=40; B's midfield at x=50, back line at x=70."""
    pos = np.array([[40, 34],                      # 0 carrier
                    [38, 10],                      # 1 back/sideways, open
                    [62, 14],                      # 2 free between B's lines, in the gap on the left
                    [78, 50],                      # 3 behind B's back line, B's defender in the way
                    [55, 34],                      # 4 straight ahead, B midfielder in the lane
                    [50, 36], [50, 50], [53, 43],  # B midfield (5, 6, 7), shifted to the right
                    [70, 18], [70, 30], [70, 40], [76, 50]], float)   # B back line (8-11)
    teams = np.array([TEAM_A] * 5 + [TEAM_B] * 7)
    return pos, teams


def test_forward_gain_and_bypassed_opponents():
    from pitchlens.tactics.passing_lanes import evaluate_lanes
    pos, teams = _attack()
    lanes = {ln.receiver_index: ln for ln in evaluate_lanes(pos, teams, carrier=0, attacks_right=True)}
    assert lanes[1].forward_m == pytest.approx(-2.0) and lanes[1].bypassed == 0
    assert lanes[2].forward_m == pytest.approx(22.0) and lanes[2].bypassed == 3   # past the midfield three
    assert lanes[3].bypassed == 7                                                 # past everyone
    assert lanes[1].value() < lanes[2].value() < lanes[3].value()


def test_best_option_is_the_most_valuable_open_lane():
    from pitchlens.tactics.passing_lanes import best_option, evaluate_lanes
    pos, teams = _attack()
    lanes = evaluate_lanes(pos, teams, carrier=0, attacks_right=True)
    by_idx = {ln.receiver_index: ln for ln in lanes}
    assert by_idx[4].status == BLOCKED                  # midfielder standing in the lane
    assert by_idx[3].status != OPEN                     # defender next to the runner
    best = best_option(lanes)
    assert best is not None and best.status == OPEN
    assert best.receiver_index == 2                     # through the lines to the free man
    # the same picture attacking left (mirrored) gives the same answer
    mirrored = np.c_[105 - pos[:, 0], 68 - pos[:, 1]]
    best_left = best_option(evaluate_lanes(mirrored, teams, carrier=0, attacks_right=False))
    assert best_left.receiver_index == 2


def test_nothing_open():
    from pitchlens.tactics.passing_lanes import best_option
    pos = np.array([[30, 34], [40, 34], [35, 34]], float)       # defender right between them
    teams = np.array([TEAM_A, TEAM_A, TEAM_B])
    assert best_option(passing_lanes(pos, teams, carrier=0)) is None
