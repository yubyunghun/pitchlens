import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.detection.classes import GOALKEEPER, PLAYER
from pitchlens.offside.direction import ATTACKS_LEFT, ATTACKS_RIGHT, DirectionEstimate
from pitchlens.offside.line import (BALL, DEFENDER, HALFWAY, frame_offside_lines, goal_area_in_view,
                                    offside_line)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B, UNKNOWN

L = 105.0
NAN = np.nan


def _players(defenders_x, attackers_x=(), y=34.0, extra=None):
    """Team A defends; B attacks. Returns positions, teams."""
    xs = list(defenders_x) + list(attackers_x)
    pos = np.array([[x, y + i] for i, x in enumerate(xs)], float)
    teams = np.array([TEAM_A] * len(defenders_x) + [TEAM_B] * len(attackers_x))
    if extra is not None:                      # e.g. a referee
        pos = np.vstack([pos, extra[0]])
        teams = np.append(teams, extra[1])
    return pos, teams


def test_second_last_defender_sets_the_line():
    pos, teams = _players([0.8, 12.0, 10.5, 25.0], attackers_x=[9.0, 30.0])
    line = offside_line(pos, teams, TEAM_A, defends_left=True)
    assert line.source == DEFENDER and line.x == 10.5 and line.defender_index == 2
    assert not line.keeper_assumed
    assert line.is_beyond(9.0) and not line.is_beyond(11.0)


def test_ball_nearer_goal_than_second_last_defender():
    pos, teams = _players([0.8, 12.0, 15.0])
    line = offside_line(pos, teams, TEAM_A, True, ball=np.array([7.0, 30.0]))
    assert line.source == BALL and line.x == 7.0 and line.defender_index is None
    # ball further out than the defender: defender still sets it
    assert offside_line(pos, teams, TEAM_A, True, ball=np.array([20.0, 30.0])).source == DEFENDER


def test_line_never_beyond_halfway():
    pos, teams = _players([40.0, 58.0, 60.0, 62.0])        # keeper well up, back line past halfway
    line = offside_line(pos, teams, TEAM_A, True)
    assert line.source == HALFWAY and line.x == L / 2


def test_defending_the_right_goal_is_mirrored():
    pos, teams = _players([104.0, 93.0, 95.0, 80.0])
    line = offside_line(pos, teams, TEAM_A, defends_left=False)
    assert line.x == 95.0 and line.depth == 10.0
    assert line.is_beyond(97.0) and not line.is_beyond(90.0)
    pos, teams = _players([104.0, 60.0, 62.0])              # 43-45 m out: still their own half
    assert offside_line(pos, teams, TEAM_A, False).x == 62.0
    pos, teams = _players([70.0, 48.0, 45.0])              # back line pushed past halfway
    assert offside_line(pos, teams, TEAM_A, False).x == L / 2


def test_keeper_out_of_view_is_assumed():
    pos, teams = _players([18.0, 15.0, 27.0])              # camera shows the back line, not the goal
    line = offside_line(pos, teams, TEAM_A, True, keeper_in_view=False)
    assert line.keeper_assumed and line.x == 15.0          # deepest visible = second-last
    # if the keeper area IS in view, the same players mean the deepest one is the keeper
    assert offside_line(pos, teams, TEAM_A, True, keeper_in_view=True).x == 18.0
    # a detected goalkeeper (fine-tuned model) means nothing has to be assumed
    classes = np.array([PLAYER, GOALKEEPER, PLAYER])
    line = offside_line(pos, teams, TEAM_A, True, keeper_in_view=False, classes=classes)
    assert not line.keeper_assumed and line.x == 18.0


def test_player_in_own_goal_area_counts_as_keeper():
    # goal area reported out of view (e.g. at the frame edge), but someone is tracked inside it
    pos, teams = _players([3.0, 14.0, 16.0])
    line = offside_line(pos, teams, TEAM_A, True, keeper_in_view=False)
    assert not line.keeper_assumed and line.x == 14.0
    pos, teams = _players([101.0, 62.0, 58.0])                # same for the right goal
    line = offside_line(pos, teams, TEAM_A, False, keeper_in_view=False)
    assert not line.keeper_assumed and line.x == 62.0


def test_not_enough_defenders_visible():
    pos, teams = _players([3.0])                               # only the keeper is visible
    assert offside_line(pos, teams, TEAM_A, True, keeper_in_view=True) is None
    assert offside_line(pos, teams, TEAM_A, True, keeper_in_view=False) is None
    pos, teams = _players([20.0])                              # one defender, keeper off-screen
    assert offside_line(pos, teams, TEAM_A, True, keeper_in_view=False).x == 20.0
    assert offside_line(np.empty((0, 2)), np.empty(0, int), TEAM_A, True, keeper_in_view=False) is None


def test_referee_and_unknown_positions_ignored():
    pos, teams = _players([0.8, 14.0, NAN, 20.0], extra=([5.0, 30.0], UNKNOWN))
    assert offside_line(pos, teams, TEAM_A, True).x == 14.0


def _calib(view_left_goal: bool):
    """Camera looking at the left third of the pitch, or at the middle."""
    x0, x1 = (0, 40) if view_left_goal else (35, 70)
    cam = cv2.getPerspectiveTransform(np.float32([[x0, 0], [x1, 0], [x1, 68], [x0, 68]]),
                                      np.float32([[150, 150], [1130, 150], [1350, 700], [-70, 700]]))
    return PitchCalibration(np.linalg.inv(cam), (1280, 720))


def test_goal_area_visibility():
    assert goal_area_in_view(_calib(True), left_goal=True)
    assert not goal_area_in_view(_calib(True), left_goal=False)
    assert not goal_area_in_view(_calib(False), left_goal=True)


def test_frame_lines_for_both_teams():
    directions = DirectionEstimate({TEAM_A: ATTACKS_RIGHT, TEAM_B: ATTACKS_LEFT}, 1.0, 10)
    # camera on the left third: A's goal (left) visible, B's goal (right) not
    pos = np.array([[1, 34], [12, 30], [14, 40], [30, 20],      # A: keeper + defenders
                    [10, 32], [25, 45], [38, 34]], float)        # B: attackers / B's deepest visible
    teams = np.array([TEAM_A] * 4 + [TEAM_B] * 3)
    lines = frame_offside_lines(pos, teams, directions, _calib(True))
    assert lines[TEAM_A].x == 12 and not lines[TEAM_A].keeper_assumed
    # B defends the right goal, which is off-screen: deepest visible B player is second-last,
    # but that's beyond halfway from B's goal, so the halfway line applies
    assert lines[TEAM_B].keeper_assumed and lines[TEAM_B].source == HALFWAY

