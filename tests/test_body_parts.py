import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.offside.body_parts import (COCO_NAMES, COUNTING, ground_points, nearest_to_goal_x,
                                          positions_toward_goal)

# simple top-down-ish camera: 20 px per meter, image x = pitch x * 20, image y = pitch y * 20
CALIB = PitchCalibration(np.linalg.inv(np.diag([20.0, 20.0, 1.0])), (2100, 1360))


def _pose(foot_x_m, foot_y_m, lean_m=0.0, arm_reach_m=0.0, height_px=36.0):
    """A standing player (pixels): body leaning `lean_m` meters toward +x, arm stretched `arm_reach_m`."""
    fx, fy = foot_x_m * 20, foot_y_m * 20
    kp = np.zeros((17, 3))
    for i, n in enumerate(COCO_NAMES):
        frac = {"nose": 1.0, "left_eye": 0.98, "right_eye": 0.98, "left_ear": 0.97, "right_ear": 0.97,
                "left_shoulder": 0.82, "right_shoulder": 0.82, "left_elbow": 0.65, "right_elbow": 0.65,
                "left_wrist": 0.5, "right_wrist": 0.5, "left_hip": 0.5, "right_hip": 0.5,
                "left_knee": 0.27, "right_knee": 0.27, "left_ankle": 0.0, "right_ankle": 0.0}[n]
        x = fx + lean_m * 20 * frac                                   # leaning: higher parts further forward
        if "wrist" in n or "elbow" in n:
            x += arm_reach_m * 20
        kp[i] = [x, fy - frac * height_px, 0.9]
    box = np.array([fx - 10, fy - height_px - 2, fx + 10, fy])
    return kp, box


def test_arms_do_not_count():
    assert all("elbow" not in COCO_NAMES[i] and "wrist" not in COCO_NAMES[i] for i in COUNTING)
    assert len(COUNTING) == 13


def test_leaning_head_counts_arm_does_not():
    kp, box = _pose(50.0, 30.0, lean_m=0.6, arm_reach_m=1.5)
    # attacking toward the goal at x = 105 (right): head 0.6 m ahead of the feet counts, the arm doesn't
    assert abs(nearest_to_goal_x(kp, box, CALIB, goal_at_left=False) - 50.6) < 1e-6
    # toward the goal at x = 0 the nearest counting part is the feet (the body leans the other way)
    assert abs(nearest_to_goal_x(kp, box, CALIB, goal_at_left=True) - 50.0) < 1e-6


def test_body_points_dropped_to_foot_level():
    kp, box = _pose(40.0, 20.0, lean_m=0.4)
    pts = ground_points(kp, box)
    assert len(pts) == 13 and np.allclose(pts[:, 1], 20.0 * 20)    # all at the ankles' image row
    kp[[15, 16], 2] = 0.0                                          # ankles not detected: use the box bottom
    assert np.allclose(ground_points(kp, box)[:, 1], box[3])


def test_low_confidence_and_missing_poses_fall_back_to_feet():
    kp, box = _pose(30.0, 30.0, lean_m=0.5)
    kp[:, 2] = 0.1
    assert nearest_to_goal_x(kp, box, CALIB, False) is None
    assert nearest_to_goal_x(None, box, CALIB, False) is None
    feet = np.array([[30.0, 30.0], [60.0, 20.0], [np.nan, np.nan]])
    kp2, box2 = _pose(60.0, 20.0, lean_m=0.8)
    out = positions_toward_goal(feet, [kp, kp2, None], np.stack([box, box2, box]), CALIB, goal_at_left=False)
    assert out[0, 0] == 30.0                                       # unusable pose: feet
    assert abs(out[1, 0] - 60.8) < 1e-6 and out[1, 1] == 20.0      # only x moves
    assert np.isnan(out[2]).all()


def test_a_lean_makes_the_difference_at_the_line():
    """Feet level with the second-last defender; the attacker's head is 0.4 m ahead -> offside by body."""
    from pitchlens.offside.line import offside_line
    from pitchlens.offside.position import OFFSIDE, TOO_CLOSE, check_against_line
    from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B
    feet = np.array([[104.0, 34.0], [92.0, 20.0], [88.0, 50.0], [92.0, 40.0]])     # B keeper, B, B, A attacker
    teams = np.array([TEAM_B, TEAM_B, TEAM_B, TEAM_A])
    poses = [_pose(*f, lean_m=0.0) for f in feet[:3]] + [_pose(92.0, 40.0, lean_m=0.4)]
    kps, boxes = [p[0] for p in poses], np.stack([p[1] for p in poses])
    by_feet = check_against_line(feet, teams, offside_line(feet, teams, TEAM_B, defends_left=False))[0]
    body = positions_toward_goal(feet, kps, boxes, CALIB, goal_at_left=False)
    by_body = check_against_line(body, teams, offside_line(body, teams, TEAM_B, defends_left=False))[0]
    assert by_feet.status == TOO_CLOSE and by_body.status == OFFSIDE
    assert abs(by_body.margin_m - 0.4) < 1e-6
