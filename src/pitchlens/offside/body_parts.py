"""Offside by body parts instead of feet.

Law 11: the parts that count are the head, body and feet; the arms don't
(the arm starts at the bottom of the armpit). From the 17 COCO pose keypoints
that leaves nose, eyes, ears, shoulders, hips, knees and ankles, not elbows
or wrists.

The calibration only maps the ground, so a head 1.8 m up can't be projected
directly. Each body point is dropped straight down in the image to the
player's foot level first: close to a player, the world's vertical is nearly
vertical in a broadcast image, so this lands on the point of the ground under
that body part. It's an approximation that gets worse toward the image edges
and with strong camera tilt.

For offside, what matters for every player is the counting part nearest the
goal line being judged: for the second-last defender (the line) and for the
attacker (the position), both measured toward the same goal.
"""

from __future__ import annotations

import numpy as np

from pitchlens.calibration.homography import PitchCalibration

COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle"]
ARMS = {"left_elbow", "right_elbow", "left_wrist", "right_wrist"}
COUNTING = [i for i, n in enumerate(COCO_NAMES) if n not in ARMS]
ANKLES = [COCO_NAMES.index("left_ankle"), COCO_NAMES.index("right_ankle")]


def ground_points(keypoints: np.ndarray, box: np.ndarray, min_confidence: float = 0.3) -> np.ndarray:
    """Image points under each confident counting body part, at the player's foot level.

    keypoints: (17, 3) COCO keypoints (x, y, confidence) in image pixels
    box:       the player's xyxy box; its bottom is the foot level if no ankle is confident
    """
    kp = np.asarray(keypoints, dtype=float).reshape(-1, 3)
    ankles = [kp[i, 1] for i in ANKLES if kp[i, 2] >= min_confidence]
    foot_y = max(ankles) if ankles else float(box[3])
    use = [i for i in COUNTING if kp[i, 2] >= min_confidence]
    return np.array([[kp[i, 0], foot_y] for i in use]).reshape(-1, 2)


def nearest_to_goal_x(keypoints: np.ndarray | None, box: np.ndarray, calib: PitchCalibration,
                      goal_at_left: bool, min_confidence: float = 0.3) -> float | None:
    """Pitch x of the player's counting body part nearest the goal at x=0 (goal_at_left) or x=length.

    None if no counting part was detected confidently (callers fall back to the feet).
    """
    if keypoints is None:
        return None
    pts = ground_points(keypoints, box, min_confidence)
    if len(pts) == 0:
        return None
    xs = calib.to_pitch(pts)[:, 0]
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return None
    return float(xs.min() if goal_at_left else xs.max())


def positions_toward_goal(feet_pitch: np.ndarray, keypoints: list[np.ndarray | None], boxes: np.ndarray,
                          calib: PitchCalibration, goal_at_left: bool, min_confidence: float = 0.3) -> np.ndarray:
    """Players' pitch positions for judging offside toward one goal: x moved to the body part nearest it.

    Players without a usable pose keep their feet position.
    """
    out = np.asarray(feet_pitch, dtype=float).reshape(-1, 2).copy()
    for i, kp in enumerate(keypoints):
        x = nearest_to_goal_x(kp, boxes[i], calib, goal_at_left, min_confidence)
        if x is not None and np.isfinite(out[i]).all():
            out[i, 0] = x
    return out
