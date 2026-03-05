"""Where is the offside line?

Laws of the Game (Law 11): an attacker is in an offside position if any part
of their head, body or feet is in the opponents' half and nearer to the
opponents' goal line than both the ball and the second-last opponent.

So for the defending team the offside line sits at whichever of these is
nearer to the goal line they defend:
  * the second-last defender (the goalkeeper is usually the last),
  * the ball,
and never further out than the halfway line.

Broadcast cameras often don't show the goalkeeper. If the goal area isn't in
view (and no goalkeeper was detected) we assume the keeper is there unseen, so
the deepest *visible* defender is the second-last one. The result says when
that assumption was made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.pitch import GOAL_AREA_DEPTH, GOAL_AREA_WIDTH, Pitch
from pitchlens.detection.classes import GOALKEEPER
from pitchlens.offside.direction import ATTACKS_RIGHT, DirectionEstimate

DEFENDER, BALL, HALFWAY = "defender", "ball", "halfway"


@dataclass
class OffsideLine:
    defending_team: int
    x: float                     # pitch x of the line (meters)
    depth: float                 # distance from the goal line the team defends (meters)
    source: str                  # DEFENDER, BALL or HALFWAY: what set the line
    defender_index: int | None   # index (into the frame's players) of the second-last defender
    keeper_assumed: bool         # True if the goalkeeper was out of view and assumed
    defends_left: bool           # True if the defending team defends the goal at x = 0
    pitch_length: float = 105.0

    def depth_of(self, x: float) -> float:
        """Distance of pitch position x from the goal line this team defends."""
        return x if self.defends_left else self.pitch_length - x

    def is_beyond(self, x: float) -> bool:
        """Is pitch position x nearer to the defended goal line than the line?"""
        return self.depth_of(x) < self.depth


def offside_line(positions: np.ndarray, teams: np.ndarray, defending_team: int, defends_left: bool,
                 pitch: Pitch = Pitch(), ball: np.ndarray | None = None, keeper_in_view: bool = True,
                 classes: np.ndarray | None = None) -> OffsideLine | None:
    """Offside line of `defending_team` in one frame, or None if it can't be determined.

    positions:      (N, 2) pitch meters of all tracked players (NaN = unknown)
    defends_left:   True if the team defends the goal at x = 0
    ball:           (2,) ball pitch position, if known
    keeper_in_view: whether the goal area the team defends is visible in the frame
    classes:        detection class per player; a visible GOALKEEPER removes the need to assume one
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    teams = np.asarray(teams)
    depth = positions[:, 0] if defends_left else pitch.length - positions[:, 0]
    mine = np.where((teams == defending_team) & np.isfinite(positions).all(axis=1))[0]
    order = mine[np.argsort(depth[mine])]                    # deepest (nearest own goal) first

    keeper_seen = classes is not None and any(np.asarray(classes)[order] == GOALKEEPER)
    # someone standing in their own six-yard box is the keeper, whatever the camera check says
    keeper_seen = keeper_seen or (len(order) > 0 and depth[order[0]] <= GOAL_AREA_DEPTH)
    keeper_assumed = not keeper_in_view and not keeper_seen
    rank = 0 if keeper_assumed else 1                        # which visible player is second-last
    if len(order) <= rank:
        return None                                          # second-last defender not in view

    idx = int(order[rank])
    line_depth, source, defender = float(depth[idx]), DEFENDER, idx
    if ball is not None and np.isfinite(ball).all():
        ball_depth = float(ball[0] if defends_left else pitch.length - ball[0])
        if ball_depth < line_depth:
            line_depth, source, defender = ball_depth, BALL, None
    half = pitch.length / 2
    if line_depth > half:
        line_depth, source, defender = half, HALFWAY, None

    return OffsideLine(defending_team, line_depth if defends_left else pitch.length - line_depth,
                       line_depth, source, defender, keeper_assumed, defends_left, pitch.length)


def goal_area_in_view(calib: PitchCalibration, left_goal: bool, margin_px: float = 0.0) -> bool:
    """Is the whole goal area (six-yard box) of that goal inside the camera frame?"""
    p = calib.pitch
    x0, x1 = (0.0, GOAL_AREA_DEPTH) if left_goal else (p.length - GOAL_AREA_DEPTH, p.length)
    y0, y1 = p.width / 2 - GOAL_AREA_WIDTH / 2, p.width / 2 + GOAL_AREA_WIDTH / 2
    img = calib.to_image(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    w, h = calib.image_size
    return bool(np.isfinite(img).all()
                and (img[:, 0] >= -margin_px).all() and (img[:, 0] <= w + margin_px).all()
                and (img[:, 1] >= -margin_px).all() and (img[:, 1] <= h + margin_px).all())


def frame_offside_lines(positions: np.ndarray, teams: np.ndarray, directions: DirectionEstimate,
                        calib: PitchCalibration, ball: np.ndarray | None = None,
                        classes: np.ndarray | None = None) -> dict[int, OffsideLine]:
    """Offside line for every team in one frame (teams whose line can't be found are left out)."""
    out = {}
    for team, attack in directions.directions.items():
        defends_left = attack == ATTACKS_RIGHT
        line = offside_line(positions, teams, team, defends_left, calib.pitch, ball,
                            goal_area_in_view(calib, defends_left), classes)
        if line is not None:
            out[team] = line
    return out
