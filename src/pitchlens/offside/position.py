"""Which attackers are in an offside position?

An attacker is in an offside position when they are nearer to the opponents'
goal line than the offside line (which already accounts for the ball, the
second-last defender and the halfway line, see line.py).

Being in an offside position is not an offence by itself: it only matters at
the moment a teammate plays the ball (the pass moment, a later step).

Limits of this check:
  * positions are the players' feet; the Laws also count the head, body and
    knees (not the arms), which needs pose detection;
  * positions carry calibration/tracking error, so attackers within
    `tolerance_m` of the line are reported as too close to call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchlens.offside.line import OffsideLine
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

OFFSIDE = "offside"          # clearly beyond the line
ONSIDE = "onside"            # clearly level or behind
TOO_CLOSE = "too_close"      # within the measurement tolerance

DEFAULT_TOLERANCE_M = 0.3


@dataclass
class PositionCheck:
    player_index: int
    defending_team: int      # whose line the player was judged against
    status: str              # OFFSIDE / ONSIDE / TOO_CLOSE
    margin_m: float          # > 0: beyond the line (toward goal) by this much; < 0: behind it


def check_against_line(positions: np.ndarray, teams: np.ndarray, line: OffsideLine,
                       tolerance_m: float = DEFAULT_TOLERANCE_M) -> list[PositionCheck]:
    """Checks every visible player of the team attacking `line.defending_team`'s goal."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    teams = np.asarray(teams)
    attacking = TEAM_B if line.defending_team == TEAM_A else TEAM_A
    out = []
    for i in np.where(teams == attacking)[0]:
        x = positions[i, 0]
        if not np.isfinite(x):
            continue
        margin = line.depth - line.depth_of(x)
        status = OFFSIDE if margin > tolerance_m else ONSIDE if margin < -tolerance_m else TOO_CLOSE
        out.append(PositionCheck(int(i), line.defending_team, status, float(margin)))
    return out


def check_frame(positions: np.ndarray, teams: np.ndarray, lines: dict[int, OffsideLine],
                tolerance_m: float = DEFAULT_TOLERANCE_M,
                toward_goal: dict[bool, np.ndarray] | None = None) -> dict[int, PositionCheck]:
    """Every attacker in the frame, judged against the opponents' line. Keyed by player index.

    toward_goal: body-part positions per goal (goal_at_left -> positions), used when given.
    """
    out = {}
    for line in lines.values():
        pos = toward_goal.get(line.defends_left, positions) if toward_goal else positions
        for check in check_against_line(pos, teams, line, tolerance_m):
            out[check.player_index] = check
    return out
