"""Passing options in every frame where a team has the ball.

For each frame: who has the ball (possession from the touches, carrier = that
team's player within a few meters of the ball), and the lanes from the ball to
every visible teammate with their openness and value (passing_lanes.py).
Computed once and shared by the video overlay, the CSV and the pass analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.passing_lanes import Lane, LaneModel, ball_carrier, best_option, evaluate_lanes
from pitchlens.tactics.possession import NO_POSSESSION


@dataclass
class FrameOptions:
    frame: int
    team: int
    carrier: int                 # index into the frame's players
    carrier_track_id: int
    start: np.ndarray            # where passes start (ball position, pitch meters)
    lanes: list[Lane]
    best: Lane | None            # most valuable open lane, if any


def frame_options(players: Sequence[FramePlayers], ball: np.ndarray, possession: np.ndarray,
                  directions: dict[int, int], model: LaneModel = LaneModel(),
                  carrier_distance: float = 3.0) -> list[FrameOptions | None]:
    """One entry per frame (None when no team clearly has the ball at its feet).

    ball: (n, 2) ball path in pitch meters (NaN where unknown).
    """
    out: list[FrameOptions | None] = []
    for t, fp in enumerate(players):
        team = int(possession[t]) if t < len(possession) else NO_POSSESSION
        b = ball[t] if t < len(ball) else None
        if team == NO_POSSESSION or team not in directions or b is None or not np.isfinite(b).all():
            out.append(None)
            continue
        carrier = ball_carrier(fp.positions, fp.teams, b, team, carrier_distance)
        if carrier is None:
            out.append(None)
            continue
        lanes = evaluate_lanes(fp.positions, fp.teams, carrier, directions[team] > 0, b, model)
        out.append(FrameOptions(t, team, carrier, int(fp.track_ids[carrier]), np.asarray(b, float).copy(),
                                lanes, best_option(lanes)))
    return out
