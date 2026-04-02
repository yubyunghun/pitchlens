"""Passing options in every frame where a team has the ball.

For each frame: who has the ball (possession from the touches, carrier = that
team's player within a few meters of the ball), and the lanes from the ball to
every visible teammate with their openness and value (passing_lanes.py).
Computed once and shared by the video overlay, the CSV and the pass analysis.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.passing_lanes import (BLOCKED, OPEN, RISKY, Lane, LaneModel, ball_carrier, best_option,
                                             evaluate_lanes)
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
    track_ids: np.ndarray | None = None   # the frame's track ids (to name receivers)

    def receiver_track_id(self, lane: Lane) -> int | None:
        return None if self.track_ids is None else int(self.track_ids[lane.receiver_index])

    def count(self, status: str) -> int:
        return sum(ln.status == status for ln in self.lanes)


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
                                lanes, best_option(lanes), np.asarray(fp.track_ids)))
    return out


CSV_HEADER = ["frame", "time_s", "team", "carrier_track_id", "open", "risky", "blocked",
              "best_receiver_track_id", "best_forward_m", "best_bypassed", "best_openness"]


def write_options_csv(path: str | Path, options: Sequence[FrameOptions | None], fps: float) -> int:
    """One row per frame where a team has the ball at a player's feet. Returns the number of rows."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for o in options:
            if o is None:
                continue
            b = o.best
            w.writerow([o.frame, round(o.frame / fps, 2), o.team, o.carrier_track_id,
                        o.count(OPEN), o.count(RISKY), o.count(BLOCKED),
                        "" if b is None else o.receiver_track_id(b),
                        "" if b is None else round(b.forward_m, 1),
                        "" if b is None else b.bypassed,
                        "" if b is None else round(b.openness, 3)])
            rows += 1
    return rows
