"""Who has the ball, frame by frame, and each team's shape with and without it.

Possession comes from the detected touches (offside/passes.py): from a touch,
the ball belongs to that player's team until the next touch. It is unknown
  * before the first touch,
  * after a touch nobody could be credited with (e.g. an off-screen player),
  * once `max_hold_s` passes without any touch (ball out of play, or the
    ball simply wasn't detected for a while).

Teams often play two shapes: e.g. a 4-3-3 with the ball and a 4-1-4-1 or
4-4-2 without it, so shapes are averaged separately per phase.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pitchlens.calibration.pitch import Pitch
from pitchlens.offside.passes import FramePlayers, Touch
from pitchlens.tactics.average_shape import TeamShape, team_shapes

NO_POSSESSION = -1
IN, OUT, ALL = "in_possession", "out_of_possession", "all"


def possession_by_frame(touches: Sequence[Touch], n_frames: int, fps: float,
                        max_hold_s: float = 8.0) -> np.ndarray:
    """(n_frames,) team id in possession per frame, or NO_POSSESSION."""
    out = np.full(n_frames, NO_POSSESSION, dtype=int)
    max_hold = int(round(max_hold_s * fps))
    ordered = sorted(touches, key=lambda t: t.frame)
    for k, touch in enumerate(ordered):
        if touch.team is None:
            continue                                        # unknown until the next credited touch
        end = ordered[k + 1].frame if k + 1 < len(ordered) else n_frames
        end = min(end, touch.frame + max_hold, n_frames)
        out[max(0, touch.frame):end] = touch.team
    return out


def phase_frames(possession: np.ndarray, team: int) -> dict[str, np.ndarray]:
    """Frame indices per phase for one team."""
    known = possession != NO_POSSESSION
    return {IN: np.flatnonzero(possession == team),
            OUT: np.flatnonzero(known & (possession != team)),
            ALL: np.arange(len(possession))}


def phase_shapes(players: Sequence[FramePlayers], directions: dict[int, int], possession: np.ndarray,
                 classes: Sequence[np.ndarray | None] | None = None, pitch: Pitch = Pitch(),
                 min_frames: int = 25) -> dict[int, dict[str, TeamShape]]:
    """team -> {in_possession / out_of_possession / all -> average shape}."""
    out: dict[int, dict[str, TeamShape]] = {}
    for team in directions:
        phases = phase_frames(possession, team)
        out[team] = {phase: team_shapes(players, {team: directions[team]}, classes, frames, pitch,
                                        min_frames=min_frames)[team]
                     for phase, frames in phases.items()}
    return out
