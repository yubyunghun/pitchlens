"""Which goal does each team attack?

Pitch x runs from the left goal line (x = 0) to the right one (x = length).
Offside is judged toward the goal a team attacks, so this comes first.

Evidence, per frame with enough calibrated players of both teams:
  * shape: the team defending the left goal sits closer to it than its
    opponents (median x), even under attack, because its defenders stand
    between the attackers and the goal.
  * deepest players: each team's last man (goalkeeper / last defender) is
    the player nearest the goal it defends, at both ends of the visible play.

Frames vote; the majority decides and the share of agreeing votes is the
confidence. Teams don't swap ends within a half, so one answer per clip.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

ATTACKS_RIGHT = 1    # toward x = pitch length
ATTACKS_LEFT = -1    # toward x = 0

SHAPE_WEIGHT = 1.0
DEEPEST_WEIGHT = 0.5   # each end; weaker because the camera often shows only part of the pitch


@dataclass
class DirectionEstimate:
    directions: dict[int, int]   # team id -> ATTACKS_RIGHT / ATTACKS_LEFT
    confidence: float            # 0.5 (coin flip) .. 1.0 (all evidence agrees)
    frames_used: int

    def describe(self, team: int) -> str:
        return "left -> right" if self.directions[team] == ATTACKS_RIGHT else "right -> left"


def infer_attack_directions(frames: Iterable[tuple[np.ndarray, np.ndarray]],
                            min_players: int = 3) -> DirectionEstimate | None:
    """frames: (positions (N, 2) in pitch meters, team ids (N,)) per frame. NaN positions are ignored.

    Returns None if there isn't enough evidence (no frame with `min_players` of each team,
    or the evidence is perfectly split).
    """
    votes = total = 0.0
    used = 0
    for xy, teams in frames:
        xy, teams = np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(teams)
        ok = np.isfinite(xy).all(axis=1)
        a, b = xy[ok & (teams == TEAM_A), 0], xy[ok & (teams == TEAM_B), 0]
        if len(a) < min_players or len(b) < min_players:
            continue
        used += 1
        # positive vote = team A defends the left goal, i.e. attacks right
        for weight, diff in ((SHAPE_WEIGHT, np.median(b) - np.median(a)),
                             (DEEPEST_WEIGHT, b.min() - a.min()),
                             (DEEPEST_WEIGHT, b.max() - a.max())):
            votes += weight * np.sign(diff)
            total += weight
    if used == 0 or votes == 0:
        return None
    a_dir = ATTACKS_RIGHT if votes > 0 else ATTACKS_LEFT
    return DirectionEstimate({TEAM_A: a_dir, TEAM_B: -a_dir}, (abs(votes) / total + 1) / 2, used)


def frames_from_tracks_csv(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reads a tracks.csv written by pitchlens-track into per-frame (positions, teams)."""
    by_frame: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["x_m"] == "" or row["y_m"] == "":
                continue
            by_frame[int(row["frame"])].append((float(row["x_m"]), float(row["y_m"]), int(row["team"])))
    out = []
    for frame in sorted(by_frame):
        rows = np.array(by_frame[frame])
        out.append((rows[:, :2], rows[:, 2].astype(int)))
    return out
