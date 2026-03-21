"""Average position of every player, per team: the shape a team holds over time.

Tracking on broadcast footage has two problems this deals with:

  * ID fragments: when the tracker loses a player (occlusion, leaving the
    frame) and finds them again, they get a new track id, so one player can be
    several short tracks. Two tracks of the same team that are never seen in
    the same frame and hold the same place *relative to their team* are merged.
    Relative, because fragments come from different moments: the whole team
    may have moved up the pitch in between, but the player's role hasn't.
  * the goalkeeper: formations are about the 10 outfield players. The keeper
    is the player detected as a goalkeeper if the model can tell, otherwise
    the team's deepest player if they are near their own goal and clearly
    behind everyone else. If the keeper was never tracked, nobody is dropped.

Positions are pitch meters; averages only use frames where the player's
position is known (the frame was calibrated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pitchlens.calibration.pitch import Pitch
from pitchlens.detection.classes import GOALKEEPER
from pitchlens.offside.passes import FramePlayers
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B


@dataclass
class PlayerAverage:
    track_ids: list[int]          # one player can be several merged track fragments
    team: int
    position: np.ndarray          # (2,) average pitch position in meters
    frames: set[int] = field(default_factory=set)   # frames where they were seen
    keeper_votes: int = 0         # frames detected as a goalkeeper
    relative: np.ndarray = field(default_factory=lambda: np.zeros(2))  # average offset from team center

    @property
    def track_id(self) -> int:
        return self.track_ids[0]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def collect_averages(players: Sequence[FramePlayers], classes: Sequence[np.ndarray | None] | None = None,
                     frames: Sequence[int] | None = None) -> dict[int, PlayerAverage]:
    """Average position per track id (no merging yet). `frames` restricts which frames count."""
    sums: dict[int, np.ndarray] = {}
    rel_sums: dict[int, np.ndarray] = {}
    out: dict[int, PlayerAverage] = {}
    use = range(len(players)) if frames is None else frames
    for t in use:
        fp = players[t]
        cls = None if classes is None else classes[t]
        known = np.isfinite(fp.positions).all(axis=1) if len(fp.positions) else np.zeros(0, bool)
        centers = {team: fp.positions[known & (fp.teams == team)].mean(axis=0)
                   for team in (TEAM_A, TEAM_B) if (known & (fp.teams == team)).any()}
        for i, (tid, team) in enumerate(zip(fp.track_ids, fp.teams)):
            if team not in (TEAM_A, TEAM_B) or not known[i]:
                continue
            tid = int(tid)
            if tid not in out:
                out[tid] = PlayerAverage([tid], int(team), np.zeros(2))
                sums[tid], rel_sums[tid] = np.zeros(2), np.zeros(2)
            out[tid].frames.add(t)
            sums[tid] += fp.positions[i]
            rel_sums[tid] += fp.positions[i] - centers[int(team)]
            if cls is not None and cls[i] == GOALKEEPER:
                out[tid].keeper_votes += 1
    for tid, avg in out.items():
        avg.position = sums[tid] / avg.n_frames
        avg.relative = rel_sums[tid] / avg.n_frames
    return out


def merge_fragments(averages: list[PlayerAverage], max_distance: float = 5.0) -> list[PlayerAverage]:
    """Merges tracks of one team that never overlap in time and hold the same place in the team.

    Longest tracks first: each shorter track joins the compatible longer one whose average
    offset from the team center is nearest (within max_distance meters).
    """
    merged: list[PlayerAverage] = []
    for avg in sorted(averages, key=lambda a: -a.n_frames):
        best, best_d = None, max_distance
        for m in merged:
            if m.team != avg.team or m.frames & avg.frames:
                continue
            d = float(np.linalg.norm(m.relative - avg.relative))
            if d <= best_d:
                best, best_d = m, d
        if best is None:
            merged.append(PlayerAverage(list(avg.track_ids), avg.team, avg.position.copy(),
                                        set(avg.frames), avg.keeper_votes, avg.relative.copy()))
            continue
        n1, n2 = best.n_frames, avg.n_frames
        best.position = (best.position * n1 + avg.position * n2) / (n1 + n2)
        best.relative = (best.relative * n1 + avg.relative * n2) / (n1 + n2)
        best.frames |= avg.frames
        best.track_ids += avg.track_ids
        best.keeper_votes += avg.keeper_votes
    return merged


def find_goalkeeper(team_players: list[PlayerAverage], attacks_right: bool, pitch: Pitch = Pitch(),
                    max_depth: float = 20.0, min_gap: float = 5.0) -> PlayerAverage | None:
    """The team's goalkeeper among its averages, or None if they don't seem to be tracked."""
    if not team_players:
        return None
    voted = [p for p in team_players if p.keeper_votes > p.n_frames / 2]
    if voted:
        return max(voted, key=lambda p: p.keeper_votes)
    depth = [p.position[0] if attacks_right else pitch.length - p.position[0] for p in team_players]
    order = np.argsort(depth)
    deepest = order[0]
    gap = depth[order[1]] - depth[deepest] if len(order) > 1 else np.inf
    if depth[deepest] <= max_depth and gap >= min_gap:
        return team_players[deepest]
    return None


@dataclass
class TeamShape:
    team: int
    outfield: list[PlayerAverage]          # up to 10, most-seen first
    goalkeeper: PlayerAverage | None

    @property
    def positions(self) -> np.ndarray:
        return np.array([p.position for p in self.outfield]).reshape(-1, 2)


def team_shapes(players: Sequence[FramePlayers], directions: dict[int, int],
                classes: Sequence[np.ndarray | None] | None = None, frames: Sequence[int] | None = None,
                pitch: Pitch = Pitch(), min_frames: int = 25, max_outfield: int = 10,
                merge_distance: float = 5.0) -> dict[int, TeamShape]:
    """Average outfield shape of each team.

    directions:  team -> +1 (attacks right) / -1 (attacks left)
    min_frames:  a (merged) player must be seen at least this often to count
    """
    merged = merge_fragments(list(collect_averages(players, classes, frames).values()), merge_distance)
    out = {}
    for team, direction in directions.items():
        mine = [p for p in merged if p.team == team and p.n_frames >= min_frames]
        keeper = find_goalkeeper(mine, direction > 0, pitch)
        outfield = sorted((p for p in mine if p is not keeper), key=lambda p: -p.n_frames)[:max_outfield]
        out[team] = TeamShape(team, outfield, keeper)
    return out
