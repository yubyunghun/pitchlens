"""Freeze the offside situation at the moment the ball is played.

Offside is judged at the moment a teammate plays the ball, not when it is
received. So for each pass we take the kick frame and check every teammate of
the passer against the opponents' offside line *in that frame*. A player who
runs past the line after the kick is fine; one who was past it at the kick
stays in an offside position even if they come back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pitchlens.offside.line import OffsideLine
from pitchlens.offside.passes import FramePlayers, Pass
from pitchlens.offside.position import DEFAULT_TOLERANCE_M, PositionCheck, check_against_line
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B


@dataclass
class FrozenFrame:
    frame: int                                   # the kick frame
    attacking_team: int | None                   # the passer's team
    line: OffsideLine | None                     # opponents' offside line at the kick (None = unknown)
    checks: dict[int, PositionCheck] = field(default_factory=dict)   # track id -> check (passer excluded)

    def offside_track_ids(self) -> list[int]:
        return [tid for tid, c in self.checks.items() if c.status == "offside"]


def freeze_pass(p: Pass, players: list[FramePlayers], lines: list[dict[int, OffsideLine]],
                tolerance_m: float = DEFAULT_TOLERANCE_M) -> FrozenFrame:
    """players[t] / lines[t]: tracked players and offside lines (by defending team) in frame t."""
    t = p.kick.frame
    team = p.kick.team
    if team not in (TEAM_A, TEAM_B) or t >= len(players) or t >= len(lines):
        return FrozenFrame(t, team, None)
    defending = TEAM_B if team == TEAM_A else TEAM_A
    line = lines[t].get(defending)
    if line is None:
        return FrozenFrame(t, team, None)
    fp = players[t]
    checks = {}
    for c in check_against_line(fp.positions_toward(line.defends_left), fp.teams, line, tolerance_m):
        tid = int(fp.track_ids[c.player_index])
        if tid != p.kick.track_id:            # the passer is not judged on their own pass
            checks[tid] = c
    return FrozenFrame(t, team, line, checks)
