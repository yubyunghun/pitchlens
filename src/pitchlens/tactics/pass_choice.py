"""Did the passer pick a good option? And how well does each team cut lanes off?

For every detected pass (offside/passes.py) the passing options are taken from
just before the kick, while the passer still had the ball at their feet
(tactics/lane_options.py). From them:
  * the lane actually used: was it open, how much ground did it gain, how
    many opponents did it bypass (lines broken)?
  * was it the best open option available?

Per team that adds up to a passing profile (completed / forward /
line-breaking / best-option shares) and a defensive one: how many open
forward lanes the team allowed while the opponent had the ball.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pitchlens.offside.passes import Pass
from pitchlens.tactics.lane_options import FrameOptions
from pitchlens.tactics.passing_lanes import OPEN, Lane


@dataclass
class PassChoice:
    frame: int                       # kick frame
    team: int
    passer_track_id: int
    receiver_track_id: int | None
    completed: bool
    options_frame: int | None        # frame the options were taken from (None: no options found)
    used: Lane | None                # the lane to the receiver (None: receiver not a visible teammate)
    best: Lane | None
    best_receiver_track_id: int | None

    @property
    def chose_best(self) -> bool | None:
        if self.best is None or self.receiver_track_id is None:
            return None
        return self.receiver_track_id == self.best_receiver_track_id

    @property
    def forward(self) -> bool | None:
        return None if self.used is None else self.used.forward_m > 0

    @property
    def line_breaking(self) -> bool | None:
        return None if self.used is None else self.used.bypassed > 0


def options_before_kick(p: Pass, options: Sequence[FrameOptions | None], lookback: int = 5) -> FrameOptions | None:
    """The passer's options from the last frame (at or before the kick) where they had the ball."""
    for t in range(p.kick.frame, max(-1, p.kick.frame - lookback - 1), -1):
        if 0 <= t < len(options):
            o = options[t]
            if o is not None and o.carrier_track_id == p.kick.track_id:
                return o
    return None


def analyze_passes(passes: Sequence[Pass], options: Sequence[FrameOptions | None]) -> list[PassChoice]:
    out = []
    for p in passes:
        if p.kick.track_id is None or p.kick.team is None:
            continue
        receiver = None if p.receive is None else p.receive.track_id
        o = options_before_kick(p, options)
        used = best = None
        best_id = None
        if o is not None:
            best = o.best
            best_id = None if best is None else o.receiver_track_id(best)
            if receiver is not None:
                used = next((ln for ln in o.lanes if o.receiver_track_id(ln) == receiver), None)
        out.append(PassChoice(p.kick.frame, p.kick.team, p.kick.track_id, receiver, p.completed,
                              None if o is None else o.frame, used, best, best_id))
    return out


@dataclass
class TeamPassing:
    team: int
    passes: int
    completed: float | None          # shares (0..1); None when nothing to measure
    forward: float | None
    line_breaking: float | None
    chose_best: float | None
    open_lanes_on_ball: float | None         # average open lanes per frame with the ball
    open_forward_lanes_allowed: float | None  # average open forward lanes the OPPONENT had


def _share(values: list[bool | None]) -> float | None:
    known = [v for v in values if v is not None]
    return None if not known else float(np.mean(known))


def team_passing(choices: Sequence[PassChoice], options: Sequence[FrameOptions | None],
                 teams: Sequence[int]) -> list[TeamPassing]:
    out = []
    for team in teams:
        mine = [c for c in choices if c.team == team]
        on_ball = [o for o in options if o is not None and o.team == team]
        against = [o for o in options if o is not None and o.team != team]
        out.append(TeamPassing(
            team, len(mine),
            _share([c.completed for c in mine]),
            _share([c.forward for c in mine]),
            _share([c.line_breaking for c in mine]),
            _share([c.chose_best for c in mine]),
            None if not on_ball else float(np.mean([o.count(OPEN) for o in on_ball])),
            None if not against else float(np.mean(
                [sum(ln.status == OPEN and ln.forward_m > 0 for ln in o.lanes) for o in against])),
        ))
    return out


def _pct(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:4.0%}"


def summarize_passing(stats: Sequence[TeamPassing], team_names: dict[int, str]) -> str:
    lines = ["Passing (completed / forward / line-breaking / best option taken; lanes):"]
    for s in stats:
        name = team_names.get(s.team, f"team {s.team}")
        if s.passes == 0:
            lines.append(f"  {name}: no passes detected")
            continue
        lanes = "" if s.open_lanes_on_ball is None else f"; {s.open_lanes_on_ball:.1f} open lanes on the ball"
        allowed = ("" if s.open_forward_lanes_allowed is None
                   else f"; allowed {s.open_forward_lanes_allowed:.1f} open forward lanes when defending")
        lines.append(f"  {name}: {s.passes} passes: {_pct(s.completed)} / {_pct(s.forward)} / "
                     f"{_pct(s.line_breaking)} / {_pct(s.chose_best)}{lanes}{allowed}")
    return "\n".join(lines)


CSV_HEADER = ["kick_frame", "time_s", "team", "passer_track_id", "receiver_track_id", "completed",
              "lane_status", "lane_openness", "forward_m", "bypassed", "best_receiver_track_id", "chose_best"]


def write_pass_choices_csv(path: str | Path, choices: Sequence[PassChoice], fps: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for c in choices:
            u = c.used
            w.writerow([c.frame, round(c.frame / fps, 2), c.team, c.passer_track_id,
                        "" if c.receiver_track_id is None else c.receiver_track_id, c.completed,
                        "" if u is None else u.status, "" if u is None else round(u.openness, 3),
                        "" if u is None else round(u.forward_m, 1), "" if u is None else u.bypassed,
                        "" if c.best_receiver_track_id is None else c.best_receiver_track_id,
                        "" if c.chose_best is None else c.chose_best])
