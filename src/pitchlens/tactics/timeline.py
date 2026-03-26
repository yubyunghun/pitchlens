"""How a team's formation changes over the clip.

A window (default 15 s) slides along the clip in steps (default 5 s). In each
window every team's average shape is classified (tactics/formations.py);
consecutive windows with the same formation are merged into segments, e.g.
"0:00-0:35 4-4-2, 0:35-1:10 3-5-2". Windows where the formation isn't clear
(too few players seen, or low confidence) are reported as unclear instead of
guessed.

Windows use all frames, with and without the ball, so a segment shows the
shape the team spent most of that time in.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pitchlens.calibration.pitch import Pitch
from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.average_shape import team_shapes
from pitchlens.tactics.formations import classify_formation

UNCLEAR = "unclear"


@dataclass
class WindowFormation:
    team: int
    start: int                 # first frame of the window
    end: int                   # one past the last frame
    formation: str             # template name or UNCLEAR
    confidence: float
    players: int               # outfield players with enough frames in the window


@dataclass
class Segment:
    team: int
    start: int
    end: int
    formation: str
    confidence: float          # mean over the merged windows


def window_formations(players: Sequence[FramePlayers], directions: dict[int, int], fps: float,
                      classes: Sequence[np.ndarray | None] | None = None, pitch: Pitch = Pitch(),
                      window_s: float = 15.0, step_s: float = 5.0, min_seconds: float = 3.0,
                      min_confidence: float = 0.2) -> list[WindowFormation]:
    n = len(players)
    window = max(1, int(round(window_s * fps)))
    step = max(1, int(round(step_s * fps)))
    starts = list(range(0, max(1, n - window + 1), step)) or [0]
    out = []
    for s in starts:
        e = min(n, s + window)
        shapes = team_shapes(players, directions, classes, range(s, e), pitch,
                             min_frames=int(round(min_seconds * fps)))
        for team, shape in shapes.items():
            match = classify_formation(shape.positions, directions[team] > 0, pitch) if shape.outfield else None
            if match is None or match.confidence < min_confidence:
                out.append(WindowFormation(team, s, e, UNCLEAR, 0.0 if match is None else match.confidence,
                                           len(shape.outfield)))
            else:
                out.append(WindowFormation(team, s, e, match.name, match.confidence, len(shape.outfield)))
    return out


def merge_segments(windows: list[WindowFormation]) -> list[Segment]:
    """Consecutive windows of a team with the same formation become one segment.

    Overlapping windows are split at the midpoint of their overlap, so segments don't overlap.
    """
    out: list[Segment] = []
    for team in sorted({w.team for w in windows}):
        mine = sorted((w for w in windows if w.team == team), key=lambda w: w.start)
        current: Segment | None = None
        confs: list[float] = []
        for k, w in enumerate(mine):
            start = w.start
            if k > 0 and mine[k - 1].end > w.start:       # overlaps the previous window:
                start = (mine[k - 1].end + w.start) // 2  # split at the middle of the overlap
            if current is not None and current.formation == w.formation:
                current.end = w.end
                confs.append(w.confidence)
                continue
            if current is not None:
                current.end = start
                current.confidence = float(np.mean(confs))
                out.append(current)
            current, confs = Segment(team, start, w.end, w.formation, w.confidence), [w.confidence]
        if current is not None:
            current.confidence = float(np.mean(confs))
            out.append(current)
    return out


def _clock(frame: int, fps: float) -> str:
    s = frame / fps
    return f"{int(s // 60)}:{s % 60:04.1f}"


def summarize_timeline(segments: list[Segment], fps: float, team_names: dict[int, str]) -> str:
    lines = ["Formation over time:"]
    for team in sorted({s.team for s in segments}):
        parts = [f"{_clock(s.start, fps)}-{_clock(s.end, fps)} {s.formation}" for s in segments if s.team == team]
        lines.append(f"  {team_names.get(team, f'team {team}')}: " + ", ".join(parts))
    return "\n".join(lines)


def write_timeline_csv(path: str | Path, segments: list[Segment], fps: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "start_s", "end_s", "formation", "confidence"])
        for s in segments:
            w.writerow([s.team, round(s.start / fps, 2), round(s.end / fps, 2), s.formation, round(s.confidence, 3)])
