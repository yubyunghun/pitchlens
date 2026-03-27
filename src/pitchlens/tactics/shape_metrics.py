"""Numbers that describe how a team plays, with and without the ball.

Per frame, from the team's visible outfield players (goalkeeper excluded):
  width            spread across the pitch (meters, widest minus narrowest)
  length           deepest to most advanced outfield player (meters)
  defensive_line   how high the back line stands: average distance of the
                   4 deepest outfield players from their own goal line (meters)
  area             area of the team's outline (convex hull, square meters):
                   smaller = more compact

Frames with fewer than `min_players` visible outfield players are skipped: a
partial camera view would make the team look narrower and shorter than it is.
Each metric is summarized as the median over the frames of a phase.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from pitchlens.calibration.pitch import Pitch
from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.possession import ALL, IN, OUT, phase_frames

METRICS = ("width", "length", "defensive_line", "area")
UNITS = {"width": "m", "length": "m", "defensive_line": "m", "area": "m2"}


@dataclass
class ShapeMetrics:
    team: int
    phase: str
    frames: int                       # frames that were measured
    values: dict[str, float]          # metric -> median (NaN if no frame qualified)


def frame_metrics(positions: np.ndarray, attacks_right: bool, pitch: Pitch = Pitch(),
                  back_line: int = 4) -> dict[str, float]:
    """Metrics of one team in one frame (outfield positions in pitch meters)."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    depth = pos[:, 0] if attacks_right else pitch.length - pos[:, 0]
    hull = cv2.convexHull(pos.astype(np.float32))
    return {
        "width": float(pos[:, 1].max() - pos[:, 1].min()),
        "length": float(depth.max() - depth.min()),
        "defensive_line": float(np.sort(depth)[:back_line].mean()),
        "area": float(cv2.contourArea(hull)),
    }


def team_metrics(players: Sequence[FramePlayers], team: int, attacks_right: bool, frames: Sequence[int],
                 exclude_ids: set[int] = frozenset(), pitch: Pitch = Pitch(),
                 min_players: int = 8) -> tuple[int, dict[str, float]]:
    """(frames measured, metric -> median) for one team over the given frames."""
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    for t in frames:
        fp = players[t]
        mine = (fp.teams == team) & np.isfinite(fp.positions).all(axis=1)
        if exclude_ids:
            mine &= ~np.isin(fp.track_ids, list(exclude_ids))
        if mine.sum() < min_players:
            continue
        for k, v in frame_metrics(fp.positions[mine], attacks_right, pitch).items():
            per_metric[k].append(v)
    n = len(per_metric["width"])
    return n, {k: (float(np.median(v)) if v else float("nan")) for k, v in per_metric.items()}


def shape_metrics(players: Sequence[FramePlayers], directions: dict[int, int], possession: np.ndarray,
                  keepers: dict[int, set[int]] | None = None, pitch: Pitch = Pitch(),
                  min_players: int = 8) -> list[ShapeMetrics]:
    """Metrics per team and phase. keepers: team -> goalkeeper track ids to leave out."""
    out = []
    for team, direction in sorted(directions.items()):
        for phase, frames in phase_frames(possession, team).items():
            n, values = team_metrics(players, team, direction > 0, frames,
                                     (keepers or {}).get(team, set()), pitch, min_players)
            out.append(ShapeMetrics(team, phase, n, values))
    return out


PHASE_LABELS = {IN: "with ball", OUT: "without ball", ALL: "overall"}


def summarize_metrics(metrics: list[ShapeMetrics], team_names: dict[int, str]) -> str:
    lines = ["Team shape (median per phase; width / length / defensive line height / area):"]
    for team in sorted({m.team for m in metrics}):
        lines.append(f"  {team_names.get(team, f'team {team}')}:")
        for m in (x for x in metrics if x.team == team):
            if m.frames == 0:
                lines.append(f"    {PHASE_LABELS[m.phase]:<13} not enough players visible")
                continue
            v = m.values
            lines.append(f"    {PHASE_LABELS[m.phase]:<13} {v['width']:5.1f} m / {v['length']:5.1f} m / "
                         f"{v['defensive_line']:5.1f} m / {v['area']:6.0f} m2   ({m.frames} frames)")
    return "\n".join(lines)


def write_metrics_csv(path: str | Path, metrics: list[ShapeMetrics]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "phase", "frames"] + [f"{k}_{UNITS[k]}" for k in METRICS])
        for m in metrics:
            w.writerow([m.team, m.phase, m.frames] +
                       ["" if np.isnan(m.values[k]) else round(m.values[k], 2) for k in METRICS])
