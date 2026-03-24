"""Formation report: each team's shape per possession phase, as text, CSV and charts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from pitchlens.calibration.pitch import Pitch
from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.average_shape import TeamShape
from pitchlens.tactics.formations import FormationMatch, classify_formation
from pitchlens.tactics.possession import ALL, IN, OUT, phase_shapes
from pitchlens.visualization.formation_chart import draw_formation
from pitchlens.visualization.minimap import Minimap

PHASES = (IN, OUT, ALL)
PHASE_TITLES = {IN: "with the ball", OUT: "without the ball", ALL: "overall"}
CSV_HEADER = ["team", "phase", "formation", "confidence", "role", "track_ids", "x_m", "y_m", "frames"]


@dataclass
class PhaseFormation:
    shape: TeamShape
    match: FormationMatch | None       # None: too few players seen long enough to classify


def formation_report(players: Sequence[FramePlayers], directions: dict[int, int], possession: np.ndarray,
                     classes: Sequence[np.ndarray | None] | None = None, pitch: Pitch = Pitch(),
                     min_frames: int = 25) -> dict[int, dict[str, PhaseFormation]]:
    """team -> phase -> average shape + best formation."""
    shapes = phase_shapes(players, directions, possession, classes, pitch, min_frames)
    out: dict[int, dict[str, PhaseFormation]] = {}
    for team, phases in shapes.items():
        out[team] = {}
        for phase in PHASES:
            shape = phases[phase]
            match = classify_formation(shape.positions, directions[team] > 0, pitch) if shape.outfield else None
            out[team][phase] = PhaseFormation(shape, match)
    return out


def summarize_formations(report: dict[int, dict[str, PhaseFormation]], team_names: dict[int, str]) -> str:
    lines = ["Formations (average positions):"]
    for team in sorted(report):
        parts = []
        for phase in PHASES:
            pf = report[team][phase]
            if pf.match is None:
                parts.append(f"{PHASE_TITLES[phase]}: not enough data ({len(pf.shape.outfield)} players)")
            else:
                parts.append(f"{PHASE_TITLES[phase]}: {pf.match.describe()} ({pf.match.confidence:.0%})")
        lines.append(f"  {team_names.get(team, f'team {team}')}: " + "; ".join(parts))
    return "\n".join(lines)


def write_average_positions_csv(path: str | Path, report: dict[int, dict[str, PhaseFormation]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for team in sorted(report):
            for phase in PHASES:
                pf = report[team][phase]
                name = "" if pf.match is None else pf.match.name
                conf = "" if pf.match is None else round(pf.match.confidence, 3)
                rows = [(p, "" if pf.match is None else pf.match.slots[i].role)
                        for i, p in enumerate(pf.shape.outfield)]
                if pf.shape.goalkeeper is not None:
                    rows.append((pf.shape.goalkeeper, "GK"))
                for p, role in rows:
                    w.writerow([team, phase, name, conf, role, " ".join(map(str, p.track_ids)),
                                round(float(p.position[0]), 2), round(float(p.position[1]), 2), p.n_frames])


def formation_chart(team_report: dict[str, PhaseFormation], color: tuple[int, int, int], pitch: Pitch = Pitch(),
                    panel_width: int = 640) -> np.ndarray:
    """One image per team: a panel per phase, side by side."""
    mm = Minimap(pitch, width_px=panel_width, margin_m=5.0)
    panels = []
    for phase in PHASES:
        pf = team_report[phase]
        img = mm.render(np.empty((0, 2)), [])
        pos = pf.shape.positions
        if pf.match is not None:
            title = f"{PHASE_TITLES[phase]}: {pf.match.describe()} ({pf.match.confidence:.0%})"
            draw_formation(img, mm, pos, pf.match, color, title=title)
        else:
            if len(pos):
                img = mm.render(pos, [color] * len(pos))
            cv2.putText(img, f"{PHASE_TITLES[phase]}: not enough data", (10, int(mm.margin * mm.scale * 0.7)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        gk = pf.shape.goalkeeper
        if gk is not None:
            p = tuple(int(v) for v in mm.to_px(gk.position))
            r = max(5, int(round(mm.scale * 1.1)))
            cv2.circle(img, p, r, (60, 60, 60), -1, cv2.LINE_AA)
            cv2.circle(img, p, r, color, 2, cv2.LINE_AA)
            cv2.putText(img, "GK", (p[0] - 8, p[1] + r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(img)
    gap = np.full((panels[0].shape[0], 6, 3), 20, np.uint8)
    return np.hstack([panels[0], gap, panels[1], gap, panels[2]])


def save_formation_charts(directory: str | Path, report: dict[int, dict[str, PhaseFormation]],
                          team_colors: dict[int, tuple[int, int, int]], pitch: Pitch = Pitch()) -> list[Path]:
    Path(directory).mkdir(parents=True, exist_ok=True)
    saved = []
    for team in sorted(report):
        path = Path(directory) / f"team{team}_formation.png"
        cv2.imwrite(str(path), formation_chart(report[team], team_colors.get(team, (200, 200, 200)), pitch))
        saved.append(path)
    return saved
