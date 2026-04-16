"""Run the whole offside chain over a clip and report the results.

ball detections -> ball path -> touches -> passes -> freeze at each kick
-> decision per pass. Results go to a CSV (one row per pass) and a short
printed summary.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.offside.ball_path import BallPath, build_ball_path
from pitchlens.offside.decision import (NO_DECISION, NO_OFFENCE, OFFSIDE, ONSIDE, TOO_CLOSE,
                                        OffsideDecision, decide)
from pitchlens.offside.direction import DirectionEstimate
from pitchlens.offside.freeze import freeze_pass
from pitchlens.offside.line import OffsideLine, frame_offside_lines
from pitchlens.offside.passes import FramePlayers, detect_touches, find_passes
from pitchlens.offside.position import DEFAULT_TOLERANCE_M

CSV_HEADER = ["pass_frame", "pass_time_s", "passer_track_id", "attacking_team", "receiver_track_id",
              "receive_frame", "verdict", "margin_m", "line_x_m", "line_source", "keeper_assumed",
              "uninvolved_offside_track_ids", "reason"]


def find_offside_incidents(players: list[FramePlayers], ball_raw: Sequence[np.ndarray | None],
                           calibs: Sequence[PitchCalibration | None], classes: Sequence[np.ndarray | None],
                           directions: DirectionEstimate, fps: float,
                           tolerance_m: float = DEFAULT_TOLERANCE_M
                           ) -> tuple[BallPath, list[dict[int, OffsideLine]], list[OffsideDecision]]:
    """players / ball_raw / calibs / classes: one entry per frame (ball and calib may be None)."""
    path = build_ball_path(ball_raw, fps)
    lines: list[dict[int, OffsideLine]] = []
    for t, (fp, calib) in enumerate(zip(players, calibs)):
        if calib is None:
            lines.append({})
            continue
        ball = path.positions[t] if np.isfinite(path.positions[t]).all() else None
        lines.append(frame_offside_lines(fp.positions, fp.teams, directions, calib, ball, classes[t],
                                         fp.toward_goal))
    passes = find_passes(detect_touches(path, players))
    decisions = [decide(p, freeze_pass(p, players, lines, tolerance_m)) for p in passes]
    return path, lines, decisions


def write_incidents_csv(path: str | Path, decisions: list[OffsideDecision], fps: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for d in decisions:
            kick, rec, line, check = d.pass_.kick, d.pass_.receive, d.frozen.line, d.receiver_check
            w.writerow([kick.frame, round(kick.frame / fps, 2), kick.track_id, d.frozen.attacking_team,
                        "" if rec is None or rec.track_id is None else rec.track_id,
                        "" if rec is None else rec.frame,
                        d.verdict,
                        "" if check is None else round(check.margin_m, 2),
                        "" if line is None else round(line.x, 2),
                        "" if line is None else line.source,
                        "" if line is None else line.keeper_assumed,
                        " ".join(str(t) for t in d.uninvolved_offside),
                        d.reason])


def summarize_incidents(decisions: list[OffsideDecision], fps: float) -> str:
    if not decisions:
        return "Offside: no passes detected (is the ball being detected? see README)"
    counts = Counter(d.verdict for d in decisions)
    order = [OFFSIDE, TOO_CLOSE, ONSIDE, NO_OFFENCE, NO_DECISION]
    parts = ", ".join(f"{counts[v]} {v.replace('_', ' ')}" for v in order if counts[v])
    lines = [f"Offside: {len(decisions)} passes checked: {parts}"]
    for d in decisions:
        if d.verdict in (OFFSIDE, TOO_CLOSE):
            k, r = d.pass_.kick, d.pass_.receive
            lines.append(f"  {k.frame / fps:6.1f}s  {d.verdict.upper():<9} #{k.track_id} -> #{r.track_id}: {d.reason}")
    return "\n".join(lines)


def banner_frames(decisions: list[OffsideDecision], fps: float, n_frames: int,
                  seconds: float = 2.5) -> dict[int, OffsideDecision]:
    """Which decision to announce on which frame: from the moment the ball is received."""
    out: dict[int, OffsideDecision] = {}
    for d in decisions:
        if d.verdict not in (OFFSIDE, TOO_CLOSE):
            continue
        start = d.pass_.receive.frame if d.pass_.receive is not None else d.pass_.kick.frame
        for t in range(start, min(n_frames, start + int(round(seconds * fps)))):
            out[t] = d
    return out
