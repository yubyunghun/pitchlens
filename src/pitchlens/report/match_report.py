"""One self-contained HTML match report with everything PitchLens found.

All images are embedded (base64 PNG), so the single file can be emailed or
opened anywhere. Every section is optional: whatever the run produced is shown,
and missing parts (e.g. no calibration -> no formations) are explained instead.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pitchlens.offside.decision import NO_DECISION, NO_OFFENCE, OFFSIDE, ONSIDE, TOO_CLOSE, OffsideDecision
from pitchlens.tactics.pass_choice import TeamPassing
from pitchlens.tactics.possession import ALL, IN, OUT
from pitchlens.tactics.shape_metrics import ShapeMetrics
from pitchlens.tactics.timeline import Segment


@dataclass
class TeamInfo:
    name: str                   # e.g. "team 0"
    color_bgr: tuple[int, int, int]
    attacks: str | None = None  # e.g. "left -> right"

    @property
    def color_hex(self) -> str:
        b, g, r = self.color_bgr
        return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class ReportData:
    video: str
    fps: float
    frames: int
    teams: dict[int, TeamInfo] = field(default_factory=dict)
    calibrated_frames: int = 0
    keyframes: int = 0
    cuts: list[int] = field(default_factory=list)
    direction_confidence: float | None = None
    formations: dict | None = None              # tactics/report.formation_report() output
    formation_charts: dict[int, np.ndarray] = field(default_factory=dict)   # team -> chart image
    timeline: list[Segment] = field(default_factory=list)
    metrics: list[ShapeMetrics] = field(default_factory=list)
    passing: list[TeamPassing] = field(default_factory=list)
    offside: list[OffsideDecision] = field(default_factory=list)
    offside_images: dict[int, np.ndarray] = field(default_factory=dict)     # index into offside -> still

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def _esc(x) -> str:
    return html.escape(str(x))


def _img(image: np.ndarray, alt: str, max_width: int = 1280) -> str:
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        return ""
    data = base64.b64encode(png.tobytes()).decode("ascii")
    return f'<img alt="{_esc(alt)}" src="data:image/png;base64,{data}">'


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def _team_label(data: ReportData, team: int) -> str:
    info = data.teams.get(team)
    if info is None:
        return f"team {team}"
    return f'<span class="dot" style="background:{info.color_hex}"></span>{_esc(info.name)}'


def _section_overview(d: ReportData) -> str:
    rows = [("Clip", _esc(d.video)), ("Length", f"{_clock(d.duration_s)} ({d.frames} frames at {d.fps:.0f} fps)"),
            ("Calibrated frames", f"{d.calibrated_frames}/{d.frames} from {d.keyframes} keyframe(s)"),
            ("Camera cuts", ", ".join(_clock(c / d.fps) for c in d.cuts) if d.cuts else "none detected")]
    for team, info in sorted(d.teams.items()):
        if info.attacks:
            conf = "" if d.direction_confidence is None else f" ({d.direction_confidence:.0%} confidence)"
            rows.append((f"{_team_label(d, team)} attacks", f"{_esc(info.attacks)}{conf}"))
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<h2>Overview</h2><table class='kv'>{body}</table>"


PHASES = ((IN, "With the ball"), (OUT, "Without the ball"), (ALL, "Overall"))


def _section_formations(d: ReportData) -> str:
    if not d.formations:
        return "<h2>Formations</h2><p class='na'>Not available: needs calibrated frames.</p>"
    head = "".join(f"<th>{label}</th>" for _, label in PHASES)
    rows = []
    for team in sorted(d.formations):
        cells = []
        for phase, _ in PHASES:
            pf = d.formations[team][phase]
            if pf.match is None:
                cells.append(f"<td class='na'>not enough data ({len(pf.shape.outfield)} players)</td>")
            else:
                cells.append(f"<td><b>{_esc(pf.match.describe())}</b> <small>{pf.match.confidence:.0%}</small></td>")
        rows.append(f"<tr><th>{_team_label(d, team)}</th>{''.join(cells)}</tr>")
    charts = "".join(f"<figure>{_img(img, f'formation chart team {t}')}"
                     f"<figcaption>{_team_label(d, t)}: average positions</figcaption></figure>"
                     for t, img in sorted(d.formation_charts.items()))
    return (f"<h2>Formations</h2><table><tr><th></th>{head}</tr>{''.join(rows)}</table>"
            f"<p class='note'>Average positions per phase matched to formation templates; the "
            f"percentage is how clearly the best template beats the runner-up.</p>{charts}")


def _section_timeline(d: ReportData) -> str:
    if not d.timeline:
        return ""
    total = max(s.end for s in d.timeline) or 1
    bars = []
    for team in sorted({s.team for s in d.timeline}):
        blocks = "".join(
            f"<div class='seg{' unclear' if s.formation == 'unclear' else ''}' "
            f"style='width:{100 * (s.end - s.start) / total:.2f}%' "
            f"title='{_clock(s.start / d.fps)}-{_clock(s.end / d.fps)}'>{_esc(s.formation)}</div>"
            for s in d.timeline if s.team == team)
        bars.append(f"<div class='tl-row'><div class='tl-name'>{_team_label(d, team)}</div>"
                    f"<div class='tl-bar'>{blocks}</div></div>")
    return f"<h2>Formation over time</h2>{''.join(bars)}<p class='note'>0:00 to {_clock(total / d.fps)}</p>"


def _section_metrics(d: ReportData) -> str:
    if not d.metrics:
        return ""
    labels = dict(PHASES)
    rows = []
    for m in d.metrics:
        v = m.values
        if m.frames == 0:
            cells = "<td colspan='4' class='na'>not enough players visible</td>"
        else:
            cells = (f"<td>{v['width']:.1f} m</td><td>{v['length']:.1f} m</td>"
                     f"<td>{v['defensive_line']:.1f} m</td><td>{v['area']:.0f} m²</td>")
        rows.append(f"<tr><th>{_team_label(d, m.team)}</th><td>{labels.get(m.phase, m.phase)}</td>{cells}</tr>")
    return ("<h2>Team shape</h2><table><tr><th>Team</th><th>Phase</th><th>Width</th><th>Length</th>"
            "<th>Defensive line</th><th>Area</th></tr>" + "".join(rows) + "</table>"
            "<p class='note'>Medians over frames with at least 8 outfield players visible. Defensive line = "
            "the back four's distance from their own goal.</p>")


def _section_passing(d: ReportData) -> str:
    if not d.passing:
        return ""
    rows = []
    for s in d.passing:
        lanes = "–" if s.open_lanes_on_ball is None else f"{s.open_lanes_on_ball:.1f}"
        allowed = "–" if s.open_forward_lanes_allowed is None else f"{s.open_forward_lanes_allowed:.1f}"
        rows.append(f"<tr><th>{_team_label(d, s.team)}</th><td>{s.passes}</td><td>{_pct(s.completed)}</td>"
                    f"<td>{_pct(s.forward)}</td><td>{_pct(s.line_breaking)}</td><td>{_pct(s.chose_best)}</td>"
                    f"<td>{lanes}</td><td>{allowed}</td></tr>")
    return ("<h2>Passing</h2><table><tr><th>Team</th><th>Passes</th><th>Completed</th><th>Forward</th>"
            "<th>Line-breaking</th><th>Best option taken</th><th>Open lanes on the ball</th>"
            "<th>Open forward lanes allowed</th></tr>" + "".join(rows) + "</table>")


VERDICT_CLASS = {OFFSIDE: "bad", TOO_CLOSE: "warn", ONSIDE: "ok", NO_OFFENCE: "ok", NO_DECISION: "na"}


def _section_offside(d: ReportData) -> str:
    if not d.offside:
        return "<h2>Offside</h2><p class='na'>No passes detected (the ball may not have been detected).</p>"
    rows = []
    for i, dec in enumerate(d.offside):
        k, r = dec.pass_.kick, dec.pass_.receive
        rows.append(f"<tr><td>{_clock(k.frame / d.fps)}</td><td>{_team_label(d, dec.frozen.attacking_team)}</td>"
                    f"<td>#{k.track_id} → {'#' + str(r.track_id) if r and r.track_id is not None else '?'}</td>"
                    f"<td class='{VERDICT_CLASS.get(dec.verdict, '')}'>{_esc(dec.verdict.replace('_', ' '))}</td>"
                    f"<td>{_esc(dec.reason)}</td></tr>")
    stills = "".join(f"<figure>{_img(img, f'offside call {i}')}<figcaption>"
                     f"{_clock(d.offside[i].pass_.kick.frame / d.fps)}: {_esc(d.offside[i].reason)}</figcaption></figure>"
                     for i, img in sorted(d.offside_images.items()))
    return ("<h2>Offside</h2><table><tr><th>Time</th><th>Attacking</th><th>Pass</th><th>Verdict</th>"
            "<th>Why</th></tr>" + "".join(rows) + "</table>" + stills)


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;
     padding:0 16px;color:#1d2327;background:#fafafa}
h1{margin-bottom:4px} h2{margin-top:36px;border-bottom:2px solid #2e7d32;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:8px 0;background:#fff}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}
table.kv th{width:220px;background:#f2f4f3}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;border:1px solid #333}
.na{color:#888;font-style:italic} .note{color:#666;font-size:13px}
.bad{background:#fde2e1;font-weight:600} .warn{background:#fff3cd} .ok{background:#e6f4ea}
figure{margin:16px 0} figure img{max-width:100%;border:1px solid #ccc} figcaption{font-size:13px;color:#555}
.tl-row{display:flex;align-items:center;margin:6px 0} .tl-name{width:160px}
.tl-bar{flex:1;display:flex;height:26px;border:1px solid #ccc;background:#fff}
.seg{background:#c8e6c9;border-right:1px solid #fff;font-size:12px;line-height:26px;text-align:center;
     overflow:hidden;white-space:nowrap} .seg.unclear{background:#eee;color:#888}
"""


def render_html(d: ReportData, title: str = "PitchLens match report") -> str:
    sections = [_section_overview(d), _section_formations(d), _section_timeline(d), _section_metrics(d),
                _section_passing(d), _section_offside(d)]
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>"
            f"<h1>{_esc(title)}</h1><p class='note'>{_esc(Path(d.video).name)}</p>"
            + "".join(s for s in sections if s) +
            "<h2>Limits</h2><p class='note'>Positions come from one broadcast camera and a pitch calibration; "
            "offside is measured at the players' feet unless body-part detection is enabled; passing lanes are a "
            "ground-pass geometry model. Treat numbers as estimates.</p></body></html>")


def write_report(path: str | Path, d: ReportData, title: str = "PitchLens match report") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(d, title), encoding="utf-8")
    return path
