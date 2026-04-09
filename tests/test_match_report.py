import sys
from pathlib import Path

import numpy as np

from pitchlens.report.match_report import ReportData, TeamInfo, render_html, write_report
from pitchlens.tactics.timeline import Segment
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))


def _full_data():
    import test_formation_report as tfr
    import test_incidents as ti
    from pitchlens.offside.incidents import find_offside_incidents
    from pitchlens.tactics.pass_choice import TeamPassing
    from pitchlens.tactics.report import formation_chart, formation_report
    from pitchlens.tactics.shape_metrics import shape_metrics

    frames, possession = tfr._game()
    formations = formation_report(frames, tfr.DIRS, possession)
    players, raw, calibs, classes = ti._clip(forward_x=95.0)
    _, _, decisions = find_offside_incidents(players, raw, calibs, classes, ti.DIRECTIONS, ti.FPS)
    teams = {TEAM_A: TeamInfo("team 0", (40, 40, 210), "left -> right"),
             TEAM_B: TeamInfo("team 1", (210, 90, 30), "right -> left")}
    return ReportData(
        video="data/raw/<match>.mp4", fps=25.0, frames=len(frames), teams=teams,
        calibrated_frames=len(frames) - 10, keyframes=2, cuts=[100], direction_confidence=0.95,
        formations=formations,
        formation_charts={t: formation_chart(formations[t], teams[t].color_bgr) for t in formations},
        timeline=[Segment(TEAM_A, 0, 150, "4-3-3", 0.8), Segment(TEAM_A, 150, 300, "unclear", 0.0),
                  Segment(TEAM_B, 0, 300, "4-4-2", 0.9)],
        metrics=shape_metrics(frames, tfr.DIRS, possession),
        passing=[TeamPassing(TEAM_A, 12, 0.75, 0.5, 0.25, 0.4, 2.1, 1.3), TeamPassing(TEAM_B, 0, None, None,
                                                                                         None, None, None, None)],
        offside=decisions,
        offside_images={0: np.zeros((720, 1280, 3), np.uint8)},
    )


def test_full_report_has_every_section(tmp_path):
    d = _full_data()
    page = render_html(d)
    for heading in ("Overview", "Formations", "Formation over time", "Team shape", "Passing", "Offside", "Limits"):
        assert f"<h2>{heading}</h2>" in page
    assert "4-3-3" in page and "4-1-4-1" in page and "4-4-2" in page        # A in / out, B
    assert "#d22828" in page                                                # team colour as hex (from BGR)
    assert "left -&gt; right (95% confidence)" in page
    assert "0:04.0" in page                                                 # the camera cut at frame 100
    assert page.count("data:image/png;base64,") == 3                       # 2 formation charts + 1 still
    assert "&lt;match&gt;.mp4" in page and "<match>" not in page            # file names are escaped
    assert "class='bad'>offside" in page                                   # the offside call is highlighted
    assert "75%" in page and "–" in page                                    # passing shares, missing as dash
    assert "class='seg unclear'" in page
    out = write_report(tmp_path / "out" / "report.html", d)
    assert out.read_text(encoding="utf-8") == page


def test_report_without_analysis_explains_what_is_missing():
    page = render_html(ReportData(video="clip.mp4", fps=25.0, frames=250))
    assert "Not available: needs calibrated frames." in page
    assert "No passes detected" in page
    assert "Formation over time" not in page and "<h2>Passing</h2>" not in page
    assert "0:10.0 (250 frames at 25 fps)" in page and "none detected" in page


def test_large_images_are_shrunk():
    d = ReportData(video="c.mp4", fps=25.0, frames=10, offside=_full_data().offside,
                   offside_images={0: np.zeros((2160, 3840, 3), np.uint8)})
    small = ReportData(video="c.mp4", fps=25.0, frames=10, offside=d.offside,
                       offside_images={0: np.zeros((720, 1280, 3), np.uint8)})
    assert abs(len(render_html(d)) - len(render_html(small))) < 2000      # 4K still stored at 1280 wide


def test_pipeline_collects_report_data(tmp_path):
    """build_report_data turns a run's results into ReportData, reading freeze-frames back from disk."""
    import cv2
    import test_formation_report as tfr
    import test_incidents as ti
    from pitchlens.offside.direction import DirectionEstimate
    from pitchlens.offside.incidents import find_offside_incidents
    from pitchlens.pipeline import build_report_data, freeze_frame_name
    from pitchlens.tactics.report import formation_report

    frames, possession = tfr._game()
    formations = formation_report(frames, tfr.DIRS, possession)
    players, raw, calibs, classes = ti._clip(forward_x=95.0)
    _, _, decisions = find_offside_incidents(players, raw, calibs, classes, ti.DIRECTIONS, ti.FPS)
    still_dir = tmp_path / "incidents"
    still_dir.mkdir()
    cv2.imwrite(str(still_dir / freeze_frame_name(decisions[0], 25.0)), np.full((90, 160, 3), 128, np.uint8))
    data = build_report_data("clip.mp4", 25.0, 300, {TEAM_A: (40, 40, 210), TEAM_B: (210, 90, 30)},
                             DirectionEstimate({TEAM_A: 1, TEAM_B: -1}, 0.9, 50), [object()] * 250 + [None] * 50,
                             2, [120], formations, [], [], [], decisions, str(still_dir))
    assert data.calibrated_frames == 250 and data.keyframes == 2 and data.cuts == [120]
    assert data.teams[TEAM_A].attacks == "left -> right" and data.teams[TEAM_B].color_hex == "#1e5ad2"
    assert set(data.formation_charts) == {TEAM_A, TEAM_B}
    assert list(data.offside_images) == [0] and data.offside_images[0].shape == (90, 160, 3)
    assert "data:image/png;base64," in render_html(data)
    # nothing analysed (no calibration): still a valid report
    empty = build_report_data("clip.mp4", 25.0, 300, {0: (1, 2, 3)}, None, [None] * 300, 0, [], None, [], [], [], [])
    assert empty.teams[0].attacks is None and "Not available" in render_html(empty)
