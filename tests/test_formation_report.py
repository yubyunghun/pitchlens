import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from pitchlens.offside.passes import FramePlayers, Touch
from pitchlens.tactics.possession import ALL, IN, OUT, possession_by_frame
from pitchlens.tactics.report import (formation_chart, formation_report, save_formation_charts,
                                      summarize_formations, write_average_positions_csv)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_average_shape import _layout  # noqa: E402

FPS = 25.0
DIRS = {TEAM_A: 1, TEAM_B: -1}


def _game(n=300, seed=0):
    """A: 4-3-3 with the ball, 4-1-4-1 without (with a keeper). B: 4-4-2, keeper never tracked."""
    rng = np.random.default_rng(seed)
    a_in, a_out = _layout("4-3-3", (30, 70)), _layout("4-1-4-1", (15, 50))
    b_lay = _layout("4-4-2", (20, 55))
    b = np.c_[105 - b_lay[:, 0], 68 - b_lay[:, 1]]
    touches = [Touch(t, 1, TEAM_A if (t // 50) % 2 == 0 else TEAM_B, None, np.zeros(2), 10.0)
               for t in range(0, n, 25)]
    possession = possession_by_frame(touches, n, FPS)
    frames = []
    for t in range(n):
        a = (a_in if possession[t] == TEAM_A else a_out) + rng.normal(0, 1.5, (10, 2))
        pos = np.vstack([a, [[3.0, 34.0]], b + rng.normal(0, 1.5, (10, 2))])
        teams = np.array([TEAM_A] * 11 + [TEAM_B] * 10)
        ids = np.r_[np.arange(1, 11), [0], np.arange(21, 31)]
        frames.append(FramePlayers(pos, teams, ids))
    return frames, possession


def test_report_per_team_and_phase():
    frames, possession = _game()
    report = formation_report(frames, DIRS, possession)
    assert report[TEAM_A][IN].match.name == "4-3-3"
    assert report[TEAM_A][OUT].match.name == "4-1-4-1"
    assert report[TEAM_B][ALL].match.name == "4-4-2"
    assert report[TEAM_A][ALL].shape.goalkeeper.track_id == 0
    assert report[TEAM_B][ALL].shape.goalkeeper is None

    text = summarize_formations(report, {TEAM_A: "team 0 (#d22828)"})
    assert "team 0 (#d22828): with the ball: 4-3-3" in text
    assert "without the ball: 4-1-4-1" in text and "team 1:" in text


def test_not_enough_data():
    frames, possession = _game(n=40)
    report = formation_report(frames, DIRS, possession, min_frames=25)
    assert report[TEAM_A][IN].match is not None             # A had the ball all 40 frames
    assert report[TEAM_B][IN].match is None                 # B never had it
    assert "not enough data" in summarize_formations(report, {})
    chart = formation_chart(report[TEAM_B], (210, 90, 30))
    assert chart.ndim == 3 and chart.shape[1] > chart.shape[0] * 2


def test_csv_and_charts(tmp_path):
    frames, possession = _game()
    report = formation_report(frames, DIRS, possession)
    out = tmp_path / "average_positions.csv"
    write_average_positions_csv(out, report)
    rows = list(csv.DictReader(out.open()))
    a_in = [r for r in rows if r["team"] == "0" and r["phase"] == IN]
    assert len(a_in) == 11 and {r["formation"] for r in a_in} == {"4-3-3"}
    assert sorted(r["role"] for r in a_in).count("CB") == 2 and "GK" in {r["role"] for r in a_in}
    assert len([r for r in rows if r["team"] == "1" and r["phase"] == ALL]) == 10    # no keeper row

    saved = save_formation_charts(tmp_path / "formations", report, {TEAM_A: (40, 40, 210), TEAM_B: (210, 90, 30)})
    assert [p.name for p in saved] == ["team0_formation.png", "team1_formation.png"]
    img = cv2.imread(str(saved[0]))
    assert img is not None and img.shape[1] > 3 * 600


def test_pipeline_formation_step(tmp_path):
    """The pipeline's formation step runs on real pipeline records (fake detector, no ball)."""
    import supervision as sv
    import test_pipeline as tp
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.calibration.pitch import Pitch
    from pitchlens.offside.ball_path import build_ball_path
    from pitchlens.offside.direction import DirectionEstimate
    from pitchlens.pipeline import analyze, formations_from_inputs, offside_inputs, solve
    from pitchlens.tracking.team_assigner import TeamAssigner

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    assigner.collect(first, tp._boxes(0))
    assigner.collect(first, tp._boxes(0))
    assigner.fit()
    records = analyze(str(video), tp.FakeDetector(), assigner, fps=25)
    calibs = solve(records, CalibrationSet({0: tp._calib_for(tp._cam(0))}), tp.SIZE)
    inputs = offside_inputs(records, calibs)
    path = build_ball_path(inputs[1], 25.0)
    report = formations_from_inputs(inputs, path, DirectionEstimate(DIRS, 1.0, 1), 25.0, Pitch(), min_seconds=1.0)
    assert set(report) == {TEAM_A, TEAM_B}
    for team in report:
        assert len(report[team][ALL].shape.outfield) == 3       # 3 players per team in the clip
        assert report[team][ALL].match is None                  # too few for a formation
        assert report[team][IN].shape.outfield == []            # no ball -> possession unknown
