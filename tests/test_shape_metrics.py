import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from pitchlens.offside.passes import FramePlayers, Touch
from pitchlens.tactics.possession import ALL, IN, OUT, possession_by_frame
from pitchlens.tactics.shape_metrics import (frame_metrics, shape_metrics, summarize_metrics, team_metrics,
                                             write_metrics_csv)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_average_shape import _layout  # noqa: E402

FPS = 25.0


def test_frame_metrics_on_a_known_block():
    # a 4-4-2 laid out in a 40 m deep, 50 m wide block, back line 20 m from goal
    team = _layout("4-4-2", (20.0, 60.0), width=(9.0, 59.0))
    m = frame_metrics(team, attacks_right=True)
    assert m["width"] == pytest.approx(0.8 * 50)                    # widest players at 0.1 / 0.9
    assert m["length"] == pytest.approx((0.75 - 0.15) * 40)
    assert m["defensive_line"] == pytest.approx(20 + 0.15 * 40)     # back four at depth 0.15
    assert 0 < m["area"] < m["width"] * m["length"]
    mirrored = np.c_[105 - team[:, 0], 68 - team[:, 1]]             # same block attacking left
    assert frame_metrics(mirrored, attacks_right=False) == pytest.approx(m)


def _game(n=300, seed=0):
    """A presses high and wide with the ball; drops into a compact low block without it."""
    rng = np.random.default_rng(seed)
    a_in = _layout("4-3-3", (35, 80), width=(4, 64))
    a_out = _layout("4-4-2", (10, 40), width=(14, 54))
    b_lay = _layout("4-4-2", (20, 55))
    b = np.c_[105 - b_lay[:, 0], 68 - b_lay[:, 1]]
    touches = [Touch(t, 1, TEAM_A if (t // 50) % 2 == 0 else TEAM_B, None, np.zeros(2), 10.0)
               for t in range(0, n, 25)]
    possession = possession_by_frame(touches, n, FPS)
    frames = []
    for t in range(n):
        a = (a_in if possession[t] == TEAM_A else a_out) + rng.normal(0, 1.0, (10, 2))
        pos = np.vstack([a, [[3.0, 34.0]], b + rng.normal(0, 1.0, (10, 2))])
        frames.append(FramePlayers(pos, np.array([TEAM_A] * 11 + [TEAM_B] * 10),
                                   np.r_[np.arange(1, 11), [0], np.arange(21, 31)]))
    return frames, possession


def test_with_and_without_the_ball():
    frames, possession = _game()
    metrics = {(m.team, m.phase): m for m in shape_metrics(frames, {TEAM_A: 1, TEAM_B: -1}, possession,
                                                             keepers={TEAM_A: {0}})}
    a_in, a_out = metrics[(TEAM_A, IN)].values, metrics[(TEAM_A, OUT)].values
    assert a_in["defensive_line"] > a_out["defensive_line"] + 15      # higher line with the ball
    assert a_in["width"] > a_out["width"] + 10                        # wider with the ball
    assert a_in["area"] > 1.5 * a_out["area"]                         # compact without it
    assert metrics[(TEAM_A, ALL)].frames == 300
    # B's shape doesn't change
    assert metrics[(TEAM_B, IN)].values["width"] == pytest.approx(metrics[(TEAM_B, OUT)].values["width"], abs=2)


def test_keeper_excluded_and_partial_views_skipped():
    frames, possession = _game(n=50)
    all_frames = range(50)
    _, with_keeper = team_metrics(frames, TEAM_A, True, all_frames)
    _, without = team_metrics(frames, TEAM_A, True, all_frames, exclude_ids={0})
    assert with_keeper["defensive_line"] < without["defensive_line"]   # keeper drags the line back
    for fp in frames:
        fp.positions[:4] = np.nan                                      # only 6 outfield players visible
    n, values = team_metrics(frames, TEAM_A, True, all_frames, exclude_ids={0})
    assert n == 0 and np.isnan(values["width"])


def test_summary_and_csv(tmp_path):
    frames, possession = _game()
    metrics = shape_metrics(frames, {TEAM_A: 1, TEAM_B: -1}, possession, keepers={TEAM_A: {0}})
    text = summarize_metrics(metrics, {TEAM_A: "team 0 (red)"})
    assert "team 0 (red):" in text and "with ball" in text and "without ball" in text
    out = tmp_path / "shape_metrics.csv"
    write_metrics_csv(out, metrics)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
    assert set(rows[0]) == {"team", "phase", "frames", "width_m", "length_m", "defensive_line_m", "area_m2"}


def test_pipeline_metrics_step_uses_report_keepers():
    from pitchlens.calibration.pitch import Pitch
    from pitchlens.offside.ball_path import build_ball_path
    from pitchlens.offside.direction import DirectionEstimate
    from pitchlens.pipeline import metrics_from_inputs
    from pitchlens.tactics.report import formation_report

    frames, possession = _game()
    report = formation_report(frames, {TEAM_A: 1, TEAM_B: -1}, possession)
    assert report[TEAM_A][ALL].shape.goalkeeper.track_ids == [0]
    inputs = (frames, [None] * len(frames), [None] * len(frames), [None] * len(frames))
    path = build_ball_path(inputs[1], FPS)                           # no ball: possession unknown
    metrics = {(m.team, m.phase): m for m in metrics_from_inputs(
        inputs, path, DirectionEstimate({TEAM_A: 1, TEAM_B: -1}, 1.0, 1), FPS, Pitch(), report)}
    assert metrics[(TEAM_A, IN)].frames == 0 and metrics[(TEAM_A, ALL)].frames == 300
    # the keeper (id 0, at x=3) was left out: the back line is far above 3 m
    assert metrics[(TEAM_A, ALL)].values["defensive_line"] > 15
