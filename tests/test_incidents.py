import csv

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.offside.decision import OFFSIDE, ONSIDE
from pitchlens.offside.direction import ATTACKS_LEFT, ATTACKS_RIGHT, DirectionEstimate
from pitchlens.offside.incidents import (banner_frames, find_offside_incidents, summarize_incidents,
                                         write_incidents_csv)
from pitchlens.offside.passes import FramePlayers
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

FPS = 25.0
DIRECTIONS = DirectionEstimate({TEAM_A: ATTACKS_RIGHT, TEAM_B: ATTACKS_LEFT}, 1.0, 100)


def _right_half_calib():
    """Camera on the right part of the pitch (x 55-105), right goal area in view."""
    cam = cv2.getPerspectiveTransform(np.float32([[55, 0], [105, 0], [105, 68], [55, 68]]),
                                      np.float32([[100, 120], [1180, 120], [1400, 700], [-120, 700]]))
    return PitchCalibration(np.linalg.inv(cam), (1280, 720))


def _clip(forward_x, second_forward_x=None, seed=0):
    """A passer at x=70 plays a through ball to a forward; B defends the right goal (line at 92)."""
    ids = [20, 21, 22, 10, 11] + ([12] if second_forward_x is not None else [])
    teams = [TEAM_B, TEAM_B, TEAM_B, TEAM_A, TEAM_A] + ([TEAM_A] if second_forward_x is not None else [])
    pos = [[104, 34], [92, 20], [90, 50], [70, 34], [forward_x, 34]]
    if second_forward_x is not None:
        pos.append([second_forward_x, 55])
    pos = np.array(pos, float)
    passer, forward = pos[3], pos[4]
    n = int(round(np.linalg.norm(forward - passer) / 15.0 * FPS))
    ball = [passer] * 15 + [passer + (forward - passer) * k / n for k in range(1, n + 1)] + [forward] * 20
    rng = np.random.default_rng(seed)
    raw = [b + rng.normal(0, 0.1, 2) for b in ball]
    raw[30] = None                                                  # a missed detection
    players = [FramePlayers(pos, np.array(teams), np.array(ids)) for _ in raw]
    calibs = [_right_half_calib()] * len(raw)
    return players, raw, calibs, [None] * len(raw)


def test_offside_through_ball_found():
    players, raw, calibs, classes = _clip(forward_x=95.0, second_forward_x=85.0)
    path, lines, decisions = find_offside_incidents(players, raw, calibs, classes, DIRECTIONS, FPS)
    assert len(path) == len(raw) and len(lines) == len(raw)
    assert lines[10][TEAM_B].x == 92.0 and not lines[10][TEAM_B].keeper_assumed
    first = decisions[0]
    assert first.verdict == OFFSIDE and first.pass_.receive.track_id == 11
    assert abs(first.receiver_check.margin_m - 3.0) < 0.5


def test_onside_with_uncalibrated_frames():
    players, raw, calibs, classes = _clip(forward_x=88.0)
    calibs = list(calibs)
    for t in range(40, 50):                                         # e.g. no keyframe reaches these
        calibs[t] = None
    _, lines, decisions = find_offside_incidents(players, raw, calibs, classes, DIRECTIONS, FPS)
    assert lines[45] == {}
    assert decisions[0].verdict == ONSIDE


def test_csv_summary_and_banners(tmp_path):
    players, raw, calibs, classes = _clip(forward_x=95.0)
    _, _, decisions = find_offside_incidents(players, raw, calibs, classes, DIRECTIONS, FPS)
    out = tmp_path / "incidents.csv"
    write_incidents_csv(out, decisions, FPS)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(decisions)
    assert rows[0]["verdict"] == OFFSIDE and rows[0]["passer_track_id"] == "10"
    assert rows[0]["receiver_track_id"] == "11" and rows[0]["line_x_m"] == "92.0"
    assert abs(float(rows[0]["margin_m"]) - 3.0) < 0.5

    text = summarize_incidents(decisions, FPS)
    assert "1 offside" in text and "#10 -> #11" in text
    assert "no passes detected" in summarize_incidents([], FPS)

    banners = banner_frames(decisions, FPS, n_frames=len(raw))
    start = decisions[0].pass_.receive.frame
    assert start in banners and banners[start] is decisions[0]
    assert (start - 1) not in banners
    assert max(banners) == len(raw) - 1 or max(banners) == start + int(2.5 * FPS) - 1


def test_banner_drawn_at_the_top():
    from pitchlens.visualization.offside_overlay import draw_decision_banner
    players, raw, calibs, classes = _clip(forward_x=95.0)
    _, _, decisions = find_offside_incidents(players, raw, calibs, classes, DIRECTIONS, FPS)
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_decision_banner(frame.copy(), decisions[0], FPS)
    changed = (out != frame).any(axis=2)
    assert changed[:150].any() and not changed[200:].any()      # only the top strip
    assert changed[:, 640].any()                                # centered


def test_pipeline_records_become_offside_inputs(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import supervision as sv
    import test_pipeline as tp
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.pipeline import analyze, offside_inputs, solve
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
    players, ball_raw, frame_calibs, classes = offside_inputs(records, calibs)
    assert len(players) == len(ball_raw) == len(frame_calibs) == len(classes) == len(records)
    assert all(b is None for b in ball_raw)                      # the fake detector finds no ball
    t = 10
    assert len(players[t].positions) == len(records[t].players)
    assert np.isfinite(players[t].positions).all()
    assert list(players[t].track_ids) == list(records[t].players.tracker_id)
    # no ball -> no passes, and the pipeline still runs
    from pitchlens.offside.incidents import find_offside_incidents as find
    _, _, decisions = find(players, ball_raw, frame_calibs, classes, DIRECTIONS, 25.0)
    assert decisions == []

