import numpy as np
import pytest

from pitchlens.offside.line import offside_line
from pitchlens.offside.position import (ONSIDE, OFFSIDE, TOO_CLOSE, check_against_line,
                                        check_frame)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B, UNKNOWN


def _scene(attackers_x, defenders_x=(0.8, 12.0, 14.0, 20.0)):
    """Team A defends the left goal; B's attackers at the given x."""
    xs = list(defenders_x) + list(attackers_x)
    pos = np.array([[x, 10.0 + 3 * i] for i, x in enumerate(xs)], float)
    teams = np.array([TEAM_A] * len(defenders_x) + [TEAM_B] * len(attackers_x))
    return pos, teams


def test_statuses_and_margins():
    pos, teams = _scene([10.0, 11.85, 12.0, 12.2, 15.0])
    line = offside_line(pos, teams, TEAM_A, True)               # second-last defender at x = 12
    checks = check_against_line(pos, teams, line)
    got = {c.player_index: (c.status, round(c.margin_m, 2)) for c in checks}
    assert got == {4: (OFFSIDE, 2.0), 5: (TOO_CLOSE, 0.15), 6: (TOO_CLOSE, 0.0),
                   7: (TOO_CLOSE, -0.2), 8: (ONSIDE, -3.0)}
    assert all(c.defending_team == TEAM_A for c in checks)


def test_tolerance_is_adjustable():
    pos, teams = _scene([11.85])
    line = offside_line(pos, teams, TEAM_A, True)
    assert check_against_line(pos, teams, line, tolerance_m=0.1)[0].status == OFFSIDE
    assert check_against_line(pos, teams, line, tolerance_m=0.0)[0].margin_m == pytest.approx(0.15)


def test_only_attackers_checked_and_unknown_positions_skipped():
    pos, teams = _scene([10.0, np.nan])
    pos = np.vstack([pos, [5.0, 30.0]])
    teams = np.append(teams, UNKNOWN)                           # referee beyond the line: ignored
    line = offside_line(pos, teams, TEAM_A, True)
    checks = check_against_line(pos, teams, line)
    assert [c.player_index for c in checks] == [4]


def test_right_goal_is_mirrored():
    # A defends the right goal: second-last defender at x = 93
    pos = np.array([[104, 34], [93, 20], [90, 40], [95.5, 30], [91, 50]], float)
    teams = np.array([TEAM_A, TEAM_A, TEAM_A, TEAM_B, TEAM_B])
    line = offside_line(pos, teams, TEAM_A, False)
    assert line.x == 93
    got = {c.player_index: c.status for c in check_against_line(pos, teams, line)}
    assert got == {3: OFFSIDE, 4: ONSIDE}


def test_ball_and_halfway_rules_carry_through():
    # ball ahead of the defenders: an attacker behind the ball can't be offside
    pos, teams = _scene([9.0])
    line = offside_line(pos, teams, TEAM_A, True, ball=np.array([8.0, 30.0]))
    assert check_against_line(pos, teams, line)[0].status == ONSIDE
    # back line pushed past halfway: an attacker in their own half is onside
    pos, teams = _scene([56.0], defenders_x=(40.0, 60.0, 62.0))
    line = offside_line(pos, teams, TEAM_A, True)
    assert line.x == 52.5
    assert check_against_line(pos, teams, line)[0].status == ONSIDE


def test_check_frame_uses_the_opponents_line_for_each_team():
    # A defends left (line at 12), B defends right (second-last defender at 83)
    pos = np.array([[1, 34], [12, 20], [14, 40], [85, 30],        # A: keeper, defenders, a forward past B's line
                    [104, 34], [80, 20], [83, 40], [10, 30]], float)  # B: keeper, defenders, a forward past A's line
    teams = np.array([TEAM_A] * 4 + [TEAM_B] * 4)
    lines = {TEAM_A: offside_line(pos, teams, TEAM_A, True), TEAM_B: offside_line(pos, teams, TEAM_B, False)}
    assert lines[TEAM_B].x == 83
    checks = check_frame(pos, teams, lines)
    assert checks[7].status == OFFSIDE and checks[7].defending_team == TEAM_A
    assert checks[3].status == OFFSIDE and checks[3].defending_team == TEAM_B
    assert checks[1].status == ONSIDE                              # A's defender judged vs B's line
    assert len(checks) == 8


def test_video_tags_and_minimap_rings():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_offside_projection import _scene as broadcast_scene
    from pitchlens.calibration.pitch import Pitch
    from pitchlens.visualization.minimap import Minimap, draw_player_rings
    from pitchlens.visualization.offside_overlay import draw_position_flags, flag_color

    frame, calib, boxes, pos, teams = broadcast_scene()        # blue #5 at x=10 is past red's line (11.5)
    line = offside_line(pos, teams, 0, True)
    checks = {c.player_index: c for c in check_against_line(pos, teams, line)}
    assert checks[5].status == OFFSIDE and checks[6].status == ONSIDE
    out = draw_position_flags(frame.copy(), boxes, checks)
    changed = (out != frame).any(axis=2)
    x1, y1, x2, _ = boxes[5].round().astype(int)
    assert changed[max(0, y1 - 25):y1, x1 - 20:x2 + 20].any()      # tag above the offside attacker
    x1, y1, x2, _ = boxes[6].round().astype(int)
    assert not changed[max(0, y1 - 25):y1, x1:x2].any()            # nothing above the onside one

    mm = Minimap(Pitch(), width_px=555)
    img = mm.render(np.empty((0, 2)), [])
    before = img.copy()
    draw_player_rings(img, mm, pos[[5]], [flag_color(OFFSIDE)])
    assert (img != before).any()
    assert flag_color(ONSIDE) is None and flag_color(TOO_CLOSE) is not None


def test_pipeline_writes_offside_columns(tmp_path):
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import supervision as sv
    import test_pipeline as tp
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.offside.direction import ATTACKS_LEFT, ATTACKS_RIGHT, DirectionEstimate
    from pitchlens.pipeline import analyze, render, solve
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
    red = assigner.assign(first, tp._boxes(0)[:1], np.array([999]))[0]
    directions = DirectionEstimate({red: ATTACKS_RIGHT, 1 - red: ATTACKS_LEFT}, 1.0, 1)
    out_csv = tmp_path / "tracks.csv"
    render(str(video), str(tmp_path / "out.mp4"), sv.VideoInfo.from_video_path(str(video)), records, calibs,
           assigner.team_colors_bgr, None, str(out_csv), directions=directions)
    rows = list(csv.DictReader(out_csv.open()))
    assert {"offside_status", "offside_margin_m"} <= set(rows[0])
    statuses = {r["offside_status"] for r in rows}
    assert statuses - {""} and statuses <= {"", OFFSIDE, ONSIDE, TOO_CLOSE}
    for r in rows:   # status agrees with the margin
        if r["offside_status"] == OFFSIDE:
            assert float(r["offside_margin_m"]) > 0.3
        elif r["offside_status"] == ONSIDE:
            assert float(r["offside_margin_m"]) < -0.3
