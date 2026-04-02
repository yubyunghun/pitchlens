import sys
from pathlib import Path

import numpy as np

from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.lane_options import frame_options
from pitchlens.tactics.passing_lanes import OPEN
from pitchlens.tactics.possession import NO_POSSESSION
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_passing_lanes import _attack  # noqa: E402

DIRS = {TEAM_A: 1, TEAM_B: -1}


def _frames(n=5):
    pos, teams = _attack()
    return [FramePlayers(pos.copy(), teams, np.arange(100, 100 + len(pos))) for _ in range(n)], pos


def test_options_when_a_team_has_the_ball_at_its_feet():
    frames, pos = _frames()
    ball = np.tile(pos[0] + [0.5, 0.0], (5, 1))              # at the carrier's feet
    possession = np.array([TEAM_A, TEAM_A, NO_POSSESSION, TEAM_B, TEAM_A])
    ball[4] = [np.nan, np.nan]                               # ball not known in frame 4
    opts = frame_options(frames, ball, possession, DIRS)
    assert opts[0] is not None and opts[0].carrier == 0 and opts[0].carrier_track_id == 100
    assert opts[0].best.receiver_index == 2 and opts[0].best.status == OPEN
    assert len(opts[0].lanes) == 4
    assert opts[2] is None                                   # nobody in possession
    assert opts[3] is None                                   # B "has the ball" but none of B is near it
    assert opts[4] is None                                   # ball unknown


def test_minimap_lanes_drawn_under_the_dots():
    from pitchlens.calibration.pitch import Pitch
    from pitchlens.visualization.lanes_overlay import LANE_COLORS, draw_lanes_minimap
    from pitchlens.visualization.minimap import Minimap
    frames, pos = _frames(1)
    opts = frame_options(frames, pos[:1] + [0.5, 0.0], np.array([TEAM_A]), DIRS)[0]
    mm = Minimap(Pitch(), width_px=740)
    colors = [(40, 40, 210) if t == TEAM_A else (210, 90, 30) for t in frames[0].teams]
    img = draw_lanes_minimap(mm.render(pos, colors), mm, pos, opts)
    mm.draw_players(img, pos, colors)
    mid = mm.to_px((opts.start + pos[opts.best.receiver_index]) / 2)
    patch = img[mid[1] - 3:mid[1] + 4, mid[0] - 3:mid[0] + 4].reshape(-1, 3).astype(int)
    assert (np.linalg.norm(patch - np.array(LANE_COLORS[OPEN]), axis=1) < 40).any()   # green best lane
    x, y = mm.to_px(pos[2])
    assert tuple(img[y, x]) == (40, 40, 210)                  # receiver dot drawn on top of the lane


def test_video_lanes_stay_behind_players():
    from test_offside_projection import _scene
    from pitchlens.visualization.lanes_overlay import draw_lanes_video
    frame, calib, boxes, pos, teams = _scene()               # broadcast view, red defends left
    fp = FramePlayers(pos, teams, np.arange(len(pos)))
    blue = int(np.flatnonzero(teams == 1)[1])                # blue player with the ball
    opts = frame_options([fp], pos[blue:blue + 1], np.array([1]), {0: 1, 1: -1})[0]
    assert opts is not None
    out = draw_lanes_video(frame.copy(), calib, pos, opts, occluders=boxes)
    changed = (out != frame).any(axis=2)
    assert changed.any()
    for x1, y1, x2, y2 in boxes.round().astype(int):
        assert not changed[y1:y2, x1:x2].any()               # no lane drawn over any player


def test_render_draws_lanes(tmp_path):
    import supervision as sv
    import test_pipeline as tp
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.calibration.pitch import Pitch
    from pitchlens.pipeline import analyze, offside_inputs, render, solve
    from pitchlens.tracking.team_assigner import TeamAssigner
    from pitchlens.visualization.minimap import Minimap

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    assigner.collect(first, tp._boxes(0))
    assigner.collect(first, tp._boxes(0))
    assigner.fit()
    records = analyze(str(video), tp.FakeDetector(), assigner, fps=25)
    calibs = solve(records, CalibrationSet({0: tp._calib_for(tp._cam(0))}), tp.SIZE)
    players = offside_inputs(records, calibs)[0]
    red = assigner.assign(first, tp._boxes(0)[:1], np.array([999]))[0]
    ball = np.array([fp.positions[np.flatnonzero(fp.teams == red)[0]] if (fp.teams == red).any()
                     else [np.nan, np.nan] for fp in players])
    opts = frame_options(players, ball, np.full(len(players), red), {red: 1, 1 - red: -1})
    assert sum(o is not None for o in opts) > len(opts) // 2
    info = sv.VideoInfo.from_video_path(str(video))
    plain = tmp_path / "plain.mp4"
    lanes = tmp_path / "lanes.mp4"
    render(str(video), str(plain), info, records, calibs, assigner.team_colors_bgr, Minimap(Pitch(), 288), None)
    render(str(video), str(lanes), info, records, calibs, assigner.team_colors_bgr, Minimap(Pitch(), 288), None,
           lane_options=opts)
    f_plain = list(sv.get_video_frames_generator(str(plain)))[10]
    f_lanes = list(sv.get_video_frames_generator(str(lanes)))[10]
    assert (np.abs(f_plain.astype(int) - f_lanes.astype(int)) > 30).any()


def test_options_csv(tmp_path):
    import csv
    from pitchlens.tactics.lane_options import write_options_csv
    frames, pos = _frames(4)
    ball = np.tile(pos[0] + [0.5, 0.0], (4, 1))
    opts = frame_options(frames, ball, np.array([TEAM_A, NO_POSSESSION, TEAM_A, TEAM_A]), DIRS)
    out = tmp_path / "passing_options.csv"
    assert write_options_csv(out, opts, fps=25.0) == 3
    rows = list(csv.DictReader(out.open()))
    assert [r["frame"] for r in rows] == ["0", "2", "3"]
    r = rows[0]
    assert r["carrier_track_id"] == "100" and r["best_receiver_track_id"] == "102"     # track id, not index
    assert int(r["open"]) + int(r["risky"]) + int(r["blocked"]) == 4
    assert float(r["best_forward_m"]) == 21.5 and r["best_bypassed"] == "3"   # from the ball, 0.5 m ahead
    assert opts[0].count("blocked") == int(r["blocked"]) >= 2
