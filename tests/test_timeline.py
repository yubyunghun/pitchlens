import csv
import sys
from pathlib import Path

import numpy as np

from pitchlens.offside.passes import FramePlayers
from pitchlens.tactics.timeline import (UNCLEAR, WindowFormation, merge_segments, summarize_timeline,
                                        window_formations, write_timeline_csv)
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_average_shape import _layout  # noqa: E402

FPS = 10.0                       # low frame rate keeps the test fast
DIRS = {TEAM_A: 1, TEAM_B: -1}


def _game(seconds=60, switch_s=30, seed=0):
    """A plays 4-4-2, then switches to 3-5-2 at `switch_s`. B is a 4-3-3 throughout."""
    rng = np.random.default_rng(seed)
    a1, a2 = _layout("4-4-2", (20, 58)), _layout("3-5-2", (20, 58))
    b_lay = _layout("4-3-3", (20, 58))
    b = np.c_[105 - b_lay[:, 0], 68 - b_lay[:, 1]]
    frames = []
    for t in range(int(seconds * FPS)):
        a = (a1 if t < switch_s * FPS else a2) + rng.normal(0, 1.5, (10, 2))
        frames.append(FramePlayers(np.vstack([a, b + rng.normal(0, 1.5, (10, 2))]),
                                   np.array([TEAM_A] * 10 + [TEAM_B] * 10),
                                   np.r_[np.arange(1, 11), np.arange(21, 31)]))
    return frames


def test_switch_is_found_near_the_right_time():
    windows = window_formations(_game(), DIRS, FPS)
    segments = merge_segments(windows)
    a = [s for s in segments if s.team == TEAM_A]
    names = [s.formation for s in a]
    assert names[0] == "4-4-2" and names[-1] == "3-5-2"
    switch = next(s for s in a if s.formation != "4-4-2").start / FPS
    assert 20 <= switch <= 35                                  # within a window of the real 30 s switch
    b = [s for s in segments if s.team == TEAM_B]
    assert [s.formation for s in b] == ["4-3-3"]               # B never changed
    assert b[0].start == 0 and b[0].end == len(_game())


def test_short_clip_gets_one_window():
    frames = _game(seconds=8, switch_s=100)
    windows = window_formations(frames, DIRS, FPS, window_s=15.0)
    assert {w.start for w in windows} == {0}
    assert {w.formation for w in windows if w.team == TEAM_A} == {"4-4-2"}


def test_unclear_windows():
    frames = _game(seconds=20, switch_s=100)
    windows = window_formations(frames, DIRS, FPS, min_seconds=30.0)  # nobody seen 30 s in a window
    assert {w.formation for w in windows} == {UNCLEAR}
    assert all(w.players == 0 for w in windows)


def test_segments_do_not_overlap_and_summary_csv(tmp_path):
    ws = [WindowFormation(TEAM_A, 0, 150, "4-4-2", 0.6, 10), WindowFormation(TEAM_A, 50, 200, "4-4-2", 0.4, 10),
          WindowFormation(TEAM_A, 100, 250, "3-5-2", 0.5, 10)]
    segs = merge_segments(ws)
    assert [(s.formation, s.start, s.end) for s in segs] == [("4-4-2", 0, 150), ("3-5-2", 150, 250)]
    assert abs(segs[0].confidence - 0.5) < 1e-9
    text = summarize_timeline(segs, FPS, {TEAM_A: "team 0 (red)"})
    assert text.splitlines()[1] == "  team 0 (red): 0:00.0-0:15.0 4-4-2, 0:15.0-0:25.0 3-5-2"
    out = tmp_path / "timeline.csv"
    write_timeline_csv(out, segs, FPS)
    rows = list(csv.DictReader(out.open()))
    assert [(r["formation"], r["start_s"], r["end_s"]) for r in rows] == [("4-4-2", "0.0", "15.0"),
                                                                          ("3-5-2", "15.0", "25.0")]


def test_with_per_frame_classes_like_the_pipeline():
    from pitchlens.detection.classes import PLAYER
    frames = _game(seconds=20, switch_s=100)
    classes = [np.full(len(f.track_ids), PLAYER) for f in frames]
    windows = window_formations(frames, DIRS, FPS, classes, window_s=15.0, step_s=5.0, min_seconds=3.0)
    assert {w.formation for w in windows if w.team == TEAM_A} == {"4-4-2"}
