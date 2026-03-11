import numpy as np
import pytest

from pitchlens.offside.ball_path import build_ball_path

FPS = 25.0


def _rolling(n=50, start=(20.0, 30.0), velocity=(10.0, 0.0)):
    """Ball rolling at constant velocity (m/s)."""
    t = np.arange(n)[:, None] / FPS
    return list(np.array(start) + t * np.array(velocity))


def test_clean_path_keeps_positions_and_speed():
    path = build_ball_path(_rolling(), FPS)
    assert path.observed.all() and path.known.all()
    assert np.allclose(path.positions, np.array(_rolling()), atol=1e-9)
    assert np.allclose(path.speed(), 10.0)


def test_short_gaps_filled_long_gaps_left_open():
    raw = _rolling(60)
    for t in range(10, 18):                  # 8 missed frames (0.32 s)
        raw[t] = None
    for t in range(30, 50):                  # 20 missed frames (0.8 s)
        raw[t] = None
    path = build_ball_path(raw, FPS, max_gap_s=0.5)
    assert path.known[10:18].all() and not path.observed[10:18].any()
    assert np.allclose(path.positions[10:18], np.array(_rolling(60))[10:18], atol=1e-9)
    assert not path.known[30:50].any()


def test_false_detection_removed():
    raw = _rolling(40)
    raw[20] = np.array([60.0, 10.0])         # e.g. a shoe 40 m away
    path = build_ball_path(raw, FPS)
    assert not path.observed[20]
    assert np.allclose(path.positions[20], _rolling(40)[20], atol=1e-6)   # filled from neighbors


def test_fast_shot_is_not_mistaken_for_an_outlier():
    raw = _rolling(20, velocity=(1.0, 0.0)) + _rolling(20, start=(20.8, 30.0), velocity=(33.0, 0.0))
    path = build_ball_path(raw, FPS)
    assert path.observed.all()
    assert path.speed()[25:35].mean() == pytest.approx(33.0, rel=0.05)


def test_smoothing_reduces_jitter_but_keeps_the_kick():
    rng = np.random.default_rng(0)
    before = _rolling(25, velocity=(0.0, 0.0))                              # ball at rest
    after = _rolling(25, start=(20.0, 30.0), velocity=(15.0, 0.0))          # then kicked at 15 m/s
    raw = [p + rng.normal(0, 0.08, 2) for p in before + after]
    smooth = build_ball_path(raw, FPS, smooth_window=3).speed()
    rough = build_ball_path(raw, FPS, smooth_window=1).speed()
    assert smooth[:22].std() < rough[:22].std()             # less jitter at rest
    assert smooth[:22].mean() < 4 and smooth[30:].mean() > 12
    kick = int(np.argmax(np.diff(smooth)))                  # sharpest speed-up
    assert 23 <= kick <= 26


def test_no_detections():
    path = build_ball_path([None] * 10, FPS)
    assert not path.known.any() and np.isnan(path.speed()).all()
