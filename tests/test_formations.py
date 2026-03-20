import numpy as np
import pytest

from pitchlens.tactics.formations import (DEF, FWD, MID, TEMPLATES, classify_formation,
                                          to_attacking_right)


def _team(name, rng=None, noise=0.0, missing=0, depth=(20.0, 58.0), width=(6.0, 62.0)):
    """Template laid out on the pitch (attacking right), with noise and missing players."""
    rng = rng or np.random.default_rng(0)
    pts = np.array([[depth[0] + s.depth * (depth[1] - depth[0]), width[0] + s.width * (width[1] - width[0])]
                    for s in TEMPLATES[name]])
    pts += rng.normal(0, noise, pts.shape)
    return pts[np.sort(rng.permutation(10)[:10 - missing])]


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_recognized(name):
    m = classify_formation(_team(name), attacks_right=True)
    assert m.name == name and m.cost < 1e-9 and m.confidence > 0.5


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_noisy_compact_and_stretched_blocks(name):
    rng = np.random.default_rng(1)
    for depth in [(30.0, 55.0), (15.0, 65.0)]:            # low compact block / stretched team
        assert classify_formation(_team(name, rng, noise=1.5, depth=depth), True).name == name


def test_attacking_left_is_mirrored():
    rng = np.random.default_rng(2)
    team = _team("4-2-3-1", rng, noise=1.0)
    mirrored = np.c_[105 - team[:, 0], 68 - team[:, 1]]
    assert np.allclose(to_attacking_right(mirrored, attacks_right=False), team)
    m = classify_formation(mirrored, attacks_right=False)
    assert m.name == "4-2-3-1"
    left_back = [i for i, s in m.slots.items() if s.role == "LB"][0]
    assert mirrored[left_back, 1] > 34                     # attacking left, their left is the bottom


def test_missing_players():
    rng = np.random.default_rng(3)
    assert classify_formation(_team("4-3-3", rng, noise=1.0, missing=2), True).name == "4-3-3"
    assert classify_formation(_team("4-3-3", rng, missing=4), True) is None   # only 6 visible
    team = _team("4-4-2", rng)
    team[[0, 5]] = np.nan                                  # unknown positions are ignored
    assert classify_formation(team, True).name == "4-4-2"


def test_slots_and_lines():
    m = classify_formation(_team("3-5-2"), True)
    lines = m.lines()
    assert [len(lines[k]) for k in (DEF, MID, FWD)] == [3, 5, 2]
    roles = sorted(s.role for s in m.slots.values())
    assert roles.count("CB") == 3 and "LWB" in roles and "RWB" in roles
    team = _team("3-5-2")
    widths = [team[i, 1] for i in lines[MID]]
    assert widths == sorted(widths)                        # each line ordered across the pitch


def test_ranking_and_describe():
    m = classify_formation(_team("4-4-2"), True)
    assert [n for n, _ in m.ranking][0] == "4-4-2" and len(m.ranking) == len(TEMPLATES)
    assert m.describe() == "4-4-2"
    # halfway between 4-2-3-1 and 4-1-4-1: the answer should admit the doubt
    a, b = _team("4-2-3-1"), _team("4-1-4-1")
    blend = classify_formation((a + b) / 2, True)
    assert blend.confidence < 0.5
    assert "(or " in blend.describe(min_confidence=0.5)

