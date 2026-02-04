import numpy as np

from pitchlens.tracking.team_assigner import TeamAssigner, jersey_color

GRASS = (40, 140, 40)  # BGR


def _frame_with_players(colors_by_x: dict[int, tuple[int, int, int]]) -> np.ndarray:
    frame = np.full((200, 400, 3), GRASS, dtype=np.uint8)
    for x, color in colors_by_x.items():
        frame[40:140, x:x + 30] = color
    return frame


def _box(x):
    return np.array([x, 40, x + 30, 140], dtype=float)


def test_jersey_color_ignores_grass():
    frame = _frame_with_players({100: (0, 0, 255)})
    # Box wider than the player, so it includes grass on the sides.
    c = jersey_color(frame, np.array([90, 40, 140, 140], dtype=float))
    red = jersey_color(_frame_with_players({0: (0, 0, 255)}), _box(0))
    assert c is not None and np.allclose(c, red, atol=2)


def test_two_teams_separated_and_stable():
    red, blue = (0, 0, 220), (220, 60, 0)
    xs_red, xs_blue = [10, 60, 110], [210, 260, 310]
    frame = _frame_with_players({**{x: red for x in xs_red}, **{x: blue for x in xs_blue}})
    boxes = np.stack([_box(x) for x in xs_red + xs_blue])

    assigner = TeamAssigner()
    for _ in range(3):
        assigner.collect(frame, boxes)
    assigner.fit()

    teams = assigner.assign(frame, boxes, np.arange(6))
    assert len(set(teams[:3])) == 1 and len(set(teams[3:])) == 1
    assert teams[0] != teams[3]
