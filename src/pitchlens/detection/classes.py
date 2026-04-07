"""Canonical class ids used across PitchLens, independent of the YOLO model."""

BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
CLASS_NAMES = {BALL: "ball", GOALKEEPER: "goalkeeper", PLAYER: "player", REFEREE: "referee"}

# How model / dataset class names map onto PitchLens classes. Covers COCO models
# ("person", "sports ball") and football-trained models / datasets.
NAME_MAP = {
    "ball": BALL,
    "sports ball": BALL,
    "football": BALL,
    "goalkeeper": GOALKEEPER,
    "keeper": GOALKEEPER,
    "player": PLAYER,
    "person": PLAYER,
    "referee": REFEREE,
}


def map_class_names(names: dict[int, str] | list[str]) -> dict[int, int]:
    """Model/dataset class id -> PitchLens class id; names that don't match are left out."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(i): NAME_MAP[n.strip().lower()] for i, n in items if n.strip().lower() in NAME_MAP}
