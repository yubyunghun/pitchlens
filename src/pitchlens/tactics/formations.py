"""Which formation does a set of outfield positions look like?

Each formation is a template of 10 typical outfield positions, in a
normalized "team block": depth 0 = own goal side, 1 = attacking side, width
0..1 across the pitch. Real positions (pitch meters) are turned into the same
attacking-right orientation, then both shapes are standardized (centered and
scaled per axis) so a compact low block and a stretched high line can both
read as, say, a 4-4-2.

Matching: players are paired with template slots one-to-one (Hungarian
algorithm, minimal total squared distance). Broadcast footage rarely shows all
10 outfield players, so with fewer players the template is re-standardized
using only the slots that were matched, and the pairing is repeated a few
times until it settles. The template with the lowest mean distance wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from pitchlens.calibration.pitch import Pitch

DEF, MID, FWD = "DEF", "MID", "FWD"


@dataclass(frozen=True)
class Slot:
    depth: float      # 0 = own goal side of the team block, 1 = attacking side
    width: float      # 0 = one touchline side, 1 = the other
    line: str         # DEF / MID / FWD
    role: str         # e.g. "CB", "LW"


def _line(depth: float, line: str, roles: list[str], spread: float) -> list[Slot]:
    n = len(roles)
    widths = [0.5] if n == 1 else list(np.linspace(0.5 - spread / 2, 0.5 + spread / 2, n))
    return [Slot(depth, float(w), line, r) for w, r in zip(widths, roles)]


TEMPLATES: dict[str, list[Slot]] = {
    "4-4-2": (_line(0.15, DEF, ["LB", "CB", "CB", "RB"], 0.8)
              + _line(0.45, MID, ["LM", "CM", "CM", "RM"], 0.8)
              + _line(0.75, FWD, ["ST", "ST"], 0.3)),
    "4-3-3": (_line(0.15, DEF, ["LB", "CB", "CB", "RB"], 0.8)
              + _line(0.45, MID, ["CM", "CM", "CM"], 0.5)
              + _line(0.78, FWD, ["LW", "ST", "RW"], 0.75)),
    "4-2-3-1": (_line(0.15, DEF, ["LB", "CB", "CB", "RB"], 0.8)
                + _line(0.37, MID, ["DM", "DM"], 0.3)
                + _line(0.6, MID, ["LAM", "AM", "RAM"], 0.65)
                + _line(0.82, FWD, ["ST"], 0.0)),
    "4-1-4-1": (_line(0.15, DEF, ["LB", "CB", "CB", "RB"], 0.8)
                + _line(0.32, MID, ["DM"], 0.0)
                + _line(0.53, MID, ["LM", "CM", "CM", "RM"], 0.75)
                + _line(0.8, FWD, ["ST"], 0.0)),
    "3-5-2": (_line(0.15, DEF, ["CB", "CB", "CB"], 0.5)
              + [Slot(0.45, 0.08, MID, "LWB"), Slot(0.45, 0.92, MID, "RWB")]
              + _line(0.42, MID, ["CM", "CM", "CM"], 0.4)
              + _line(0.75, FWD, ["ST", "ST"], 0.3)),
    "3-4-3": (_line(0.15, DEF, ["CB", "CB", "CB"], 0.5)
              + _line(0.45, MID, ["LWB", "CM", "CM", "RWB"], 0.84)
              + _line(0.78, FWD, ["LW", "ST", "RW"], 0.7)),
    "5-3-2": (_line(0.15, DEF, ["CB", "CB", "CB"], 0.4)
              + [Slot(0.22, 0.08, DEF, "LWB"), Slot(0.22, 0.92, DEF, "RWB")]
              + _line(0.45, MID, ["CM", "CM", "CM"], 0.5)
              + _line(0.75, FWD, ["ST", "ST"], 0.3)),
    "5-4-1": (_line(0.15, DEF, ["CB", "CB", "CB"], 0.4)
              + [Slot(0.22, 0.08, DEF, "LWB"), Slot(0.22, 0.92, DEF, "RWB")]
              + _line(0.45, MID, ["LM", "CM", "CM", "RM"], 0.75)
              + _line(0.78, FWD, ["ST"], 0.0)),
}
assert all(len(slots) == 10 for slots in TEMPLATES.values())


@dataclass
class FormationMatch:
    name: str
    cost: float                               # mean squared distance after standardizing (lower = better)
    ranking: list[tuple[str, float]]          # every template, best first
    slots: dict[int, Slot]                    # player index -> template slot it was matched to

    @property
    def confidence(self) -> float:
        """How clearly the best template beats the runner-up (0 = tie, -> 1 = clear)."""
        if len(self.ranking) < 2 or self.ranking[1][1] <= 0:
            return 1.0
        return float(1 - self.ranking[0][1] / self.ranking[1][1])

    def describe(self, min_confidence: float = 0.2) -> str:
        """'4-4-2', or '4-2-3-1 (or 4-1-4-1)' when the runner-up is nearly as good."""
        if self.confidence >= min_confidence or len(self.ranking) < 2:
            return self.name
        return f"{self.name} (or {self.ranking[1][0]})"

    def lines(self) -> dict[str, list[int]]:
        """Player indices per line (DEF / MID / FWD), each sorted across the pitch."""
        out: dict[str, list[int]] = {DEF: [], MID: [], FWD: []}
        for i, slot in sorted(self.slots.items(), key=lambda kv: kv[1].width):
            out[slot.line].append(i)
        return out

    def chains(self, depth_tolerance: float = 0.08) -> list[list[int]]:
        """Players to join when drawing: same line and (nearly) the same depth, sorted across.

        Splits e.g. a 4-1-4-1's holding midfielder from the midfield four, while a back
        five (wing-backs slightly higher than the center-backs) stays one chain.
        """
        groups: list[tuple[str, float, list[int]]] = []
        for i, slot in sorted(self.slots.items(), key=lambda kv: (kv[1].line, kv[1].depth)):
            if groups and groups[-1][0] == slot.line and slot.depth - groups[-1][1] <= depth_tolerance:
                groups[-1][2].append(i)
            else:
                groups.append((slot.line, slot.depth, [i]))
        return [sorted(idx, key=lambda i: self.slots[i].width) for _, _, idx in groups]


def to_attacking_right(positions: np.ndarray, attacks_right: bool, pitch: Pitch = Pitch()) -> np.ndarray:
    """Pitch meters -> (depth, width) meters with the team attacking toward +depth."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    if attacks_right:
        return pos.copy()
    return np.c_[pitch.length - pos[:, 0], pitch.width - pos[:, 1]]   # rotate 180 degrees


def _standardize(pts: np.ndarray) -> np.ndarray:
    std = pts.std(axis=0)
    std[std < 1e-6] = 1.0
    return (pts - pts.mean(axis=0)) / std


def _match_template(obs: np.ndarray, template: np.ndarray, rounds: int = 4) -> tuple[float, np.ndarray]:
    """(mean squared distance, slot index per observed player)."""
    o = _standardize(obs)
    used = np.arange(len(template))
    cols = None
    for _ in range(rounds):
        t = _standardize(template[used])
        cost = ((o[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
        rows, c = linear_sum_assignment(cost)
        new_cols = used[c]
        if cols is not None and np.array_equal(np.sort(new_cols), np.sort(cols)):
            cols = new_cols
            break
        cols = new_cols
        used = np.sort(cols)                     # refit on the slots that are actually filled
    t = _standardize(template[np.sort(cols)])
    order = {s: k for k, s in enumerate(np.sort(cols))}
    matched = t[[order[s] for s in cols]]
    return float(((o - matched) ** 2).sum(axis=1).mean()), cols


def classify_formation(positions: np.ndarray, attacks_right: bool, pitch: Pitch = Pitch(),
                       min_players: int = 7,
                       templates: dict[str, list[Slot]] | None = None) -> FormationMatch | None:
    """Best-matching formation for a team's OUTFIELD players (goalkeeper excluded).

    positions:     (k, 2) pitch meters, k <= 10 (NaN rows are ignored)
    attacks_right: True if the team attacks toward x = pitch length
    Returns None with fewer than `min_players` known positions.
    """
    templates = templates or TEMPLATES
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    known = np.flatnonzero(np.isfinite(pos).all(axis=1))
    if len(known) < min_players:
        return None
    obs = to_attacking_right(pos[known], attacks_right, pitch)

    results = []
    for name, slots in templates.items():
        tpl = np.array([[s.depth, s.width] for s in slots])
        cost, cols = _match_template(obs, tpl)
        results.append((cost, name, cols))
    results.sort(key=lambda r: r[0])
    best_cost, best_name, best_cols = results[0]
    slots = {int(known[i]): templates[best_name][int(c)] for i, c in enumerate(best_cols)}
    return FormationMatch(best_name, best_cost, [(n, c) for c, n, _ in results], slots)
