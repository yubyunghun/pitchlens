"""Draw a team's shape: players joined line by line, role labels, formation name.

Built on the Minimap pitch so it can be shown in the video corner or saved as
a standalone chart (e.g. average positions).
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchlens.tactics.formations import FormationMatch
from pitchlens.visualization.minimap import Minimap


def draw_formation(img: np.ndarray, minimap: Minimap, positions: np.ndarray, match: FormationMatch,
                   color: tuple[int, int, int], title: str | None = None,
                   show_roles: bool = True) -> np.ndarray:
    """Draws one team's formation onto a rendered minimap image (in place).

    positions: (N, 2) pitch meters, indexed like `match.slots`
    title:     text for the top margin (default: the formation, e.g. '4-4-2')
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    r = max(5, int(round(minimap.scale * 1.1)))
    thick = max(2, int(round(minimap.scale * 0.3)))

    # 1. join each line (back four, midfield, front; split where a line has layers) across the pitch
    for chain in match.chains():
        idx = [i for i in chain if np.isfinite(pos[i]).all()]
        if len(idx) >= 2:
            pts = minimap.to_px(pos[idx]).reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [pts], False, (20, 20, 20), thick + 2, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)

    # 2. players and roles on top
    font_scale = max(0.3, minimap.scale / 18)
    for i, slot in match.slots.items():
        if not np.isfinite(pos[i]).all():
            continue
        p = tuple(int(v) for v in minimap.to_px(pos[i]))
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (20, 20, 20), 1, cv2.LINE_AA)
        if show_roles:
            (tw, th), _ = cv2.getTextSize(slot.role, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            org = (p[0] - tw // 2, p[1] + r + th + 3)
            cv2.putText(img, slot.role, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, slot.role, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. formation name in the top margin
    text = title if title is not None else match.describe()
    margin_px = minimap.margin * minimap.scale
    scale = max(0.45, margin_px / 40)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    y = int(round(margin_px / 2 + th / 2))
    dot = max(4, th // 2)
    cv2.circle(img, (8 + dot, y - th // 2), dot, color, -1, cv2.LINE_AA)
    cv2.circle(img, (8 + dot, y - th // 2), dot, (20, 20, 20), 1, cv2.LINE_AA)
    x = 8 + 2 * dot + 8
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img
