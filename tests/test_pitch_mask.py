import cv2
import numpy as np

from pitchlens.calibration.pitch_mask import background_mask, grass_mask

GRASS = (40, 130, 45)    # BGR
STANDS = (90, 80, 100)
LINE = (240, 240, 240)


def _scene():
    """Stands on top, grass below with a white line, two players and a score bug."""
    f = np.zeros((360, 640, 3), np.uint8)
    f[:120] = STANDS
    f[120:] = GRASS
    cv2.line(f, (0, 240), (639, 240), LINE, 4)        # pitch marking
    f[200:260, 100:120] = (0, 0, 220)                  # player A
    f[250:310, 400:420] = (220, 60, 0)                 # player B
    f[20:50, 20:160] = (30, 30, 30)                    # score bug (on stands)
    boxes = np.array([[100, 200, 120, 260], [400, 250, 420, 310]], float)
    return f, boxes


def test_grass_mask_fills_lines_and_excludes_stands():
    f, _ = _scene()
    m = grass_mask(f)
    assert m[240, 320] == 255          # white line inside the pitch is kept
    assert m[230, 110] == 255          # player hole is filled (removed later by boxes)
    assert m[60, 320] == 0             # stands
    assert m[35, 90] == 0              # score bug


def test_background_mask_removes_players_and_edge():
    f, boxes = _scene()
    m = background_mask(f, boxes, box_padding=0.2, edge_margin=10)
    assert m[230, 110] == 0            # player A
    assert m[280, 410] == 0            # player B
    assert m[262, 110] == 0            # padding below player A's feet (shadow)
    assert m[125, 320] == 0            # within edge margin of the stands
    assert m[300, 600] == 255          # open grass
    assert m[240, 320] == 255          # pitch line is useful texture


def test_no_grass_gives_empty_mask():
    f = np.full((100, 100, 3), STANDS, np.uint8)
    assert background_mask(f).sum() == 0
