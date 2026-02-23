"""End-to-end pipeline test on a synthetic panning video (fake detector, no YOLO needed)."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv

from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.pitch import Pitch
from pitchlens.detection.classes import PLAYER
from pitchlens.pipeline import analyze, render, solve
from pitchlens.tracking.team_assigner import TeamAssigner
from pitchlens.visualization.minimap import Minimap

sys.path.insert(0, str(Path(__file__).parent))
from test_calibration_tracker import PX_PER_M, _calib_for  # noqa: E402
from test_camera_motion import _camera, _pitch_texture, _view  # noqa: E402
from pitchlens.calibration.homography import apply_homography  # noqa: E402

N_FRAMES = 40
SIZE = (960, 540)
START = {(0, 0, 220): [(30, 25), (40, 40), (50, 30)],      # red team, pitch meters
         (220, 80, 0): [(35, 35), (45, 20), (55, 45)]}     # blue team


def _cam(t):
    return _camera(150 + 12 * t, 120 + 2 * t, zoom=1 + 0.002 * t)


def _truth(t):
    """[(color, (x_m, y_m))] at frame t; players jog 0.1 m/frame."""
    return [(c, (x + 0.1 * t, y)) for c, pts in START.items() for x, y in pts]


def _boxes(t):
    out = []
    for _, p in _truth(t):
        fx, fy = apply_homography(_cam(t), np.array([p]) * PX_PER_M)[0]
        out.append([fx - 10, fy - 50, fx + 10, fy])
    return np.array(out)


def _make_video(path):
    world = _pitch_texture(5)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, SIZE)
    for t in range(N_FRAMES):
        frame = _view(world, _cam(t))
        for (color, _), (x1, y1, x2, y2) in zip(_truth(t), _boxes(t)):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        writer.write(frame)
    writer.release()


class FakeDetector:
    """Returns the known player boxes for each frame in order."""

    def __init__(self):
        self.t = 0

    def detect(self, frame):
        boxes = _boxes(self.t)
        self.t += 1
        players = sv.Detections(xyxy=boxes, confidence=np.full(len(boxes), 0.9),
                                class_id=np.full(len(boxes), PLAYER))
        return SimpleNamespace(players=players, ball=sv.Detections.empty())


def _run(tmp_path, track_camera):
    video = tmp_path / "clip.avi"
    _make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    for _ in range(3):
        assigner.collect(first, _boxes(0))
    assigner.fit()

    records = analyze(str(video), FakeDetector(), assigner, fps=25, track_camera=track_camera)
    keys = CalibrationSet({0: _calib_for(_cam(0))})
    calibs = solve(records, keys, SIZE, track_camera)
    info = sv.VideoInfo.from_video_path(str(video))
    out_csv = tmp_path / "tracks.csv"
    render(str(video), str(tmp_path / "out.mp4"), info, records, calibs, assigner.team_colors_bgr,
           Minimap(Pitch(), width_px=288), str(out_csv))
    rows = list(csv.DictReader(out_csv.open()))
    return records, rows


def _position_errors(rows):
    errs = []
    for r in rows:
        t = int(r["frame"])
        truth = np.array([p for _, p in _truth(t)])
        est = np.array([float(r["x_m"]), float(r["y_m"])])
        errs.append(np.linalg.norm(truth - est, axis=1).min())   # match to the nearest true player
    return np.array(errs)


def test_pipeline_tracks_positions_through_camera_pan(tmp_path):
    records, rows = _run(tmp_path, track_camera=True)
    assert len(records) == N_FRAMES
    assert sum(r.motion is not None for r in records[1:]) >= N_FRAMES - 2
    assert (tmp_path / "out.mp4").stat().st_size > 0

    frames_seen = {int(r["frame"]) for r in rows}
    assert len(frames_seen) >= N_FRAMES - 2                  # tracker needs 2 hits to confirm
    assert all(r["x_m"] != "" for r in rows)                 # every row calibrated
    errs = _position_errors(rows)
    assert errs.max() < 0.3, f"worst position error {errs.max():.2f} m"

    # the same physical player keeps one track id and one team
    by_id = {}
    for r in rows:
        by_id.setdefault(r["track_id"], set()).add(r["team"])
    assert len(by_id) == 6 and all(len(teams) == 1 for teams in by_id.values())


def test_static_mode_drifts_when_camera_pans(tmp_path):
    _, rows = _run(tmp_path, track_camera=False)
    errs = _position_errors(rows)
    assert errs.max() > 5.0          # without camera tracking positions end up meters off
