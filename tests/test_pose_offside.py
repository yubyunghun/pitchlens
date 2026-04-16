import sys
import types
from pathlib import Path

import numpy as np

from pitchlens.offside.passes import FramePlayers, Pass, Touch
from pitchlens.offside.pose_model import crop_boxes, pick_centered
from pitchlens.tracking.team_assigner import TEAM_A, TEAM_B

sys.path.insert(0, str(Path(__file__).parent))
from test_body_parts import CALIB, _pose  # noqa: E402


def test_crops_are_padded_and_clipped():
    crops = crop_boxes(np.array([[100, 100, 120, 160], [0, 0, 20, 40]]), (720, 1280), padding=0.25)
    assert list(crops[0]) == [95, 85, 125, 175]
    assert list(crops[1]) == [0, 0, 25, 50]                    # clipped at the frame edge


def test_pick_the_person_the_crop_is_for():
    boxes = np.array([[0, 0, 10, 30], [12, 8, 28, 52]], float)  # someone at the edge, the player centered
    assert pick_centered(boxes, (40, 60)) == 1
    assert pick_centered(np.zeros((0, 4)), (40, 60)) is None


class FakePose:
    """Returns, for every crop, one off-center person and one centered person (crop coordinates)."""

    def __init__(self, path):
        pass

    def __call__(self, crops, **kwargs):
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            kp_center = np.zeros((17, 3))
            kp_center[:, 0], kp_center[:, 1], kp_center[:, 2] = w / 2, np.linspace(5, h - 5, 17), 0.9
            kp_edge = kp_center.copy()
            kp_edge[:, 0] = 1
            boxes = np.array([[0, 0, 3, h], [w / 2 - 5, 2, w / 2 + 5, h - 2]])
            out.append(types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=boxes),
                                             keypoints=types.SimpleNamespace(data=np.stack([kp_edge, kp_center]))))
        return out


def test_pose_detector_maps_keypoints_back_to_the_frame(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakePose))
    monkeypatch.delitem(sys.modules, "pitchlens.offside.pose_model", raising=False)
    from pitchlens.offside.pose_model import PoseDetector
    det = PoseDetector("pose.pt", padding=0.25)
    frame = np.zeros((720, 1280, 3), np.uint8)
    kps = det.predict_players(frame, np.array([[500, 300, 520, 360], [900, 400, 916, 448]], float))
    assert len(kps) == 2
    assert np.allclose(kps[0][:, 0], 510) and kps[0][:, 1].min() > 280      # centered person, frame coords
    assert np.allclose(kps[1][:, 0], 908)
    assert det.predict_players(frame, np.zeros((0, 4))) == []


def test_frame_players_fall_back_to_feet():
    feet = np.array([[10.0, 5.0]])
    fp = FramePlayers(feet, np.array([TEAM_A]), np.array([1]))
    assert fp.positions_toward(True) is feet
    body = {True: np.array([[9.5, 5.0]]), False: np.array([[10.4, 5.0]])}
    fp = FramePlayers(feet, np.array([TEAM_A]), np.array([1]), body)
    assert fp.positions_toward(False)[0, 0] == 10.4


def test_pass_decision_uses_body_parts():
    """Feet level with the line (too close); the attacker's head 0.4 m ahead -> offside by body parts."""
    from pitchlens.offside.body_parts import positions_toward_goal
    from pitchlens.offside.decision import OFFSIDE, TOO_CLOSE, decide
    from pitchlens.offside.freeze import freeze_pass
    from pitchlens.offside.line import offside_line

    feet = np.array([[104.0, 34.0], [92.0, 20.0], [88.0, 50.0], [70.0, 34.0], [92.0, 40.0]])  # B x3, A passer, A
    teams = np.array([TEAM_B, TEAM_B, TEAM_B, TEAM_A, TEAM_A])
    ids = np.array([20, 21, 22, 10, 11])
    poses = [_pose(*f) for f in feet[:4]] + [_pose(92.0, 40.0, lean_m=0.4)]
    kps, boxes = [p[0] for p in poses], np.stack([p[1] for p in poses])
    body = {g: positions_toward_goal(feet, kps, boxes, CALIB, g) for g in (True, False)}
    p = Pass(Touch(0, 10, TEAM_A, 3, feet[3], 12.0), Touch(20, 11, TEAM_A, None, feet[4], 12.0))

    def verdict(fp):
        pos = fp.positions_toward(False)                    # B defends the goal at x = 105
        lines = [{TEAM_B: offside_line(pos, teams, TEAM_B, defends_left=False)}]
        return decide(p, freeze_pass(p, [fp], lines)).verdict

    assert verdict(FramePlayers(feet, teams, ids)) == TOO_CLOSE
    assert verdict(FramePlayers(feet, teams, ids, body)) == OFFSIDE


def test_analyze_runs_the_pose_model_on_tracked_players(tmp_path):
    import supervision as sv
    import test_pipeline as tp
    from pitchlens.calibration.keyframes import CalibrationSet
    from pitchlens.pipeline import analyze, offside_inputs, solve
    from pitchlens.tracking.team_assigner import TeamAssigner

    class FakePoseDetector:
        def predict_players(self, frame, boxes):
            out = []
            for x1, y1, x2, y2 in boxes:
                kp = np.zeros((17, 3))
                kp[:, 0], kp[:, 1], kp[:, 2] = (x1 + x2) / 2 + 3, np.linspace(y1, y2, 17), 0.9   # leaning +3 px
                out.append(kp)
            return out

    video = tmp_path / "clip.avi"
    tp._make_video(video)
    assigner = TeamAssigner()
    first = next(sv.get_video_frames_generator(str(video)))
    assigner.collect(first, tp._boxes(0))
    assigner.collect(first, tp._boxes(0))
    assigner.fit()
    records = analyze(str(video), tp.FakeDetector(), assigner, fps=25, pose_detector=FakePoseDetector())
    assert all(r.poses is not None and len(r.poses) == len(r.players) for r in records)
    calibs = solve(records, CalibrationSet({0: tp._calib_for(tp._cam(0))}), tp.SIZE)
    players = offside_inputs(records, calibs)[0]
    fp = players[10]
    assert fp.toward_goal is not None and set(fp.toward_goal) == {True, False}
    # the body leans toward +x in the image, so toward the right goal the body part is ahead of the feet
    assert (fp.toward_goal[False][:, 0] >= fp.positions[:, 0] - 1e-6).all()
