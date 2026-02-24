"""End-to-end: video in -> annotated video + per-frame positions CSV.

Two passes over the video:
  1. analyze: detect + track players, assign teams, measure camera motion.
     Only small per-frame results are kept in memory, not the frames.
  2. solve:   turn calibration keyframes + camera motion into a calibration for
              every frame (nearest keyframe, chained forward or backward).
  3. render:  read the video again and draw overlays, minimap and CSV using
              each frame's own calibration. No detection here, so it's fast.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from tqdm import tqdm

from pitchlens.calibration.anchored import FrameCalibration, solve_calibrations
from pitchlens.calibration.camera_motion import CameraMotionTracker
from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.shot_cuts import AppearanceChange, find_cuts
from pitchlens.detection.classes import CLASS_NAMES, REFEREE
from pitchlens.tracking.team_assigner import UNKNOWN, TeamAssigner
from pitchlens.tracking.tracker import PlayerTracker
from pitchlens.visualization.annotator import annotate, player_color
from pitchlens.visualization.minimap import Minimap, paste_minimap

CSV_HEADER = ["frame", "track_id", "class", "team", "x_px", "y_px", "x_m", "y_m", "calib_distance"]


@dataclass
class FrameRecord:
    players: sv.Detections       # tracked players/goalkeepers/referees
    teams: np.ndarray            # team per player (UNKNOWN for referees)
    ball: sv.Detections
    motion: np.ndarray | None    # camera motion from the previous frame (None if unknown)
    change: float = 0.0          # appearance change from the previous frame (0..1), for cut detection


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center of each box: where the player touches the grass."""
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1) if len(xyxy) else np.empty((0, 2))


def fit_teams(detector, video_path: str, n_frames: int = 30, stride: int = 10) -> TeamAssigner:
    """Sample frames spread across the clip and learn the two team colors."""
    assigner = TeamAssigner()
    frames = sv.get_video_frames_generator(video_path, stride=stride)
    for frame in islice(frames, n_frames):
        players = detector.detect(frame).players
        assigner.collect(frame, players.xyxy[players.class_id != REFEREE])
    assigner.fit()
    return assigner


def load_keyframes(path: str | None, info: sv.VideoInfo) -> CalibrationSet | None:
    if not path or not Path(path).exists():
        print("No calibration file found; skipping minimap and pitch coordinates "
              "(run pitchlens-calibrate to create one)")
        return None
    calib_set = CalibrationSet.load(path)
    if not len(calib_set):
        print(f"{path} has no keyframes; skipping minimap and pitch coordinates")
        return None
    return calib_set.for_image_size(info.width, info.height)


def analyze(video: str, detector, assigner: TeamAssigner, fps: float, track_camera: bool = True,
            total: int | None = None) -> list[FrameRecord]:
    """Pass 1: detection, tracking, teams and camera motion for every frame."""
    tracker = PlayerTracker(fps=fps)
    camera = CameraMotionTracker() if track_camera else None
    appearance = AppearanceChange()
    records = []
    for frame in tqdm(sv.get_video_frames_generator(video), total=total, desc="Analyzing"):
        dets = detector.detect(frame)
        players = tracker.update(dets.players)
        teams = np.full(len(players), UNKNOWN, dtype=int)
        not_ref = players.class_id != REFEREE
        if not_ref.any():
            teams[not_ref] = assigner.assign(frame, players.xyxy[not_ref], players.tracker_id[not_ref])
        motion = None
        if camera is not None:
            # cut out everything that moves by itself (raw detections, so untracked players count too)
            moving = np.vstack([dets.players.xyxy.reshape(-1, 4), dets.ball.xyxy.reshape(-1, 4)])
            result = camera.update(frame, moving)
            motion = result.H if result is not None else None
        records.append(FrameRecord(players, teams, dets.ball, motion, appearance.update(frame)))
    return records


def detect_cuts(records: list[FrameRecord]) -> list[int]:
    return find_cuts([r.change for r in records], [r.motion is not None for r in records])


def solve(records: list[FrameRecord], keyframes: CalibrationSet | None, image_size: tuple[int, int],
          track_camera: bool = True, max_gap: int = 10,
          cuts: list[int] | None = None) -> list[FrameCalibration | None]:
    """Calibration for every frame (None where unknown). Chains never cross `cuts`."""
    n = len(records)
    if keyframes is None:
        return [None] * n
    if not track_camera:  # static camera: the first keyframe applies everywhere
        first = keyframes.frames[0]
        return [FrameCalibration(keyframes.keyframes[first], first, abs(t - first)) for t in range(n)]
    if cuts is None:
        cuts = detect_cuts(records)
    return solve_calibrations([r.motion for r in records], keyframes, image_size, breaks=cuts, max_gap=max_gap)


def render(video: str, dst: str, info: sv.VideoInfo, records: list[FrameRecord],
           calibs: list[FrameCalibration | None], team_colors: dict, minimap: Minimap | None,
           csv_path: str | None) -> None:
    """Pass 2: draw overlays and minimap, write the video and the positions CSV."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    csv_file = writer = None
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
    try:
        with sv.VideoSink(dst, info) as sink:
            frames = sv.get_video_frames_generator(video)
            for idx, (frame, rec, fc) in enumerate(tqdm(zip(frames, records, calibs), total=len(records),
                                                       desc="Rendering")):
                feet = foot_points(rec.players.xyxy)
                pitch_xy = fc.calib.to_pitch(feet) if fc else np.full_like(feet, np.nan)

                out = annotate(frame, rec.players, rec.teams, team_colors, rec.ball)
                if minimap and fc:
                    colors = [player_color(c, t, team_colors) for c, t in zip(rec.players.class_id, rec.teams)]
                    # the ball is sometimes in the air, so its ground position is only approximate
                    ball_xy = fc.calib.to_pitch(foot_points(rec.ball.xyxy))[0] if len(rec.ball) else None
                    paste_minimap(out, minimap.render(pitch_xy, colors, ball=ball_xy))
                sink.write_frame(out)

                if writer:
                    for tid, cls, team, (px, py), (mx, my) in zip(rec.players.tracker_id, rec.players.class_id,
                                                                  rec.teams, feet, pitch_xy):
                        writer.writerow([idx, int(tid), CLASS_NAMES[int(cls)], int(team),
                                         round(float(px), 1), round(float(py), 1),
                                         "" if np.isnan(mx) else round(float(mx), 2),
                                         "" if np.isnan(my) else round(float(my), 2),
                                         "" if fc is None else fc.distance])
    finally:
        if csv_file:
            csv_file.close()


def run(config: dict) -> None:
    from pitchlens.detection.detector import Detector  # imported here: loads YOLO / PyTorch

    src, dst = config["video"]["input"], config["video"]["output"]
    det_cfg = config["detection"]
    detector = Detector(det_cfg["model"], det_cfg.get("confidence", 0.3),
                        det_cfg.get("ball_confidence", 0.15), det_cfg.get("device"))
    cam_cfg = config.get("camera_motion") or {}
    track_camera = cam_cfg.get("enabled", True)

    print("Learning team colors...")
    assigner = fit_teams(detector, src)

    info = sv.VideoInfo.from_video_path(src)
    records = analyze(src, detector, assigner, info.fps, track_camera, info.total_frames)

    cuts = detect_cuts(records) if track_camera else []
    if cuts:
        stamps = ", ".join(f"frame {c} ({c / info.fps:.1f}s)" for c in cuts[:10])
        print(f"Detected {len(cuts)} camera cut(s): {stamps}{' ...' if len(cuts) > 10 else ''}")
        print("  Calibrate one frame after each cut so that shot gets pitch coordinates too")

    keyframes = load_keyframes((config.get("calibration") or {}).get("path"), info)
    calibs = solve(records, keyframes, (info.width, info.height), track_camera, cam_cfg.get("max_gap", 10), cuts)
    if keyframes is not None:
        covered = sum(c is not None for c in calibs)
        print(f"Calibrated {covered}/{len(calibs)} frames from {len(keyframes)} keyframe(s)")
        if covered < len(calibs):
            print("  Frames without calibration are cut off from every keyframe (camera cut or lost "
                  "tracking); add a keyframe there with pitchlens-calibrate --frame N")

    mm_cfg = config.get("minimap") or {}
    minimap = None
    if keyframes is not None and mm_cfg.get("enabled", True):
        pitch = next(iter(keyframes.keyframes.values())).pitch
        minimap = Minimap(pitch, width_px=int(info.width * mm_cfg.get("width_fraction", 0.3)))

    csv_path = (config.get("export") or {}).get("tracks_csv")
    render(src, dst, info, records, calibs, assigner.team_colors_bgr, minimap, csv_path)
    print(f"Saved {dst}" + (f" and {csv_path}" if csv_path else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="PitchLens player tracking")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="override input video path")
    parser.add_argument("--output", help="override output video path")
    parser.add_argument("--calibration", help="override calibration JSON path")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.input:
        config["video"]["input"] = args.input
    if args.output:
        config["video"]["output"] = args.output
    if args.calibration:
        config.setdefault("calibration", {})["path"] = args.calibration
    run(config)


if __name__ == "__main__":
    main()
