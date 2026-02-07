"""End-to-end: video in -> detect -> track -> teams -> pitch positions -> annotated video + CSV."""

from __future__ import annotations

import argparse
import csv
from itertools import islice
from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from tqdm import tqdm

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.detection.classes import CLASS_NAMES, REFEREE
from pitchlens.detection.detector import Detector
from pitchlens.tracking.team_assigner import UNKNOWN, TeamAssigner
from pitchlens.tracking.tracker import PlayerTracker
from pitchlens.visualization.annotator import annotate, player_color
from pitchlens.visualization.minimap import Minimap, paste_minimap

CSV_HEADER = ["frame", "track_id", "class", "team", "x_px", "y_px", "x_m", "y_m"]


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center of each box: where the player touches the grass."""
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1) if len(xyxy) else np.empty((0, 2))


def fit_teams(detector: Detector, video_path: str, n_frames: int = 30, stride: int = 10) -> TeamAssigner:
    """Sample frames spread across the clip and learn the two team colors."""
    assigner = TeamAssigner()
    frames = sv.get_video_frames_generator(video_path, stride=stride)
    for frame in islice(frames, n_frames):
        players = detector.detect(frame).players
        assigner.collect(frame, players.xyxy[players.class_id != REFEREE])
    assigner.fit()
    return assigner


def load_calibration(config: dict, info: sv.VideoInfo) -> PitchCalibration | None:
    path = (config.get("calibration") or {}).get("path")
    if not path or not Path(path).exists():
        print("No calibration file found; skipping minimap and pitch coordinates "
              "(run pitchlens-calibrate to create one)")
        return None
    return PitchCalibration.load(path).for_image_size(info.width, info.height)


def run(config: dict) -> None:
    src, dst = config["video"]["input"], config["video"]["output"]
    det_cfg = config["detection"]
    detector = Detector(det_cfg["model"], det_cfg.get("confidence", 0.3),
                        det_cfg.get("ball_confidence", 0.15), det_cfg.get("device"))

    print("Learning team colors...")
    assigner = fit_teams(detector, src)

    info = sv.VideoInfo.from_video_path(src)
    tracker = PlayerTracker(fps=info.fps)
    calib = load_calibration(config, info)
    mm_cfg = config.get("minimap") or {}
    minimap = None
    if calib and mm_cfg.get("enabled", True):
        minimap = Minimap(calib.pitch, width_px=int(info.width * mm_cfg.get("width_fraction", 0.3)))

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    csv_path = (config.get("export") or {}).get("tracks_csv")
    csv_file = writer = None
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

    try:
        with sv.VideoSink(dst, info) as sink:
            frames = sv.get_video_frames_generator(src)
            for idx, frame in enumerate(tqdm(frames, total=info.total_frames, desc="Tracking")):
                dets = detector.detect(frame)
                players = tracker.update(dets.players)
                teams = np.full(len(players), UNKNOWN, dtype=int)
                not_ref = players.class_id != REFEREE
                if not_ref.any():
                    teams[not_ref] = assigner.assign(frame, players.xyxy[not_ref], players.tracker_id[not_ref])

                feet = foot_points(players.xyxy)
                pitch_xy = calib.to_pitch(feet) if calib else np.full_like(feet, np.nan)

                out = annotate(frame, players, teams, assigner.team_colors_bgr, dets.ball)
                if minimap:
                    colors = [player_color(c, t, assigner.team_colors_bgr) for c, t in zip(players.class_id, teams)]
                    # ball is in the air sometimes, so its ground position is only approximate
                    ball_xy = calib.to_pitch(foot_points(dets.ball.xyxy))[0] if len(dets.ball) else None
                    paste_minimap(out, minimap.render(pitch_xy, colors, ball=ball_xy))
                sink.write_frame(out)

                if writer:
                    for tid, cls, team, (px, py), (mx, my) in zip(players.tracker_id, players.class_id,
                                                                  teams, feet, pitch_xy):
                        writer.writerow([idx, int(tid), CLASS_NAMES[int(cls)], int(team),
                                         round(float(px), 1), round(float(py), 1),
                                         "" if np.isnan(mx) else round(float(mx), 2),
                                         "" if np.isnan(my) else round(float(my), 2)])
    finally:
        if csv_file:
            csv_file.close()
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
