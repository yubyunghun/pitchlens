"""End-to-end: video in -> detect -> track -> team assignment -> annotated video out."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from tqdm import tqdm

from pitchlens.detection.classes import REFEREE
from pitchlens.detection.detector import Detector
from pitchlens.tracking.team_assigner import UNKNOWN, TeamAssigner
from pitchlens.tracking.tracker import PlayerTracker
from pitchlens.visualization.annotator import annotate


def fit_teams(detector: Detector, video_path: str, n_frames: int = 30, stride: int = 10) -> TeamAssigner:
    """Sample frames spread across the clip and learn the two team colors."""
    assigner = TeamAssigner()
    frames = sv.get_video_frames_generator(video_path, stride=stride)
    for frame in islice(frames, n_frames):
        players = detector.detect(frame).players
        assigner.collect(frame, players.xyxy[players.class_id != REFEREE])
    assigner.fit()
    return assigner


def run(config: dict) -> None:
    src, dst = config["video"]["input"], config["video"]["output"]
    det_cfg = config["detection"]
    detector = Detector(det_cfg["model"], det_cfg.get("confidence", 0.3),
                        det_cfg.get("ball_confidence", 0.15), det_cfg.get("device"))

    print("Learning team colors...")
    assigner = fit_teams(detector, src)

    info = sv.VideoInfo.from_video_path(src)
    tracker = PlayerTracker(fps=info.fps)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    with sv.VideoSink(dst, info) as sink:
        for frame in tqdm(sv.get_video_frames_generator(src), total=info.total_frames, desc="Tracking"):
            dets = detector.detect(frame)
            players = tracker.update(dets.players)
            teams = np.full(len(players), UNKNOWN, dtype=int)
            not_ref = players.class_id != REFEREE
            if not_ref.any():
                teams[not_ref] = assigner.assign(frame, players.xyxy[not_ref], players.tracker_id[not_ref])
            sink.write_frame(annotate(frame, players, teams, assigner.team_colors_bgr, dets.ball))
    print(f"Saved {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PitchLens player tracking")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="override input video path")
    parser.add_argument("--output", help="override output video path")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.input:
        config["video"]["input"] = args.input
    if args.output:
        config["video"]["output"] = args.output
    run(config)


if __name__ == "__main__":
    main()
