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

import cv2
import numpy as np
import supervision as sv
import yaml
from tqdm import tqdm

from pitchlens.calibration.anchored import FrameCalibration, solve_calibrations
from pitchlens.calibration.camera_motion import CameraMotionTracker
from pitchlens.calibration.drift import line_alignment
from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.shot_cuts import AppearanceChange, find_cuts
from pitchlens.detection.classes import CLASS_NAMES, REFEREE
from pitchlens.offside.direction import DirectionEstimate, infer_attack_directions
from pitchlens.offside.line import frame_offside_lines
from pitchlens.tracking.team_assigner import UNKNOWN, TeamAssigner
from pitchlens.tracking.tracker import PlayerTracker
from pitchlens.visualization.annotator import annotate, player_color
from pitchlens.visualization.minimap import (Minimap, draw_attack_arrows, draw_offside_lines,
                                             draw_projected_pitch, paste_minimap)

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


def team_positions(records: list[FrameRecord],
                   calibs: list[FrameCalibration | None]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per calibrated frame: (pitch positions of all tracked players, their team ids)."""
    out = []
    for rec, fc in zip(records, calibs):
        if fc is not None and len(rec.players):
            out.append((fc.calib.to_pitch(foot_points(rec.players.xyxy)), rec.teams))
    return out


def _hex(bgr: tuple[int, int, int]) -> str:
    return "#{2:02x}{1:02x}{0:02x}".format(*bgr)


def _draw_debug_hud(frame: np.ndarray, idx: int, fc: FrameCalibration | None, score: float | None,
                    is_cut: bool) -> None:
    lines = [(f"frame {idx}", (255, 255, 255))]
    if fc is None:
        lines.append(("calib: none (no keyframe in this shot)", (80, 80, 255)))
    else:
        lines.append((f"calib: keyframe {fc.keyframe} {fc.distance:+d}" if fc.distance
                      else f"calib: keyframe {fc.keyframe} (exact)", (255, 255, 255)))
        if score is not None:
            color = (80, 220, 80) if score >= 0.7 else (0, 210, 255) if score >= 0.5 else (80, 80, 255)
            lines.append((f"line match {score:.0%}", color))
    if is_cut:
        lines.append(("CAMERA CUT", (0, 210, 255)))
    cv2.rectangle(frame, (8, 8), (330, 16 + 26 * len(lines)), (20, 20, 20), -1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (16, 32 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def render(video: str, dst: str, info: sv.VideoInfo, records: list[FrameRecord],
           calibs: list[FrameCalibration | None], team_colors: dict, minimap: Minimap | None,
           csv_path: str | None, debug: bool = False, cuts: list[int] = (),
           directions: DirectionEstimate | None = None) -> list[float | None]:
    """Pass 2: draw overlays and minimap, write the video and the positions CSV.

    debug:      draw the calibrated pitch lines and a status box on every frame, and
                measure how well the lines match the video (returned per frame).
    directions: if known, the minimap shows which way each team attacks.
    """
    cut_set = set(cuts)
    arrows = [] if directions is None else [
        (team_colors.get(t, (200, 200, 200)), directions.directions[t]) for t in sorted(directions.directions)]
    scores: list[float | None] = []
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
                score = None
                if debug:
                    if fc:
                        score = line_alignment(frame, fc.calib)  # measured on the clean frame
                        draw_projected_pitch(out, fc.calib, (0, 255, 255), 1)
                    _draw_debug_hud(out, idx, fc, score, idx in cut_set)
                scores.append(score)
                if minimap and fc:
                    colors = [player_color(c, t, team_colors) for c, t in zip(rec.players.class_id, rec.teams)]
                    # the ball is sometimes in the air, so its ground position is only approximate
                    ball_xy = fc.calib.to_pitch(foot_points(rec.ball.xyxy))[0] if len(rec.ball) else None
                    mini = minimap.render(pitch_xy, colors, ball=ball_xy)
                    if directions is not None:
                        lines = frame_offside_lines(pitch_xy, rec.teams, directions, fc.calib, ball_xy,
                                                    rec.players.class_id)
                        draw_offside_lines(mini, minimap, [(ln.x, team_colors.get(t, (200, 200, 200)),
                                                            ln.keeper_assumed) for t, ln in lines.items()])
                    if arrows:
                        draw_attack_arrows(mini, minimap, arrows)
                    paste_minimap(out, mini)
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
    return scores


def summarize_alignment(scores: list[float | None], fps: float, threshold: float = 0.5) -> str:
    measured = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not measured:
        return "Line match: no frames could be checked (not enough pitch markings visible)"
    vals = np.array([s for _, s in measured])
    bad = [i for i, s in measured if s < threshold]
    text = f"Line match: median {np.median(vals):.0%} over {len(measured)} checked frames"
    if bad:
        worst = min(measured, key=lambda m: m[1])
        text += (f"; {len(bad)} frames below {threshold:.0%} (worst: frame {worst[0]}, "
                 f"{worst[0] / fps:.1f}s, {worst[1]:.0%}). Adding a keyframe near there should help")
    return text


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

    directions = infer_attack_directions(team_positions(records, calibs)) if keyframes is not None else None
    if directions is not None:
        print(f"Attacking directions ({directions.confidence:.0%} confidence, {directions.frames_used} frames):")
        for team in sorted(directions.directions):
            color = _hex(assigner.team_colors_bgr.get(team, (200, 200, 200)))
            print(f"  team {team} (shirt color {color}) attacks {directions.describe(team)}")
        if directions.confidence < 0.7:
            print("  Low confidence: the clip may cover half-time, or the team colors may be mixed up")
    elif keyframes is not None:
        print("Attacking directions: not enough calibrated frames with players from both teams")

    mm_cfg = config.get("minimap") or {}
    minimap = None
    if keyframes is not None and mm_cfg.get("enabled", True):
        pitch = next(iter(keyframes.keyframes.values())).pitch
        minimap = Minimap(pitch, width_px=int(info.width * mm_cfg.get("width_fraction", 0.3)))

    csv_path = (config.get("export") or {}).get("tracks_csv")
    debug = bool((config.get("debug") or {}).get("pitch_overlay", False))
    scores = render(src, dst, info, records, calibs, assigner.team_colors_bgr, minimap, csv_path,
                    debug=debug, cuts=cuts, directions=directions)
    print(f"Saved {dst}" + (f" and {csv_path}" if csv_path else ""))
    if debug and keyframes is not None:
        print(summarize_alignment(scores, info.fps))


def main() -> None:
    parser = argparse.ArgumentParser(description="PitchLens player tracking")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="override input video path")
    parser.add_argument("--output", help="override output video path")
    parser.add_argument("--calibration", help="override calibration JSON path")
    parser.add_argument("--debug", action="store_true",
                        help="draw calibrated pitch lines + status on every frame and check line alignment")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.input:
        config["video"]["input"] = args.input
    if args.output:
        config["video"]["output"] = args.output
    if args.calibration:
        config.setdefault("calibration", {})["path"] = args.calibration
    if args.debug:
        config.setdefault("debug", {})["pitch_overlay"] = True
    run(config)


if __name__ == "__main__":
    main()
