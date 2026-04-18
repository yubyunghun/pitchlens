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
from pitchlens.calibration.keypoint_model import keyframes_from_predictions
from pitchlens.calibration.shot_cuts import AppearanceChange, find_cuts
from pitchlens.detection.classes import CLASS_NAMES, REFEREE
from pitchlens.offside.direction import DirectionEstimate, infer_attack_directions
from pitchlens.offside.line import active_defending_team, frame_offside_lines
from pitchlens.offside.body_parts import positions_toward_goal
from pitchlens.offside.incidents import (banner_frames, find_offside_incidents, summarize_incidents,
                                         write_incidents_csv)
from pitchlens.offside.passes import FramePlayers, detect_touches, find_passes
from pitchlens.offside.position import DEFAULT_TOLERANCE_M, check_frame
from pitchlens.tactics.lane_options import frame_options, write_options_csv
from pitchlens.tactics.pass_choice import analyze_passes, summarize_passing, team_passing, write_pass_choices_csv
from pitchlens.tactics.passing_lanes import LaneModel
from pitchlens.calibration.pitch import Pitch
from pitchlens.report.match_report import ReportData, TeamInfo, write_report
from pitchlens.tactics.possession import ALL, possession_by_frame
from pitchlens.tactics.report import (formation_chart, formation_report, save_formation_charts,
                                      summarize_formations, write_average_positions_csv)
from pitchlens.tactics.shape_metrics import shape_metrics, summarize_metrics, write_metrics_csv
from pitchlens.tactics.timeline import merge_segments, summarize_timeline, window_formations, write_timeline_csv
from pitchlens.tracking.team_assigner import UNKNOWN, TeamAssigner
from pitchlens.tracking.tracker import PlayerTracker
from pitchlens.visualization.annotator import annotate, player_color
from pitchlens.visualization.minimap import (Minimap, draw_attack_arrows, draw_offside_lines,
                                             draw_player_rings, draw_projected_pitch, paste_minimap)
from pitchlens.visualization.lanes_overlay import draw_lanes_minimap, draw_lanes_video
from pitchlens.visualization.offside_overlay import (draw_decision_banner, draw_offside_overlay,
                                                     draw_position_flags, flag_color)

CSV_HEADER = ["frame", "track_id", "class", "team", "x_px", "y_px", "x_m", "y_m", "calib_distance",
              "offside_status", "offside_margin_m"]


@dataclass
class FrameRecord:
    players: sv.Detections       # tracked players/goalkeepers/referees
    teams: np.ndarray            # team per player (UNKNOWN for referees)
    ball: sv.Detections
    motion: np.ndarray | None    # camera motion from the previous frame (None if unknown)
    change: float = 0.0          # appearance change from the previous frame (0..1), for cut detection
    pitch_keypoints: np.ndarray | None = None   # (33, 3) landmark predictions, on frames where they were run
    poses: list | None = None    # per tracked player: (17, 3) body keypoints or None (offside by body parts)


def toward_goal(rec: "FrameRecord", calib, pitch_xy: np.ndarray) -> dict[bool, np.ndarray] | None:
    """Body-part positions toward each goal for a frame with poses (None without poses / calibration)."""
    if rec.poses is None or calib is None or len(rec.poses) != len(pitch_xy):
        return None
    return {g: positions_toward_goal(pitch_xy, rec.poses, rec.players.xyxy, calib, goal_at_left=g)
            for g in (True, False)}


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
        print("No manual calibration file found (run pitchlens-calibrate to create one, "
              "or set calibration.auto_model for automatic calibration)")
        return None
    calib_set = CalibrationSet.load(path)
    if not len(calib_set):
        print(f"{path} has no keyframes; skipping minimap and pitch coordinates")
        return None
    return calib_set.for_image_size(info.width, info.height)


def analyze(video: str, detector, assigner: TeamAssigner, fps: float, track_camera: bool = True,
            total: int | None = None, keypoint_detector=None, keypoint_every: int = 25,
            pose_detector=None, detect_every: int = 1) -> list[FrameRecord]:
    """Pass 1: detection, tracking, teams and camera motion for every frame.

    keypoint_detector: optional pitch-landmark model, run on every `keypoint_every`-th frame
                       (its predictions become automatic calibration keyframes).
    pose_detector:     optional body-pose model, run on every tracked player (offside by body parts).
    detect_every:      run the detector (and pose model) only on every Nth frame, for speed; in
                       between, players move by their tracked velocity and the ball path fills the gap.
    """
    tracker = PlayerTracker(fps=fps)
    camera = CameraMotionTracker() if track_camera else None
    appearance = AppearanceChange()
    detect_every = max(1, int(detect_every))
    records = []
    last_detected = 0
    for t, frame in enumerate(tqdm(sv.get_video_frames_generator(video), total=total, desc="Analyzing")):
        detected = t % detect_every == 0
        if detected:
            dets = detector.detect(frame)
            players, ball = tracker.update(dets.players), dets.ball
            # cut out everything that moves by itself (raw detections, so untracked players count too)
            moving = np.vstack([dets.players.xyxy.reshape(-1, 4), dets.ball.xyxy.reshape(-1, 4)])
            last_detected = t
        else:
            players, ball = tracker.coast(t - last_detected), sv.Detections.empty()
            moving = players.xyxy.reshape(-1, 4)
        teams = np.full(len(players), UNKNOWN, dtype=int)
        not_ref = players.class_id != REFEREE
        if not_ref.any():
            teams[not_ref] = assigner.assign(frame, players.xyxy[not_ref], players.tracker_id[not_ref])
        motion = None
        if camera is not None:
            result = camera.update(frame, moving)
            motion = result.H if result is not None else None
        keypoints = None                                     # None = model not run on this frame
        if keypoint_detector is not None and t % max(1, keypoint_every) == 0:
            keypoints = keypoint_detector.predict(frame)
            if keypoints is None:
                keypoints = np.zeros((0, 3))                 # run, but no pitch found
        poses = None
        if pose_detector is not None and detected:
            poses = pose_detector.predict_players(frame, players.xyxy)
        records.append(FrameRecord(players, teams, ball, motion, appearance.update(frame), keypoints, poses))
    return records


def auto_keyframes(records: list[FrameRecord], image_size: tuple[int, int], pitch: Pitch = Pitch(),
                   min_confidence: float = 0.5) -> tuple[CalibrationSet, dict[int, str]]:
    """Keyframes from the landmark predictions stored in the records (accepted, frame -> rejection reason)."""
    predictions = {t: r.pitch_keypoints for t, r in enumerate(records) if r.pitch_keypoints is not None}
    return keyframes_from_predictions(predictions, image_size, pitch, min_confidence=min_confidence)


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


def offside_inputs(records: list[FrameRecord], calibs: list[FrameCalibration | None]):
    """Per-frame inputs for the offside chain: players, raw ball positions, calibrations, classes."""
    players, ball_raw, frame_calibs, classes = [], [], [], []
    for rec, fc in zip(records, calibs):
        feet = foot_points(rec.players.xyxy)
        pos = fc.calib.to_pitch(feet) if fc else np.full_like(feet, np.nan)
        ids = rec.players.tracker_id if rec.players.tracker_id is not None else np.zeros(0, int)
        players.append(FramePlayers(pos, rec.teams, ids, toward_goal(rec, fc.calib if fc else None, pos)))
        ball_raw.append(fc.calib.to_pitch(foot_points(rec.ball.xyxy))[0] if fc and len(rec.ball) else None)
        frame_calibs.append(fc.calib if fc else None)
        classes.append(rec.players.class_id)
    return players, ball_raw, frame_calibs, classes


def formations_from_inputs(inputs, ball_path, directions: DirectionEstimate, fps: float, pitch,
                           min_seconds: float = 2.0):
    """Formation report from the offside inputs (players per frame) and the cleaned ball path."""
    frame_players, _, _, classes = inputs
    possession = possession_by_frame(detect_touches(ball_path, frame_players), len(frame_players), fps)
    return formation_report(frame_players, directions.directions, possession, classes, pitch,
                            min_frames=int(round(min_seconds * fps)))


def metrics_from_inputs(inputs, ball_path, directions: DirectionEstimate, fps: float, pitch, report):
    """Team shape metrics per phase; goalkeepers found by the formation report are left out."""
    frame_players = inputs[0]
    possession = possession_by_frame(detect_touches(ball_path, frame_players), len(frame_players), fps)
    keepers = {team: set(phases[ALL].shape.goalkeeper.track_ids)
               for team, phases in report.items() if phases[ALL].shape.goalkeeper is not None}
    return shape_metrics(frame_players, directions.directions, possession, keepers, pitch)


def options_from_inputs(inputs, ball_path, directions: DirectionEstimate, fps: float, lane_cfg: dict):
    """Passing options of the team on the ball, per frame (see tactics/lane_options.py)."""
    frame_players = inputs[0]
    possession = possession_by_frame(detect_touches(ball_path, frame_players), len(frame_players), fps)
    model = LaneModel(pass_speed=float(lane_cfg.get("pass_speed", 15.0)),
                      run_speed=float(lane_cfg.get("run_speed", 7.0)),
                      reaction=float(lane_cfg.get("reaction_s", 0.3)))
    return frame_options(frame_players, ball_path.positions, possession, directions.directions, model)


def render_freeze_frame(frame: np.ndarray, decision, rec: FrameRecord, fc: FrameCalibration,
                        team_colors: dict, minimap: Minimap | None, fps: float) -> np.ndarray:
    """VAR still of the kick frame: the defending team's line, offside-position tags, verdict banner."""
    line = decision.frozen.line
    defending = line.defending_team
    color = team_colors.get(defending, (200, 200, 200))
    feet = foot_points(rec.players.xyxy)
    ids = list(rec.players.tracker_id) if rec.players.tracker_id is not None else []
    defender_foot = None if line.defender_index is None else feet[line.defender_index]
    out = draw_offside_overlay(frame.copy(), fc.calib, line, color, occluders=rec.players.xyxy,
                               defender_foot=defender_foot)
    out = annotate(out, rec.players, rec.teams, team_colors, rec.ball)
    # tag the passer's teammates judged in this decision (by track id -> index in this frame)
    checks = {ids.index(tid): c for tid, c in decision.frozen.checks.items() if tid in ids}
    draw_position_flags(out, rec.players.xyxy, checks)
    receiver = decision.pass_.receive.track_id if decision.pass_.receive else None
    if receiver in ids:
        x1, y1, x2, y2 = rec.players.xyxy[ids.index(receiver)].astype(int)
        cv2.rectangle(out, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 2, cv2.LINE_AA)
    draw_decision_banner(out, decision, fps)
    if minimap is not None:
        pitch_xy = fc.calib.to_pitch(feet)
        colors = [player_color(c, t, team_colors) for c, t in zip(rec.players.class_id, rec.teams)]
        mini = minimap.render(pitch_xy, colors, ball=decision.pass_.kick.ball)
        draw_offside_lines(mini, minimap, [(line.x, color, line.keeper_assumed)])
        flagged = [(i, flag_color(c.status)) for i, c in checks.items() if flag_color(c.status)]
        if flagged:
            draw_player_rings(mini, minimap, pitch_xy[[i for i, _ in flagged]], [c for _, c in flagged])
        paste_minimap(out, mini)
    return out


def freeze_frame_name(decision, fps: float) -> str:
    t = decision.pass_.kick.frame / fps
    receiver = decision.pass_.receive.track_id if decision.pass_.receive else "unknown"
    return f"{int(t // 60):02d}m{t % 60:04.1f}s_{decision.verdict}_player{receiver}.png"


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
           directions: DirectionEstimate | None = None, offside_in_video: bool = True,
           offside_tolerance_m: float = DEFAULT_TOLERANCE_M, ball_path_xy: np.ndarray | None = None,
           banners: dict | None = None, freeze: dict | None = None,
           freeze_dir: str | None = None, lane_options: list | None = None,
           lanes_in_video: bool = True, lanes_on_minimap: bool = True) -> list[float | None]:
    """Pass 2: draw overlays and minimap, write the video and the positions CSV.

    debug:            draw the calibrated pitch lines and a status box on every frame, and
                      measure how well the lines match the video (returned per frame).
    directions:       if known, the minimap shows attack directions and offside lines.
    offside_in_video: also draw the offside line of the team defending near the ball
                      into the video, VAR-style, and tag attackers in an offside position.
    offside_tolerance_m: attackers this close to the line are "too close to call".
    ball_path_xy:     cleaned ball path (pitch meters per frame) used instead of raw detections.
    banners:          frame -> offside decision to announce on that frame.
    freeze:           kick frame -> decisions to save a VAR freeze-frame image for (into freeze_dir).
    lane_options:     per frame, the passing options of the team on the ball (or None);
                      drawn in the video and/or on the minimap.
    """
    banners = banners or {}
    freeze = freeze or {}
    if freeze and freeze_dir:
        Path(freeze_dir).mkdir(parents=True, exist_ok=True)
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
                if freeze_dir and fc:
                    for decision in freeze.get(idx, []):
                        still = render_freeze_frame(frame, decision, rec, fc, team_colors, minimap, info.fps)
                        cv2.imwrite(str(Path(freeze_dir) / freeze_frame_name(decision, info.fps)), still)
                feet = foot_points(rec.players.xyxy)
                pitch_xy = fc.calib.to_pitch(feet) if fc else np.full_like(feet, np.nan)
                # the ball is sometimes in the air, so its ground position is only approximate
                ball_xy = fc.calib.to_pitch(foot_points(rec.ball.xyxy))[0] if fc and len(rec.ball) else None
                if fc and ball_path_xy is not None and np.isfinite(ball_path_xy[idx]).all():
                    ball_xy = ball_path_xy[idx]       # cleaned path: fewer misses, no false detections
                lines = {}
                body = toward_goal(rec, fc.calib if fc else None, pitch_xy)
                if fc and directions is not None:
                    lines = frame_offside_lines(pitch_xy, rec.teams, directions, fc.calib, ball_xy,
                                                rec.players.class_id, body)
                checks = check_frame(pitch_xy, rec.teams, lines, offside_tolerance_m, body) if lines else {}

                base = frame
                active = active_defending_team(lines, ball_xy, pitch_xy) if offside_in_video else None
                if active is not None:  # drawn first so players and their markers stay on top
                    ln = lines[active]
                    base = draw_offside_overlay(
                        frame.copy(), fc.calib, ln, team_colors.get(active, (200, 200, 200)),
                        occluders=rec.players.xyxy,
                        defender_foot=None if ln.defender_index is None else feet[ln.defender_index])
                opts = lane_options[idx] if lane_options is not None and idx < len(lane_options) else None
                if opts is not None and fc and lanes_in_video:   # also behind the players
                    base = draw_lanes_video(base.copy() if base is frame else base, fc.calib, pitch_xy, opts,
                                            occluders=rec.players.xyxy)
                out = annotate(base, rec.players, rec.teams, team_colors, rec.ball)
                if checks and offside_in_video:
                    draw_position_flags(out, rec.players.xyxy, checks)
                if idx in banners:
                    draw_decision_banner(out, banners[idx], info.fps)
                score = None
                if debug:
                    if fc:
                        score = line_alignment(frame, fc.calib)  # measured on the clean frame
                        draw_projected_pitch(out, fc.calib, (0, 255, 255), 1)
                    _draw_debug_hud(out, idx, fc, score, idx in cut_set)
                scores.append(score)
                if minimap and fc:
                    colors = [player_color(c, t, team_colors) for c, t in zip(rec.players.class_id, rec.teams)]
                    mini = minimap.render(pitch_xy, colors, ball=ball_xy)
                    if opts is not None and lanes_on_minimap:
                        draw_lanes_minimap(mini, minimap, pitch_xy, opts)
                        minimap.draw_players(mini, pitch_xy, colors, ball=ball_xy)   # dots on top of lanes
                    if lines:
                        draw_offside_lines(mini, minimap, [(ln.x, team_colors.get(t, (200, 200, 200)),
                                                            ln.keeper_assumed) for t, ln in lines.items()])
                    flagged = [(i, flag_color(c.status)) for i, c in checks.items() if flag_color(c.status)]
                    if flagged:
                        draw_player_rings(mini, minimap, pitch_xy[[i for i, _ in flagged]], [c for _, c in flagged])
                    if arrows:
                        draw_attack_arrows(mini, minimap, arrows)
                    paste_minimap(out, mini)
                sink.write_frame(out)

                if writer:
                    for i, (tid, cls, team, (px, py), (mx, my)) in enumerate(
                            zip(rec.players.tracker_id, rec.players.class_id, rec.teams, feet, pitch_xy)):
                        check = checks.get(i)
                        writer.writerow([idx, int(tid), CLASS_NAMES[int(cls)], int(team),
                                         round(float(px), 1), round(float(py), 1),
                                         "" if np.isnan(mx) else round(float(mx), 2),
                                         "" if np.isnan(my) else round(float(my), 2),
                                         "" if fc is None else fc.distance,
                                         "" if check is None else check.status,
                                         "" if check is None else round(check.margin_m, 2)])
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
                        det_cfg.get("ball_confidence", 0.15), det_cfg.get("device"),
                        ball_tiles=bool(det_cfg.get("ball_tiles", False)),
                        tile_size=int(det_cfg.get("tile_size", 640)),
                        tile_overlap=float(det_cfg.get("tile_overlap", 0.2)))
    cam_cfg = config.get("camera_motion") or {}
    track_camera = cam_cfg.get("enabled", True)

    print("Learning team colors...")
    assigner = fit_teams(detector, src)

    info = sv.VideoInfo.from_video_path(src)
    cal_cfg = config.get("calibration") or {}
    keypoint_detector = None
    if cal_cfg.get("auto_model"):
        from pitchlens.calibration.keypoint_model import KeypointDetector
        keypoint_detector = KeypointDetector(cal_cfg["auto_model"], det_cfg.get("device"))
    keypoint_every = max(1, int(round(float(cal_cfg.get("auto_every_s", 1.0)) * info.fps)))
    pose_detector = None
    if (config.get("offside") or {}).get("body_parts_model"):
        from pitchlens.offside.pose_model import PoseDetector
        pose_detector = PoseDetector(config["offside"]["body_parts_model"], det_cfg.get("device"))
    records = analyze(src, detector, assigner, info.fps, track_camera, info.total_frames,
                      keypoint_detector, keypoint_every, pose_detector,
                      detect_every=int(det_cfg.get("detect_every", 1)))

    cuts = detect_cuts(records) if track_camera else []
    if cuts:
        stamps = ", ".join(f"frame {c} ({c / info.fps:.1f}s)" for c in cuts[:10])
        print(f"Detected {len(cuts)} camera cut(s): {stamps}{' ...' if len(cuts) > 10 else ''}")
        print("  Calibrate one frame after each cut so that shot gets pitch coordinates too")

    keyframes = load_keyframes(cal_cfg.get("path"), info)
    if keypoint_detector is not None:
        auto, rejected = auto_keyframes(records, (info.width, info.height),
                                        min_confidence=float(cal_cfg.get("auto_min_confidence", 0.5)))
        print(f"Automatic calibration: {len(auto)} keyframe(s) from {len(auto) + len(rejected)} checked frames")
        if rejected:
            common = max(set(rejected.values()), key=list(rejected.values()).count)
            print(f"  {len(rejected)} rejected, most often: {common}")
        keyframes = (keyframes.merged_with(auto) if keyframes is not None else auto) if len(auto) else keyframes
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

    off_cfg = config.get("offside") or {}
    tolerance = float(off_cfg.get("tolerance_m", DEFAULT_TOLERANCE_M))
    export = config.get("export") or {}
    ball_path_xy, banners, freeze, lane_options = None, {}, {}, None
    decisions, report, segments, metrics, passing_stats, pitch = [], None, [], [], [], None
    if directions is not None:
        inputs = offside_inputs(records, calibs)
        path, _, decisions = find_offside_incidents(*inputs, directions, info.fps, tolerance)
        ball_path_xy = path.positions
        print(summarize_incidents(decisions, info.fps))
        if export.get("incidents_csv"):
            write_incidents_csv(export["incidents_csv"], decisions, info.fps)
            print(f"  Saved {export['incidents_csv']}")
        banners = banner_frames(decisions, info.fps, len(records))
        for d in decisions:
            if d.verdict in ("offside", "too_close") and d.frozen.line is not None:
                freeze.setdefault(d.pass_.kick.frame, []).append(d)

        # formations: average shape per team, with and without the ball
        pitch = next(iter(keyframes.keyframes.values())).pitch
        min_seconds = float((config.get("tactics") or {}).get("min_seconds", 2.0))
        report = formations_from_inputs(inputs, path, directions, info.fps, pitch, min_seconds)
        names = {t: f"team {t} ({_hex(assigner.team_colors_bgr.get(t, (200, 200, 200)))})" for t in report}
        print(summarize_formations(report, names))
        if export.get("average_positions_csv"):
            write_average_positions_csv(export["average_positions_csv"], report)
            print(f"  Saved {export['average_positions_csv']}")
        if export.get("formation_charts"):
            saved = save_formation_charts(export["formation_charts"], report, assigner.team_colors_bgr, pitch)
            print(f"  Saved {len(saved)} formation chart(s) to {export['formation_charts']}/")

        tac_cfg = config.get("tactics") or {}
        segments = merge_segments(window_formations(
            inputs[0], directions.directions, info.fps, inputs[3], pitch,
            window_s=float(tac_cfg.get("timeline_window_s", 15.0)),
            step_s=float(tac_cfg.get("timeline_step_s", 5.0)),
            min_seconds=float(tac_cfg.get("timeline_min_seconds", 3.0))))
        print(summarize_timeline(segments, info.fps, names))
        if export.get("formation_timeline_csv"):
            write_timeline_csv(export["formation_timeline_csv"], segments, info.fps)
            print(f"  Saved {export['formation_timeline_csv']}")

        metrics = metrics_from_inputs(inputs, path, directions, info.fps, pitch, report)
        print(summarize_metrics(metrics, names))
        if export.get("shape_metrics_csv"):
            write_metrics_csv(export["shape_metrics_csv"], metrics)
            print(f"  Saved {export['shape_metrics_csv']}")

        lane_options = options_from_inputs(inputs, path, directions, info.fps, config.get("passing_lanes") or {})
        if export.get("passing_options_csv"):
            n = write_options_csv(export["passing_options_csv"], lane_options, info.fps)
            print(f"Passing options: {n} frames with a team on the ball; saved {export['passing_options_csv']}")

        choices = analyze_passes(find_passes(detect_touches(path, inputs[0])), lane_options)
        passing_stats = team_passing(choices, lane_options, sorted(directions.directions))
        print(summarize_passing(passing_stats, names))
        if export.get("pass_choices_csv"):
            write_pass_choices_csv(export["pass_choices_csv"], choices, info.fps)
            print(f"  Saved {export['pass_choices_csv']}")

    mm_cfg = config.get("minimap") or {}
    minimap = None
    if keyframes is not None and mm_cfg.get("enabled", True):
        pitch = next(iter(keyframes.keyframes.values())).pitch
        minimap = Minimap(pitch, width_px=int(info.width * mm_cfg.get("width_fraction", 0.3)))

    csv_path = export.get("tracks_csv")
    debug = bool((config.get("debug") or {}).get("pitch_overlay", False))
    lane_cfg = config.get("passing_lanes") or {}
    scores = render(src, dst, info, records, calibs, assigner.team_colors_bgr, minimap, csv_path,
                    debug=debug, cuts=cuts, directions=directions,
                    offside_in_video=bool(off_cfg.get("draw_in_video", True)),
                    offside_tolerance_m=tolerance, ball_path_xy=ball_path_xy, banners=banners,
                    freeze=freeze, freeze_dir=export.get("incident_images"), lane_options=lane_options,
                    lanes_in_video=bool(lane_cfg.get("draw_in_video", True)),
                    lanes_on_minimap=bool(lane_cfg.get("draw_on_minimap", True)))
    print(f"Saved {dst}" + (f" and {csv_path}" if csv_path else ""))
    if freeze and export.get("incident_images"):
        n = sum(len(v) for v in freeze.values())
        print(f"Saved {n} VAR freeze-frame image(s) to {export['incident_images']}/")
    if debug and keyframes is not None:
        print(summarize_alignment(scores, info.fps))

    if export.get("report_html"):
        data = build_report_data(src, info.fps, len(records), assigner.team_colors_bgr, directions, calibs,
                                 0 if keyframes is None else len(keyframes), cuts, report, segments, metrics,
                                 passing_stats, decisions, export.get("incident_images"), pitch)
        print(f"Saved match report {write_report(export['report_html'], data)}")


def build_report_data(video: str, fps: float, n_frames: int, team_colors: dict,
                      directions: DirectionEstimate | None, calibs: list, n_keyframes: int, cuts: list[int],
                      formations, segments, metrics, passing_stats, decisions,
                      freeze_dir: str | None = None, pitch=None) -> ReportData:
    """Collects a run's results for the HTML report (freeze-frames are read back from freeze_dir)."""
    teams = {}
    for t in (sorted(directions.directions) if directions is not None else sorted(team_colors)):
        teams[t] = TeamInfo(f"team {t}", tuple(int(v) for v in team_colors.get(t, (200, 200, 200))),
                            None if directions is None else directions.describe(t))
    charts = {}
    if formations:
        for t in formations:
            charts[t] = formation_chart(formations[t], teams[t].color_bgr if t in teams else (200, 200, 200),
                                        pitch or Pitch())
    images = {}
    if freeze_dir:
        for i, d in enumerate(decisions):
            path = Path(freeze_dir) / freeze_frame_name(d, fps)
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    images[i] = img
    return ReportData(video=video, fps=fps, frames=n_frames, teams=teams,
                      calibrated_frames=sum(c is not None for c in calibs), keyframes=n_keyframes, cuts=list(cuts),
                      direction_confidence=None if directions is None else directions.confidence,
                      formations=formations, formation_charts=charts, timeline=list(segments),
                      metrics=list(metrics), passing=list(passing_stats), offside=list(decisions),
                      offside_images=images)


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
