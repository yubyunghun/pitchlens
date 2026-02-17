"""Interactive pitch calibration.

Usage:
    pitchlens-calibrate --video data/raw/match.mp4 --frame 0
    pitchlens-calibrate --video data/raw/match.mp4 --time 1:23   # adds another keyframe
    pitchlens-calibrate --list

Two windows open:
  * "frame": the video frame
  * "pitch": a top-down pitch with every known landmark as a dot

Click a landmark on the frame (e.g. a penalty-box corner), then click the same
landmark on the pitch (it snaps to the nearest dot). Repeat for 4+ landmarks
spread across the visible pitch. From 4 pairs the pitch lines are drawn over
the frame so you can see whether they line up.

Keys: u = undo, s = save, q / Esc = quit without saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pitchlens.calibration.homography import PitchCalibration
from pitchlens.calibration.keyframes import CalibrationSet
from pitchlens.calibration.pitch import Pitch
from pitchlens.visualization.minimap import Minimap, draw_projected_pitch

FRAME_WIN, PITCH_WIN = "frame", "pitch"
SNAP_METERS = 4.0
PENDING = (0, 0, 255)
PAIRED = (0, 220, 0)
HELP = "click frame landmark -> click same landmark on pitch | u undo | s save | q quit"


def read_frame(video: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {index} from {video}")
    return frame


class CalibrationTool:
    def __init__(self, frame: np.ndarray, pitch: Pitch = Pitch(), max_width: int = 1280):
        self.frame = frame
        self.view_scale = min(1.0, max_width / frame.shape[1])
        self.pitch = pitch
        self.map = Minimap(pitch, width_px=900, margin_m=4.0)
        kp = pitch.keypoints()
        self.kp_names = list(kp)
        self.kp_xy = np.array(list(kp.values()))
        self.pairs: list[tuple[np.ndarray, int]] = []  # (image xy, keypoint index)
        self.pending_img: np.ndarray | None = None
        self.pending_kp: int | None = None
        self.calib: PitchCalibration | None = None

    # ---- input -----------------------------------------------------------
    def on_frame_click(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_img = np.array([x, y], dtype=float) / self.view_scale
            self._try_pair()

    def on_pitch_click(self, event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        dist = np.linalg.norm(self.kp_xy - self.map.from_px((x, y)), axis=1)
        if dist.min() <= SNAP_METERS:
            self.pending_kp = int(dist.argmin())
            self._try_pair()

    def undo(self):
        if self.pending_img is not None or self.pending_kp is not None:
            self.pending_img = self.pending_kp = None
        elif self.pairs:
            self.pairs.pop()
        self._recompute()

    def _try_pair(self):
        if self.pending_img is None or self.pending_kp is None:
            return
        self.pairs = [p for p in self.pairs if p[1] != self.pending_kp]  # re-clicking replaces
        self.pairs.append((self.pending_img, self.pending_kp))
        self.pending_img = self.pending_kp = None
        self._recompute()

    def _recompute(self):
        self.calib = None
        if len(self.pairs) >= 4:
            h, w = self.frame.shape[:2]
            try:
                self.calib = PitchCalibration.from_correspondences(
                    [p[0] for p in self.pairs], self.kp_xy[[p[1] for p in self.pairs]], (w, h),
                    self.pitch, [self.kp_names[p[1]] for p in self.pairs])
            except ValueError as e:
                print(e)

    # ---- drawing ---------------------------------------------------------
    def render_frame(self) -> np.ndarray:
        img = self.frame.copy()
        errors = self.calib.point_errors() if self.calib else None
        if self.calib:
            draw_projected_pitch(img, self.calib, (0, 255, 255), max(1, int(2 / self.view_scale)))
        for n, (xy, _) in enumerate(self.pairs):
            p = tuple(np.round(xy).astype(int))
            cv2.circle(img, p, int(7 / self.view_scale), PAIRED, -1, cv2.LINE_AA)
            label = f"{n + 1}" + (f" ({errors[n]:.1f}m)" if errors is not None else "")
            cv2.putText(img, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 / self.view_scale, PAIRED, max(1, int(2 / self.view_scale)), cv2.LINE_AA)
        if self.pending_img is not None:
            cv2.circle(img, tuple(np.round(self.pending_img).astype(int)), int(9 / self.view_scale),
                       PENDING, 2, cv2.LINE_AA)
        img = cv2.resize(img, None, fx=self.view_scale, fy=self.view_scale) if self.view_scale < 1 else img
        status = f"{len(self.pairs)} pairs"
        if self.calib:
            status += f" | mean error {errors.mean():.2f} m"
        elif len(self.pairs) < 4:
            status += f" | need {4 - len(self.pairs)} more"
        banner = img[:62].copy()
        banner[:] = (20, 20, 20)
        img[:62] = cv2.addWeighted(banner, 0.6, img[:62], 0.4, 0)
        for i, text in enumerate((HELP, status)):
            cv2.putText(img, text, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def render_pitch(self) -> np.ndarray:
        img = self.map.render(np.empty((0, 2)), [])
        used = {kp: n for n, (_, kp) in enumerate(self.pairs)}
        for i, xy in enumerate(self.kp_xy):
            p = tuple(self.map.to_px(xy))
            if i == self.pending_kp:
                cv2.circle(img, p, 9, PENDING, 2, cv2.LINE_AA)
            if i in used:
                cv2.circle(img, p, 7, PAIRED, -1, cv2.LINE_AA)
                cv2.putText(img, str(used[i] + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, PAIRED, 2, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 5, (0, 200, 255), -1, cv2.LINE_AA)
        if self.pending_kp is not None:
            cv2.putText(img, self.kp_names[self.pending_kp], (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def run(self) -> PitchCalibration | None:
        cv2.namedWindow(FRAME_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(PITCH_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(FRAME_WIN, self.on_frame_click)
        cv2.setMouseCallback(PITCH_WIN, self.on_pitch_click)
        try:
            while True:
                cv2.imshow(FRAME_WIN, self.render_frame())
                cv2.imshow(PITCH_WIN, self.render_pitch())
                key = cv2.waitKey(30) & 0xFF
                if key == ord("u"):
                    self.undo()
                elif key == ord("s"):
                    if self.calib:
                        return self.calib
                    print("Need at least 4 landmark pairs before saving")
                elif key in (ord("q"), 27):
                    return None
                if cv2.getWindowProperty(FRAME_WIN, cv2.WND_PROP_VISIBLE) < 1:
                    return None
        finally:
            cv2.destroyAllWindows()


def parse_time(text: str) -> float:
    """'83', '83.5', '1:23' or '0:01:23' -> seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def video_fps(video: str) -> float:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise SystemExit(f"Could not read the frame rate of {video}")
    return fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the camera view to the pitch. Each run adds one keyframe to the output "
                    "file; calibrate the start of the clip and the first frame after each camera cut.")
    parser.add_argument("--video", default="data/raw/match.mp4")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--frame", type=int, help="frame index to calibrate on (default 0)")
    where.add_argument("--time", help="timestamp to calibrate on, e.g. 83.5 or 1:23")
    parser.add_argument("--output", default="configs/calibration.json")
    parser.add_argument("--length", type=float, default=105.0, help="pitch length (m)")
    parser.add_argument("--width", type=float, default=68.0, help="pitch width (m)")
    parser.add_argument("--list", action="store_true", help="list saved keyframes and exit")
    parser.add_argument("--remove", type=int, metavar="FRAME", help="remove a saved keyframe and exit")
    args = parser.parse_args()

    out = Path(args.output)
    calib_set = CalibrationSet.load(out) if out.exists() else CalibrationSet(video=args.video)

    if args.list or args.remove is not None:
        if args.remove is not None:
            calib_set.remove(args.remove)
            calib_set.save(out)
            print(f"Removed keyframe {args.remove}")
        for f, c in calib_set.keyframes.items():
            errs = c.point_errors()
            err = f"mean error {errs.mean():.2f} m" if len(errs) else ""
            print(f"  frame {f:>6}: {len(c.correspondences)} landmarks {err}")
        if not len(calib_set):
            print("No keyframes saved")
        return

    frame_idx = args.frame or 0
    if args.time is not None:
        frame_idx = int(round(parse_time(args.time) * video_fps(args.video)))

    tool = CalibrationTool(read_frame(args.video, frame_idx), Pitch(args.length, args.width))
    calib = tool.run()
    if calib is None:
        print("Calibration not saved")
        return
    replaced = frame_idx in calib_set.keyframes
    calib_set.video = args.video
    calib_set.add(frame_idx, calib)
    calib_set.save(out)

    errs = calib.point_errors()
    print(f"{'Updated' if replaced else 'Added'} keyframe {frame_idx} in {out} "
          f"({len(calib_set)} keyframes): {len(errs)} landmarks, "
          f"mean error {errs.mean():.2f} m, max {errs.max():.2f} m")
    if errs.max() > 1.5:
        worst = calib.correspondences[int(errs.argmax())]["name"]
        print(f"Warning: '{worst}' is {errs.max():.1f} m off; it may have been clicked on the wrong landmark")
    preview = out.with_name(f"{out.stem}_frame{frame_idx}.png")
    cv2.imwrite(str(preview), draw_projected_pitch(tool.frame.copy(), calib))
    print(f"Preview with projected pitch lines: {preview}")


if __name__ == "__main__":
    main()
