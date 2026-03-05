# PitchLens

AI-powered football VAR and tactical analysis from match footage, built with Python, OpenCV and YOLO.

## Planned features
- **Player and ball tracking**: YOLO detection + ByteTrack IDs, team assignment by jersey color
- **Offside detection**: pitch homography, second-last defender line, offside flag at pass moment
- **Formation mapping**: top-down 2D pitch view, average positions, shape detection (4-4-2, 4-3-3, ...)
- **Passing lanes**: open/blocked lanes between teammates, pass network graphs

## Layout
```
src/pitchlens/
  detection/      YOLO player/ball/referee detection
  tracking/       multi-object tracking + team classification
  calibration/    pitch keypoints -> homography to 2D pitch
  offside/        offside line + decision logic
  tactics/        formations, passing lanes, pass networks
  visualization/  overlays, minimap, heatmaps
configs/          YAML settings
data/raw/         input match footage (not committed)
models/           YOLO weights (not committed)
outputs/          annotated videos and reports
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Calibrate the pitch
Player positions in meters (needed for the minimap, offside, and formations) come from
mapping the camera view onto a 105 x 68 m pitch:
```bash
pitchlens-calibrate --video data/raw/match.mp4 --frame 0
```
Two windows open. Click a landmark on the video frame (a box corner, the halfway line meeting
the touchline, the center spot, ...), then click the same landmark on the pitch diagram.
Repeat for **4+ landmarks spread across the visible pitch**. From 4 pairs the pitch lines are
drawn over the frame in yellow: if they sit on the real lines, the calibration is good.
Each landmark shows its error in meters; `u` undoes a bad click. Press `s` to save it as a
keyframe in `configs/calibration.json` (plus a `.png` preview).

### Moving cameras and keyframes
You don't need to calibrate every frame. The pipeline measures how the camera pans and zooms
between frames and carries the calibration along, forward and backward from each keyframe.
Each run of `pitchlens-calibrate` adds one more keyframe:
```bash
pitchlens-calibrate --time 0:00      # start of the clip
pitchlens-calibrate --time 0:42      # first frame after a camera cut
pitchlens-calibrate --list           # saved keyframes
pitchlens-calibrate --remove 1050    # drop one
```
Calibrate one frame per camera shot; extra keyframes in long shots reduce drift further.
Camera cuts (close-ups, replays, switching cameras) are detected automatically, and
`pitchlens-track` prints their timestamps so you know which shots still need a keyframe.
Frames that can't be linked to any keyframe (e.g. a shot you didn't calibrate) are left
without pitch coordinates rather than given wrong ones.

## Run player tracking
Put a clip at `data/raw/match.mp4`, then:
```bash
pitchlens-track                                   # uses configs/default.yaml
pitchlens-track --input data/raw/clip.mp4 --output outputs/clip.mp4
```
Output: each player gets an ellipse in their team's color and a tracking ID; referees show as `REF` and the ball gets a triangle marker. With a calibration file, a top-down minimap is added and `outputs/tracks.csv` gets every player's position in meters per frame (`calib_distance` = frames since the nearest keyframe; lower is more accurate).

The video is read twice: first for detection, tracking and camera motion, then to draw the output with each frame's calibration.

It also works out which goal each team attacks (from team shape and each team's deepest
players) and prints it with each team's shirt color; the minimap shows it as an arrow per team.

The minimap also draws each team's **offside line** in its color: the second-last defender,
or the ball if that is nearer the goal, never past halfway (Law 11). When the goal area is
off-screen and no keeper is seen, the keeper is assumed to be there and the line is dashed.

### Checking the calibration
```bash
pitchlens-track --debug
```
Draws the calibrated pitch lines in yellow on every frame, with a status box showing which
keyframe the frame is chained from and a **line match** score: the share of projected lines
that land on real white lines in the video (green 70%+, yellow 50-70%, red below 50%).
At the end it prints a summary with the worst frames, which are good places to add a keyframe.
