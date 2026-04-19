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

## Train a football model (recommended)
The default COCO model often misses the ball and can't tell goalkeepers or referees from
players. A model trained on football footage fixes that. Get a YOLO-format football dataset
(e.g. a "football players detection" dataset exported from Roboflow as YOLOv8, with classes such
as ball / goalkeeper / player / referee), then:
```bash
pitchlens-dataset datasets/football     # checks labels, classes and ball size; exit code 1 on problems
pitchlens-train --data datasets/football --device 0
```
Training checks the dataset first, picks the image size from the ball size (1280 if the ball
would be under 8 px at 640), fine-tunes `yolov8m.pt` and saves `models/football.pt`. Set
`detection: model: models/football.pt` in the config to use it. A GPU is strongly recommended.

For very small balls, `detection: ball_tiles: true` also searches for the ball on overlapping
full-resolution tiles of each frame (slower, one model run per tile).

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

### Automatic calibration (no clicking)
A pitch-landmark model can find the box corners, line ends and center circle itself. Train one
from clips you have already calibrated by hand: every calibrated frame labels all 33 landmarks.
```bash
pitchlens-keypoint-data --video data/raw/match.mp4 --calibration configs/calibration.json
pitchlens-train --task pose --data datasets/pitch_keypoints
```
Then set `calibration: auto_model: models/pitch_keypoints.pt`. During tracking the model checks
one frame per second (`auto_every_s`); frames where enough landmarks agree become automatic
keyframes, combined with any manual ones (manual wins on the same frame). Shots the model isn't
sure about still need a manual keyframe; the run prints how many frames were rejected and why.

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

In the video itself, the offside line of the team defending near the ball is drawn VAR-style:
a line across the pitch in perspective, behind the players, with a ring on the defender who
sets it. Turn it off with `offside: draw_in_video: false` in the config.

Attackers past the opponents' line are tagged **OFFSIDE POS** with their margin in meters
(also ringed on the minimap); those within `offside: tolerance_m` (default 0.3 m) of the line
are **TOO CLOSE** to call. `tracks.csv` gets `offside_status` and `offside_margin_m` columns.
Being in an offside position is only an offence when a teammate plays the ball; positions
are measured at the feet by default. With `offside: body_parts_model: yolov8m-pose.pt`, a pose
model runs on a crop around every tracked player and offside is judged by the body part nearest
the goal line (head, body, legs and feet; arms don't count), dropped to the ground under it.

### Offside calls
From the ball path, `pitchlens-track` finds every pass, freezes the moment the ball is played,
and judges the receiver: **offside**, **too close to call**, **onside**, **no offence** (ball
to an opponent) or **no decision** (with the reason, e.g. receiver off-screen). It prints a
summary, writes `outputs/incidents.csv` (one row per pass) and shows a banner in the video
when an offside call is made.
For each offside / too-close call it also saves a VAR freeze-frame image of the moment of
the pass to `outputs/incidents/`.

Calls depend on the ball being detected well; the default COCO model often misses it, so a
football-trained model is recommended. Not covered yet: interfering with an opponent without
touching the ball, and restarts exempt from offside (throw-ins, goal kicks, corners).

### Formations
`pitchlens-track` also averages every player's position per team, separately **with the ball**,
**without the ball** and **overall** (possession comes from the detected touches), and matches
each shape to a formation: 4-4-2, 4-3-3, 4-2-3-1, 4-1-4-1, 3-5-2, 3-4-3, 5-3-2 or 5-4-1, with a
confidence (close calls read like `4-2-3-1 (or 4-1-4-1)`). Track fragments of one player are
merged and the goalkeeper is left out. Outputs:
- a printed summary per team and phase,
- `outputs/average_positions.csv`: each player's average position and role per phase,
- `outputs/formations/teamN_formation.png`: the classic average-positions chart, one panel per phase.
- `outputs/formation_timeline.csv` and a printed timeline: the formation in a 15 s window sliding
  through the clip, merged into segments (e.g. `0:00-0:35 4-4-2, 0:35-1:10 3-5-2`), so tactical
  switches show up; unclear windows are reported as `unclear`.

Players must be seen for `tactics: min_seconds` (default 2 s) in a phase to count.

### Team shape numbers
Per team and phase it also measures **width**, **length** (deepest to most advanced outfield
player), **defensive line height** (the back four's distance from their own goal: a high press
vs a low block) and **area** (how compact the team is), as medians over frames where at least 8
outfield players are visible. They are printed and saved to `outputs/shape_metrics.csv`.

### Passing lanes
Whenever a team has the ball at a player's feet, PitchLens checks the lane from the ball to
every teammate: a defender cuts it if they can reach some point of the lane before the ball
*and* before the receiver (who steps in to meet the pass). Lanes are drawn on the grass and on
the minimap: **green** open, **amber** risky, **red** blocked. The thick lane is the best
option: the open pass that gains the most ground, counting each opponent it bypasses (a line
broken) as 8 extra meters. Model speeds are in the config (`passing_lanes:`); it's geometry
only (no body orientation or lofted passes), so "open" means "probably open on the ground".
`outputs/passing_options.csv` has one row per frame on the ball: the carrier, how many lanes are
open / risky / blocked, and the best option (receiver, meters gained, opponents bypassed).

For every detected pass it also checks the options the passer had just before the kick: was the
lane used open, did it go forward, did it break a line, and was it the best option? Per team this
gives a passing profile (completed / forward / line-breaking / best option taken), the average
number of open lanes on the ball, and, on defence, how many open forward lanes the team allowed.
Printed by `pitchlens-track` and saved per pass to `outputs/pass_choices.csv`.

### Speed
Detection is the slowest step. `detection: detect_every: 2` runs it on every other frame only;
in between, players move by their tracked velocity and the ball path fills the gap. On the
synthetic test clip, positions were as accurate as detecting every frame (mean 0.13 m), and
`detect_every: 3` was close (0.16 m); fast, erratic movement will suffer more on real footage.
On a GPU, `detection: batch: 8` runs detection on 8 frames per call. Each run ends with the
time spent per stage (detection, tracking, camera motion, rendering, ...), so you can see
what to tune.

### Match report
At the end of a run, `outputs/report.html` collects everything in one self-contained page
(images embedded, so the single file can be shared): overview, formations with charts,
formation timeline, team shape numbers, passing profile, and offside calls with freeze-frames.

### Checking the calibration
```bash
pitchlens-track --debug
```
Draws the calibrated pitch lines in yellow on every frame, with a status box showing which
keyframe the frame is chained from and a **line match** score: the share of projected lines
that land on real white lines in the video (green 70%+, yellow 50-70%, red below 50%).
At the end it prints a summary with the worst frames, which are good places to add a keyframe.
