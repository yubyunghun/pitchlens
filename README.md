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
Each landmark shows its error in meters; `u` undoes a bad click. Press `s` to save
`configs/calibration.json` (plus a `.png` preview).

> The calibration is valid while the camera stays still. A panning broadcast camera needs
> per-frame calibration, which is on the roadmap.

## Run player tracking
Put a clip at `data/raw/match.mp4`, then:
```bash
pitchlens-track                                   # uses configs/default.yaml
pitchlens-track --input data/raw/clip.mp4 --output outputs/clip.mp4
```
Output: each player gets an ellipse in their team's color and a tracking ID; referees show as `REF` and the ball gets a triangle marker. With a calibration file, a top-down minimap is added and `outputs/tracks.csv` gets every player's position in meters per frame.
