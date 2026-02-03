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
pip install -r requirements.txt
```
