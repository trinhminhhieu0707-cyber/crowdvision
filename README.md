# CrowdVision

CrowdVision is a computer vision project for people-flow analysis, occupancy estimation, and crowd-density monitoring. This repository is being developed in phases.

## Phase 1: YOLO people detection

Phase 1 detects people in a single image with a COCO-pretrained YOLO11s model, draws each person bounding box with its confidence score, counts detected people, and saves an annotated image.

YOLO11s is used as the initial detector because it offers a practical accuracy-speed balance for CPU development. The checkpoint is downloaded automatically by Ultralytics the first time it is used and is not committed to Git.

### Concepts

- A **bounding box** is a rectangle around one detected object: `(x1, y1, x2, y2)`.
- A **confidence score** estimates how strongly the detector supports the prediction.
- A **class ID** identifies the predicted COCO class. The COCO `person` class is ID `0`; the module verifies this from model metadata rather than relying on a hardcoded class list.
- The **confidence threshold** removes weak detections before results are returned.
- **IoU** measures overlap between boxes. YOLO uses it during non-maximum suppression to remove duplicate boxes that describe the same person.

### Windows setup

Use Python 3.11 or 3.12. In PowerShell at the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current shell and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Run Phase 1

1. Put a crowd image in `datasets/images/`, for example `datasets/images/crowd.jpg`.
2. Run:

```powershell
python scripts/detect_image.py --input datasets/images/crowd.jpg
```

The result is saved to `outputs/images/people_detected.jpg`.

Optional parameters:

```powershell
python scripts/detect_image.py `
  --input datasets/images/crowd.jpg `
  --output outputs/images/crowd_annotated.jpg `
  --confidence 0.40 `
  --iou 0.45 `
  --device cpu
```

### Test

```powershell
python -m pytest
```

## Current project structure

```text
CrowdVision/
├── datasets/images/       # Local test images, ignored by Git
├── datasets/videos/       # Local test videos, ignored by Git
├── models/yolo/           # Optional local checkpoints, ignored by Git
├── outputs/images/        # Annotated images, ignored by Git
├── outputs/videos/        # Annotated videos, ignored by Git
├── scripts/detect_image.py
├── scripts/detect_video.py
├── src/detection/detector.py
├── src/detection/video_processor.py
├── tests/
├── requirements.txt
└── README.md
```

## Phase 2: video people detection

Phase 2 applies the Phase 1 detector to each video frame. The count shown in each frame means people visible in that frame only. It is not a total number of people who appeared during the video; line-crossing and total footfall are introduced later with tracking.

The output keeps the input video's frame rate for normal playback. `Processing FPS` is the current inference and drawing speed on the computer, while `Source FPS` is the original recording frame rate.

1. Put an MP4 video in `datasets/videos/`, for example `datasets/videos/entrance.mp4`.
2. In the PyCharm run configuration for `scripts/detect_video.py`, use these parameters:

```text
--input datasets/videos/entrance.mp4 --device cpu
```

3. Set the working directory to the project root and run it.

The annotated result is written to `outputs/videos/people_detected.mp4`.

To give the output a distinct filename, add:

```text
--output outputs/videos/entrance_annotated.mp4
```

## Phase 3: ByteTrack people tracking

Phase 3 gives each currently tracked person a Track ID. The ID is intended to remain stable while the person is visible across consecutive frames. It is not a permanent identity: it can change after a long occlusion, missed detection, abrupt camera movement, or when people overlap heavily.

Detection answers, "where are people in this frame?" Tracking answers, "which detection in this frame likely belongs to the same person from earlier frames?" The module passes each consecutive frame to Ultralytics with `persist=True` and explicitly selects the ByteTrack backend.

1. Reuse a video from `datasets/videos/`.
2. In the PyCharm run configuration for `scripts/track_video.py`, use:

```text
--input datasets/videos/entrance.mp4 --device cpu
```

3. Run it and open `outputs/videos/people_tracked.mp4`.

Each box displays a Track ID and confidence. The top label shows the number of currently tracked people, not the number of unique people in the complete video.


## Phase 6: independent CSRNet crowd density analysis

CSRNet is a separate analysis module. It estimates the visible crowd count and spatial density for an image or sampled video frames. It does not assign identities, track people, or count line crossings. Keep these estimates separate from YOLO + ByteTrack People Counting and IN/OUT metrics.

### Checkpoint

Place the trusted CSRNet Part A checkpoint in `models/csrnet/PartAmodel_best.pth.tar` (or pass another path using `--checkpoint`). The module accepts a raw state dictionary or a checkpoint containing a `state_dict`. Do not use untrusted checkpoint files.

### Analyze one image

```powershell
python scripts/analyze_crowd_density.py --mode image --input datasets/images/crowd.jpg --device cpu
```

Outputs are saved under `outputs/crowd_density/crowd/`: a heatmap overlay, a dense-region view, the raw colorized density map, and `density_analysis.json` with the estimated count and top grid regions. Dense-region counts are approximate sums of nonnegative density values in coarse grid cells, not individual detections.

### Analyze a video

```powershell
python scripts/analyze_crowd_density.py --mode video --input datasets/videos/people.mp4 --sample-every 15 --device cpu
```

The default analyzes every 15th frame and carries the most recent estimate on intervening frames for display. Use `--sample-every 1` to run CSRNet on every frame (slower). Outputs include an annotated MP4, `density_counts.csv` with one estimate per sampled frame, and `density_trend.png`. Video estimates are per-frame crowd estimates, not unique-person totals and not IN/OUT counts.

To select a different checkpoint or output folder:

```powershell
python scripts/analyze_crowd_density.py --mode image --input datasets/images/crowd.jpg --checkpoint models/csrnet/PartAmodel_best.pth.tar --output-dir outputs/crowd_density/demo --device cpu
```


## Phase 7A: frame-based model evaluation

Compare YOLO and CSRNet on the same ShanghaiTech images using the annotation point count as frame-level ground truth. The report includes MAE, RMSE, signed bias, average inference time, throughput, and a breakdown by crowd-count ranges. The per-image CSV and visual examples make individual errors inspectable. Count-level metrics alone do not prove object-level false positives or false negatives; use a defined point-to-box matching protocol before reporting those categories.

Install the dependencies, then run from the project root:

```powershell
python scripts/evaluate_frame_models.py `
  --dataset-root "D:/path/to/ShanghaiTech" `
  --part part_A `
  --split test `
  --yolo-weights yolo11s.pt `
  --csrnet-weights models/csrnet/PartAmodel_best.pth.tar `
  --device cpu
```

`--dataset-root` must contain `part_A_final/test_data/images` and `part_A_final/test_data/ground_truth`. On a CUDA machine, use `--device 0`. Outputs are written to `outputs/reports/frame_evaluation/`: `frame_metrics.json`, `frame_predictions.csv`, `metrics_by_density.csv`, and a few side-by-side examples with the largest combined count errors. The run uses the full chosen split unless you select a different split explicitly.


### Phase 4: horizontal and vertical counting lines

The line-crossing command supports a horizontal line with `--line-y` or a vertical line with `--line-x`. Downward is IN and upward is OUT for horizontal lines; rightward is IN and leftward is OUT for vertical lines. Use `--start-second` and `--end-second` to process a single continuous segment from a longer montage.

Example for the `linecrosser.mp4` dashcam sample:

```powershell
python scripts/count_line_crossing.py --input datasets/videos/linecrosser.mp4 --line-x 480 --start-second 0 --end-second 8 --output outputs/videos/linecrosser_flow.mp4 --device cpu
```

## Phase 8: zone-based monitoring

Zone monitoring is a separate YOLO + ByteTrack analysis. Configure one or more polygon ROIs and a capacity for each zone in `configs/zones.example.json`. Polygon coordinates are normalized from 0 to 1, so they scale to the input video's resolution. Each tracked person is assigned using the bottom-center (foot point) of the bounding box. A point on a polygon edge belongs to that polygon; when polygons overlap, the first matching zone in the JSON file wins, so one Track ID cannot count in two zones in the same frame. Counts are current occupancy per frame, never accumulated across frames.

Use the interactive ROI tool to select polygon vertices on a representative video frame and set each zone's capacity. From the project root in PyCharm Terminal, run:

```powershell
python scripts/configure_zones.py --input datasets/videos/people.mp4 --at-second 5 --output configs/zones.json
```

In the frame window, left-click polygon vertices in order. Press `U` to undo a point, `N` to finish a polygon (then enter its name and capacity in the terminal), and `S` to save all zones. Repeat `N` for each additional zone. `Q` cancels. Coordinates are saved normalized, so they scale to the video frame size.

Then run monitoring:

```powershell
python scripts/monitor_zones.py `
  --input datasets/videos/people.mp4 `
  --zones configs/zones.json `
  --output outputs/videos/zone_monitoring.mp4 `
  --device cpu
```

The default state thresholds are normal below 80% capacity, near-full from 80% through capacity, and over-capacity above capacity. Use `--near-full-ratio 0.9` to change the near-full threshold to 90%. The command saves an annotated video, per-frame/per-zone occupancy CSV, and an alert CSV that records each transition into over-capacity. `Zone Occupancy` is reported per configured ROI; zones describe only the portion of the scene visible to this camera and do not imply occupancy of unseen space. `people.mp4` is suitable for a basic demo; define real zones using a stable camera view, since moving cameras and scene cuts change where an image-coordinate polygon lies in the physical space.

## Phase 9: Streamlit application

Install dependencies and start the local interface from the project root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard provides colorful cards that open each module directly, while the sidebar keeps a quick page selector. It contains Dashboard, Image Analysis, Video Analysis, Tracking & Line Crossing, Crowd Density, Model Comparison, Zone Monitoring, and Evaluation & Reports. Uploads are processed in temporary folders and removed after each run; results can be downloaded from the page. Model objects are cached between interactions. Choose CPU by default, or GPU 0 when PyTorch detects CUDA.

In Zone Monitoring, upload a representative video frame, click polygon vertices on the preview, set the zone name and capacity, then save the polygon and configuration. Each zone status is normal below 80%, near-full from 80% through capacity, and over-capacity above capacity. The page shows the annotated video, per-zone occupancy chart, alerts, and downloadable CSV files. Keep People Counting (YOLO/ByteTrack) and CSRNet Crowd Density results separate.

Evaluation & Reports now builds a session report from analyses run through the app; it does not load old reports from `outputs/`. Each run records its input name and result metrics. Model Comparison reports MAE/RMSE across images where ground truth was entered, and Line Crossing records IN/OUT errors when manual ground truth is supplied. Download the current session report as JSON or CSV before closing the app; session history is held in Streamlit session state.
