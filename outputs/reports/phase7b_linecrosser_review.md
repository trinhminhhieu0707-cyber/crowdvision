# Phase 7B visual review — `linecrosser.mp4`

## Scope

- Source: `datasets/videos/linecrosser.mp4` (39.6 s, 990 frames, 25 FPS).
- Annotated output: `outputs/videos/linecrosser_flow_full.mp4` (39.6 s, 990 frames).
- Configuration: YOLO11s + ByteTrack, vertical screen line `x=480`, IN=rightward, OUT=leftward.
- Displayed final crossing-event totals: IN=7, OUT=2, TOTAL=9.
- This is a visual screening of synchronized source/output frames sampled every 2 seconds, plus inspection of the complete clip's timeline. It is not a frame-by-frame ground-truth annotation.

## Observations by requested error type

| Case | Screening result | Evidence / limitation |
|---|---|---|
| False Positive | Not confirmed | Sampled boxes mostly cover visible pedestrians or people riding motorcycles. A box on a rider is a valid `person` detection. Need frame-level manual labels to establish a false box. |
| False Negative | Possible, not confirmed | People in the distance and along busy sidewalks are very small; sampled frames do not provide enough resolution to decide which visible people were missed. |
| Missed Detection | Possible, not countable | This is the specific event behind a false negative. No exhaustive person-by-person reference annotation exists for this clip. |
| Duplicate Counting | Not confirmed | No clear duplicate box is visible in sampled frames. The 9 total is crossing events, not unique-person count; repeated crossing by one Track ID would be counted again by design. |
| ID Switching | Inconclusive | Track IDs change frequently across the montage, but cuts and changing camera views prevent distinguishing an ID switch from a new person/scene. Need continuous same-person review. |
| Tracking Failure | Scene discontinuities observed; not attributable as a rate | The video contains multiple abrupt scene/camera changes. A continuous trajectory cannot be validated across those edits. |
| Occlusion | Mild occlusion visible | Pedestrians are partly obscured by vehicles, umbrellas, or other people in some frames. No dense, sustained occlusion case is clear in the sampled review. |
| Person too small | Present as a difficult condition | Distant pedestrians/riders occupy few pixels, especially in the road depth. This is a plausible source of missed detections, but not a confirmed miss without labels. |
| Crowd too dense | Not represented adequately | The clip shows street traffic and scattered groups, not a densely packed crowd. This case cannot be evaluated with this video. |

## Important interpretation

The fixed line at `x=480` is anchored to image coordinates while the dashcam moves and the clip changes scenes. The result therefore counts detections crossing the middle of the displayed frame; it does not measure entries/exits through one stable doorway or venue boundary. The 7 IN / 2 OUT values are useful as a pipeline demonstration, not as validated real-world footfall metrics.

## What is needed for quantitative Phase 7B metrics

Use one continuous, fixed-camera segment. Manually annotate people and their crossings/identities on selected frames, then compare the detections and tracks with those labels. Until that reference exists, FP/FN, duplicate-count, ID-switch, and tracking-failure totals remain unverified; this review does not invent those counts.
