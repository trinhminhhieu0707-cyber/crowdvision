from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.tracking.tracker import TrackedPerson, Tracker, draw_tracks
from src.video_io import open_video, open_writer


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]
    capacity: int


@dataclass(frozen=True)
class ZoneVideoSummary:
    input_path: Path
    output_path: Path
    csv_path: Path
    alerts_path: Path
    processed_frames: int
    source_fps: float
    average_processing_fps: float
    final_occupancy: dict[str, int]


def load_zones(config_path: str | Path) -> list[Zone]:
    """Load normalized (0..1) polygon coordinates and capacities from JSON."""
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    entries = config.get("zones") if isinstance(config, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Zone config must contain a non-empty 'zones' list.")
    zones: list[Zone] = []
    names: set[str] = set()
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        points = entry.get("polygon")
        capacity = entry.get("capacity")
        if not name or name in names:
            raise ValueError("Every zone needs a unique, non-empty name.")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Zone {name!r} needs at least 3 polygon points.")
        polygon = tuple((float(point[0]), float(point[1])) for point in points)
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in polygon):
            raise ValueError(f"Zone {name!r} coordinates must be normalized between 0 and 1.")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError(f"Zone {name!r} capacity must be a positive integer.")
        contour = np.asarray(polygon, np.float32)
        if abs(cv2.contourArea(contour)) < 1e-6:
            raise ValueError(f"Zone {name!r} polygon must enclose an area.")
        zones.append(Zone(name=name, polygon=polygon, capacity=capacity))
        names.add(name)
    return zones


def assign_tracks(
    tracks: list[TrackedPerson], zones: list[Zone], width: int, height: int
) -> dict[str, set[int]]:
    """Assign each track by footpoint; overlapping/boundary matches go to first config zone."""
    contours = [
        np.asarray([(round(x * (width - 1)), round(y * (height - 1))) for x, y in z.polygon], np.int32)
        for z in zones
    ]
    occupants = {zone.name: set() for zone in zones}
    for track in tracks:
        foot = (float((track.x1 + track.x2) / 2), float(track.y2))
        for zone, contour in zip(zones, contours, strict=True):
            if cv2.pointPolygonTest(contour, foot, False) >= 0:
                occupants[zone.name].add(track.track_id)
                break
    return occupants


_assign_tracks = assign_tracks  # alias tương thích ngược với tên cũ



def process_zone_video(
    tracker: Tracker,
    input_path: str | Path,
    output_path: str | Path,
    zones: list[Zone],
    near_full_ratio: float = 0.8,

    alert_min_seconds: float = 1.0,
    alert_clear_seconds: float = 1.0,
) -> ZoneVideoSummary:
    """Track people in a video, assign each current Track ID to at most one zone, and save outputs."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not zones:
        raise ValueError("At least one zone is required.")
    if not 0 < near_full_ratio <= 1:
        raise ValueError("near_full_ratio must be greater than 0 and at most 1.")
    ## [SỬA 3b | zone_monitor.py | process_zone_video] Validate 2 tham số mới.
    if alert_min_seconds < 0 or alert_clear_seconds < 0:
        raise ValueError("alert_min_seconds and alert_clear_seconds must be >= 0.")

    csv_path = output_path.with_name(f"{output_path.stem}_occupancy.csv")
    alerts_path = output_path.with_name(f"{output_path.stem}_alerts.csv")
    occupancy_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []

    with open_video(input_path) as (capture, info), open_writer(
        output_path, info.fps, (info.width, info.height)
    ) as writer:
        width, height, fps = info.width, info.height, info.fps
        ## [SỬA 2d | zone_monitor.py | process_zone_video] Reset tracker trước mỗi video mới (xem SỬA 2a).
        tracker.reset()

        final_occupancy = {zone.name: 0 for zone in zones}

        alert_min_frames = max(1, round(alert_min_seconds * fps))
        alert_clear_frames = max(1, round(alert_clear_seconds * fps))
        over_streak = {zone.name: 0 for zone in zones}
        clear_streak = {zone.name: 0 for zone in zones}
        alerting = {zone.name: False for zone in zones}
        processed = 0
        elapsed_total = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = perf_counter()
            tracks = tracker.track(frame)
            occupants = assign_tracks(tracks, zones, width, height)
            timestamp = processed / fps
            for zone in zones:
                count = len(occupants[zone.name])
                rate = count / zone.capacity * 100.0
                status = "over_capacity" if count > zone.capacity else (
                    "near_full" if rate >= near_full_ratio * 100 else "normal"
                )
                final_occupancy[zone.name] = count
                occupancy_rows.append({
                    "frame": processed, "timestamp_seconds": round(timestamp, 3),
                    "zone": zone.name, "occupancy": count, "capacity": zone.capacity,
                    "occupancy_rate_percent": round(rate, 2), "status": status,
                })

                if status == "over_capacity":
                    over_streak[zone.name] += 1
                    clear_streak[zone.name] = 0
                    if not alerting[zone.name] and over_streak[zone.name] >= alert_min_frames:
                        alerting[zone.name] = True
                        alert_rows.append({
                            "frame": processed, "timestamp_seconds": round(timestamp, 3),
                            "first_exceeded_frame": processed - over_streak[zone.name] + 1,
                            "zone": zone.name, "occupancy": count, "capacity": zone.capacity,
                            "message": f"{zone.name} exceeded capacity ({count}/{zone.capacity})",
                        })
                else:
                    over_streak[zone.name] = 0
                    if alerting[zone.name]:
                        clear_streak[zone.name] += 1
                        if clear_streak[zone.name] >= alert_clear_frames:
                            alerting[zone.name] = False
                            clear_streak[zone.name] = 0

            annotated = draw_tracks(frame, tracks)
            _draw_zones(annotated, zones, occupants, near_full_ratio)
            elapsed_total += perf_counter() - started
            writer.write(annotated)
            processed += 1

    _write_csv(csv_path, occupancy_rows, ["frame", "timestamp_seconds", "zone", "occupancy", "capacity", "occupancy_rate_percent", "status"])

    _write_csv(alerts_path, alert_rows, ["frame", "timestamp_seconds", "first_exceeded_frame", "zone", "occupancy", "capacity", "message"])
    return ZoneVideoSummary(
        input_path, output_path, csv_path, alerts_path, processed, float(fps),
        processed / elapsed_total if elapsed_total else 0.0, final_occupancy,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _draw_zones(frame: np.ndarray, zones: list[Zone], occupants: dict[str, set[int]], near_full_ratio: float) -> None:
    height, width = frame.shape[:2]
    colors = [(0, 220, 255), (255, 170, 0), (255, 80, 180), (80, 230, 80), (200, 100, 255)]
    for index, zone in enumerate(zones):
        color = colors[index % len(colors)]
        polygon = np.asarray([(round(x * (width - 1)), round(y * (height - 1))) for x, y in zone.polygon], np.int32)
        count = len(occupants[zone.name])
        rate = count / zone.capacity
        status = "OVER" if count > zone.capacity else "NEAR" if rate >= near_full_ratio else "OK"
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        label = f"{zone.name}: {count}/{zone.capacity} ({rate * 100:.0f}%) {status}"
        label_y = max(24, y - 8)
        cv2.rectangle(frame, (x, max(0, label_y - 23)), (min(width - 1, x + max(220, len(label) * 9)), label_y + 4), (20, 20, 20), -1)
        cv2.putText(frame, label, (x + 4, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
