from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.tracking.tracker import PersonTracker, TrackedPerson, draw_tracks


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


def _assign_tracks(
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


def process_zone_video(
    tracker: PersonTracker,
    input_path: str | Path,
    output_path: str | Path,
    zones: list[Zone],
    near_full_ratio: float = 0.8,
) -> ZoneVideoSummary:
    """Track people in a video, assign each current Track ID to at most one zone, and save outputs."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if not zones:
        raise ValueError("At least one zone is required.")
    if not 0 < near_full_ratio <= 1:
        raise ValueError("near_full_ratio must be greater than 0 and at most 1.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open the video: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_name(f"{output_path.stem}_occupancy.csv")
    alerts_path = output_path.with_name(f"{output_path.stem}_alerts.csv")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid dimensions: {input_path}")
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Could not create output video: {output_path}")

        occupancy_rows: list[dict[str, Any]] = []
        alert_rows: list[dict[str, Any]] = []
        previous_status = {zone.name: "normal" for zone in zones}
        final_occupancy = {zone.name: 0 for zone in zones}
        processed = 0
        elapsed_total = 0.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                started = perf_counter()
                tracks = tracker.track(frame)
                occupants = _assign_tracks(tracks, zones, width, height)
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
                    if status == "over_capacity" and previous_status[zone.name] != "over_capacity":
                        alert_rows.append({
                            "frame": processed, "timestamp_seconds": round(timestamp, 3),
                            "zone": zone.name, "occupancy": count, "capacity": zone.capacity,
                            "message": f"{zone.name} exceeded capacity ({count}/{zone.capacity})",
                        })
                    previous_status[zone.name] = status

                annotated = draw_tracks(frame, tracks)
                _draw_zones(annotated, zones, occupants, near_full_ratio)
                elapsed_total += perf_counter() - started
                writer.write(annotated)
                processed += 1
        finally:
            writer.release()
    finally:
        capture.release()

    _write_csv(csv_path, occupancy_rows, ["frame", "timestamp_seconds", "zone", "occupancy", "capacity", "occupancy_rate_percent", "status"])
    _write_csv(alerts_path, alert_rows, ["frame", "timestamp_seconds", "zone", "occupancy", "capacity", "message"])
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
