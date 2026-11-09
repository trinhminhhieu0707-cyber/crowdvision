from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW = "CrowdVision | Configure polygon zones"
COLORS = [(0, 220, 255), (255, 170, 0), (255, 80, 180), (80, 230, 80), (200, 100, 255)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click polygon ROIs on a video frame and save their capacities.")
    parser.add_argument("--input", type=Path, required=True, help="Video used to choose a representative camera frame.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "configs" / "zones.json")
    parser.add_argument("--at-second", type=float, default=0.0, help="Choose the frame at this timestamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.at_second < 0:
        raise ValueError("--at-second must be non-negative.")
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.at_second * fps))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read a frame at {args.at_second:.2f} seconds.")
    finally:
        capture.release()

    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 760 / height)
    shown_size = (round(width * scale), round(height * scale))
    points: list[tuple[int, int]] = []
    zones: list[dict[str, object]] = []
    message = "Left click: add point | U: undo | N: finish zone | S: save | Q: cancel"

    def draw(_event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if _event == cv2.EVENT_LBUTTONDOWN:
            px, py = round(x / scale), round(y / scale)
            points.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, *shown_size)
    cv2.setMouseCallback(WINDOW, draw)
    print("Click polygon vertices in order. Press N to finish each zone; S saves the configuration.")
    try:
        while True:
            canvas = frame.copy()
            for index, zone in enumerate(zones):
                poly = np.asarray([(round(x * (width - 1)), round(y * (height - 1))) for x, y in zone["polygon"]], np.int32)
                color = COLORS[index % len(COLORS)]
                cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
                anchor = tuple(poly[0])
                cv2.putText(canvas, f"{zone['name']} cap={zone['capacity']}", anchor,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            if points:
                current = np.asarray(points, np.int32)
                for point in points:
                    cv2.circle(canvas, point, 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.polylines(canvas, [current], False, (0, 255, 0), 2, cv2.LINE_AA)
            display = cv2.resize(canvas, shown_size) if scale != 1 else canvas
            cv2.rectangle(display, (0, 0), (display.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(display, message, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("u") and points:
                points.pop()
            elif key == ord("n"):
                if len(points) < 3:
                    message = "A polygon needs at least 3 points. Add more, or U to undo."
                    continue
                name = input("Zone name: ").strip()
                if not name or any(str(zone["name"]).casefold() == name.casefold() for zone in zones):
                    message = "Name is empty or already used. The current polygon was kept; press N again."
                    continue
                try:
                    capacity = int(input("Maximum capacity: ").strip())
                    if capacity <= 0:
                        raise ValueError
                except ValueError:
                    message = "Capacity must be a positive whole number. The current polygon was kept; press N again."
                    continue
                zones.append({
                    "name": name,
                    "capacity": capacity,
                    "polygon": [[round(x / (width - 1), 6), round(y / (height - 1), 6)] for x, y in points],
                })
                points.clear()
                message = f"Added {name}. Draw next zone, or press S to save."
            elif key == ord("s"):
                if not zones:
                    message = "Draw and finish at least one zone before saving."
                    continue
                if points:
                    message = "Finish or undo the current polygon before saving."
                    continue
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({
                    "coordinate_system": "normalized_0_to_1",
                    "assignment_rule": "Use each tracked person's bottom-center foot point. Polygon edges count as inside. If zones overlap, the first matching zone in this file owns that Track ID for the frame.",
                    "zones": zones,
                }, indent=2), encoding="utf-8")
                print(f"Saved {len(zones)} zones to {args.output}")
                return
            elif key in (ord("q"), 27):
                print("Canceled; no configuration saved.")
                return
    finally:
        cv2.destroyWindow(WINDOW)


if __name__ == "__main__":
    main()
