from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.counting.zone_monitor import load_zones, process_zone_video
from src.tracking import PersonTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor occupancy in configured polygon zones.")
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--zones", type=Path, required=True, help="JSON file with normalized ROI polygons and capacities.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "videos" / "zone_monitoring.mp4")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--near-full-ratio", type=float, default=0.8, help="Near-full threshold, e.g. 0.8 means 80%% of capacity.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zones = load_zones(args.zones)
    tracker = PersonTracker(
        weights=args.weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.image_size,
        device=args.device,
    )
    summary = process_zone_video(
        tracker, args.input, args.output, zones, near_full_ratio=args.near_full_ratio
    )
    logging.info(
        "Processed %d frames (%.1f FPS). Final zone occupancy: %s. Video: %s | CSV: %s | alerts: %s",
        summary.processed_frames, summary.average_processing_fps, summary.final_occupancy,
        summary.output_path, summary.csv_path, summary.alerts_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
