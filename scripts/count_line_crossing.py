from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.counting.flow_video import process_flow_video
from src.tracking import PersonTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count people crossing a horizontal line in one video.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an input video.")
    line_group = parser.add_mutually_exclusive_group(required=True)
    line_group.add_argument("--line-y", type=int, help="Horizontal line position in pixels from the top.")
    line_group.add_argument("--line-x", type=int, help="Vertical line position in pixels from the left.")
    parser.add_argument("--margin", type=int, default=12, help="Stable zone around the line in pixels.")
    parser.add_argument("--start-second", type=float, default=0.0, help="Start time for processing a continuous clip segment.")
    parser.add_argument("--end-second", type=float, default=None, help="End time; omit to process through the video end.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "videos" / "people_flow.mp4")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = PersonTracker(
        weights=args.weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.image_size,
        device=args.device,
    )
    summary = process_flow_video(
        tracker,
        args.input,
        args.output,
        line_y=args.line_y,
        crossing_margin=args.margin,
        line_x=args.line_x,
        start_second=args.start_second,
        end_second=args.end_second,
    )
    logging.info(
        "Processed %d frames. IN=%d OUT=%d TOTAL=%d. Saved result to %s",
        summary.processed_frames,
        summary.counts.entered,
        summary.counts.exited,
        summary.counts.total,
        summary.output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
