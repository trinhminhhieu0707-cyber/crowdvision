from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import PersonDetector, process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and count people in every frame of one video.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an input video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "videos" / "people_detected.mp4",
        help="Path for the annotated MP4 video.",
    )
    parser.add_argument("--weights", default="yolo11s.pt", help="YOLO checkpoint or official model name.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--image-size", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="Inference device, for example 'cpu' or '0'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = PersonDetector(
        weights=args.weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.image_size,
        device=args.device,
    )
    summary = process_video(detector, args.input, args.output)
    logging.info(
        "Processed %d frames at %.1f FPS on average. Saved result to %s",
        summary.processed_frames,
        summary.average_processing_fps,
        summary.output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
