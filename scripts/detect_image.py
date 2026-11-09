from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import PersonDetector, draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and count people in one image.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an input image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "images" / "people_detected.jpg",
        help="Path for the annotated image.",
    )
    parser.add_argument("--weights", default="yolo11s.pt", help="YOLO checkpoint or official model name.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--image-size", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="Inference device, for example 'cpu' or '0'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.input}")

    image = cv2.imread(str(args.input))
    if image is None:
        raise ValueError(f"OpenCV could not decode the input image: {args.input}")

    detector = PersonDetector(
        weights=args.weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.image_size,
        device=args.device,
    )
    detections = detector.detect(image)
    annotated = draw_detections(image, detections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise OSError(f"Could not write output image: {args.output}")

    logging.info("Detected %d people. Saved result to %s", len(detections), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
