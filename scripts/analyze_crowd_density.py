from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crowd_counting import CSRNetCounter
from src.crowd_counting.analysis import draw_dense_regions, find_dense_regions, overlay_density
from src.crowd_counting.video import analyze_video_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze crowd density with CSRNet, independently of YOLO tracking.")
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/csrnet/PartAmodel_best.pth.tar")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or a CUDA device such as 0; defaults to available device.")
    parser.add_argument("--sample-every", type=int, default=15, help="Analyze every Nth video frame; use 1 to analyze all frames.")
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--top-regions", type=int, default=3)
    return parser.parse_args()


def analyze_image(args: argparse.Namespace, counter: CSRNetCounter, output_dir: Path) -> None:
    bgr = cv2.imread(str(args.input))
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {args.input}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prediction = counter.predict_rgb(rgb)
    regions = find_dense_regions(prediction.density_map, bgr.shape[:2], args.grid_rows, args.grid_cols, args.top_regions)
    overlay = overlay_density(bgr, prediction.density_map)
    regional = draw_dense_regions(overlay, regions)
    heatmap = cv2.resize(
        cv2.applyColorMap(
            np.uint8(np.clip(np.maximum(prediction.density_map, 0) / max(float(np.max(prediction.density_map)), 1e-8), 0, 1) * 255),
            cv2.COLORMAP_JET,
        ),
        (bgr.shape[1], bgr.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overlay": output_dir / "density_overlay.jpg",
        "regions": output_dir / "dense_regions.jpg",
        "heatmap": output_dir / "density_heatmap.jpg",
    }
    for key, image in [("overlay", overlay), ("regions", regional), ("heatmap", heatmap)]:
        if not cv2.imwrite(str(paths[key]), image):
            raise OSError(f"Could not save {paths[key]}")
    result = {
        "mode": "image",
        "input": str(args.input),
        "estimated_count": prediction.count,
        "device": str(counter.device),
        "checkpoint": str(args.checkpoint),
        "dense_regions": [region.to_dict() for region in regions],
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    result_path = output_dir / "density_analysis.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input does not exist: {args.input}")
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "crowd_density" / args.input.stem
    counter = CSRNetCounter(args.checkpoint, device=args.device)
    if args.mode == "image":
        analyze_image(args, counter, output_dir)
    else:
        result = analyze_video_density(
            args.input,
            output_dir,
            counter,
            sample_every=args.sample_every,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            top_regions=args.top_regions,
        )
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
