from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crowd_counting import CSRNetCounter
from src.crowd_counting.analysis import overlay_density
from src.detection import PersonDetector, draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLO frame counts with CSRNet on ShanghaiTech annotated images."
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Directory containing part_A_final and part_B_final.")
    parser.add_argument("--part", choices=["part_A", "part_B"], default="part_A")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--yolo-weights", type=Path, default=PROJECT_ROOT / "yolo11s.pt")
    parser.add_argument("--csrnet-weights", type=Path,
                        default=PROJECT_ROOT / "models/csrnet/PartAmodel_best.pth.tar")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/reports/frame_evaluation")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, or 0. Defaults to each model's available device.")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--visual-examples", type=int, default=6,
                        help="Save comparison images for the largest combined errors; set 0 to skip.")
    return parser.parse_args()


def ground_truth_count(mat_path: Path) -> int:
    data = sio.loadmat(mat_path)
    try:
        points = data["image_info"][0, 0]["location"][0, 0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unrecognized ShanghaiTech annotation format: {mat_path}") from exc
    return int(np.asarray(points).reshape(-1, 2).shape[0])


def metric_summary(rows: list[dict[str, Any]], prediction_key: str, time_key: str) -> dict[str, float]:
    truth = np.asarray([row["ground_truth"] for row in rows], dtype=np.float64)
    pred = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    errors = pred - truth
    mean_ms = float(np.mean([row[time_key] for row in rows]))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "mean_inference_ms_per_image": mean_ms,
        "throughput_fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def density_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [(0, 200, "<200"), (200, 500, "200-499"), (500, 1000, "500-999"), (1000, float("inf"), ">=1000")]
    result = []
    for lower, upper, label in bins:
        group = [r for r in rows if lower <= r["ground_truth"] < upper]
        if not group:
            continue
        counts = np.asarray([r["ground_truth"] for r in group], dtype=np.float64)
        result_row: dict[str, Any] = {
            "density_group_people": label,
            "images": len(group),
            "mean_ground_truth": float(counts.mean()),
        }
        for name, prediction_key in (("yolo", "yolo_count"), ("csrnet", "csrnet_count")):
            errors = np.asarray([r[prediction_key] - r["ground_truth"] for r in group], dtype=np.float64)
            result_row[f"{name}_MAE"] = float(np.abs(errors).mean())
            result_row[f"{name}_RMSE"] = float(np.sqrt(np.square(errors).mean()))
            result_row[f"{name}_bias"] = float(errors.mean())
        result.append(result_row)
    return result


def save_visual_examples(
    rows: list[dict[str, Any]], image_dir: Path, output_dir: Path,
    detector: PersonDetector, csrnet: CSRNetCounter, limit: int,
) -> None:
    if limit <= 0:
        return
    examples_dir = output_dir / "comparison_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    chosen = sorted(
        rows,
        key=lambda row: row["yolo_abs_error"] + row["csrnet_abs_error"],
        reverse=True,
    )[:limit]
    for row in chosen:
        path = image_dir / row["image"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        detections = detector.detect(bgr)
        yolo_view = draw_detections(bgr, detections)
        csr_prediction = csrnet.predict_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        csr_view = overlay_density(bgr, csr_prediction.density_map)
        for view, label in (
            (yolo_view, f"YOLO: {len(detections)} | GT: {row['ground_truth']}"),
            (csr_view, f"CSRNet: {csr_prediction.count:.1f} | GT: {row['ground_truth']}"),
        ):
            cv2.rectangle(view, (0, 0), (min(view.shape[1] - 1, 450), 38), (20, 20, 20), -1)
            cv2.putText(view, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
        comparison = np.concatenate([yolo_view, csr_view], axis=1)
        destination = examples_dir / f"{Path(row['image']).stem}_comparison.jpg"
        if not cv2.imwrite(str(destination), comparison):
            raise OSError(f"Could not save visual comparison: {destination}")


def main() -> None:
    args = parse_args()
    base = args.dataset_root / f"{args.part}_final" / f"{args.split}_data"
    image_dir, annotation_dir = base / "images", base / "ground_truth"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected image and ground-truth folders under {base}")
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")
    missing = [p.name for p in image_paths if not (annotation_dir / f"GT_{p.stem}.mat").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing annotation for {len(missing)} images; first: {missing[0]}")
    if not args.yolo_weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.csrnet_weights.is_file():
        raise FileNotFoundError(f"CSRNet weights not found: {args.csrnet_weights}")

    detector = PersonDetector(
        weights=args.yolo_weights,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.image_size,
        device=args.device,
    )
    csrnet = CSRNetCounter(args.csrnet_weights, device=args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warm up both networks so first-use setup does not distort timing averages.
    first_bgr = cv2.imread(str(image_paths[0]))
    if first_bgr is None:
        raise ValueError(f"Could not decode image: {image_paths[0]}")
    detector.detect(first_bgr)
    csrnet.predict_rgb(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB))

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")
        truth = ground_truth_count(annotation_dir / f"GT_{image_path.stem}.mat")

        start = time.perf_counter()
        detections = detector.detect(image)
        yolo_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        csr_prediction = csrnet.predict_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        csrnet_ms = (time.perf_counter() - start) * 1000

        yolo_error = len(detections) - truth
        csrnet_error = csr_prediction.count - truth
        rows.append({
            "image": image_path.name,
            "ground_truth": truth,
            "yolo_count": len(detections),
            "yolo_error": yolo_error,
            "yolo_abs_error": abs(yolo_error),
            "yolo_inference_ms": yolo_ms,
            "csrnet_count": csr_prediction.count,
            "csrnet_error": csrnet_error,
            "csrnet_abs_error": abs(csrnet_error),
            "csrnet_inference_ms": csrnet_ms,
        })
        if index % 10 == 0 or index == len(image_paths):
            logging.info("Evaluated %d/%d images", index, len(image_paths))

    csv_path = output_dir / "frame_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups = density_groups(rows)
    group_path = output_dir / "metrics_by_density.csv"
    with group_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0].keys()))
        writer.writeheader()
        writer.writerows(groups)

    metrics = {
        "dataset": "ShanghaiTech " + args.part + " " + args.split,
        "images": len(rows),
        "ground_truth": "number of head-point annotations in GT_*.mat",
        "yolo_weights": str(args.yolo_weights),
        "csrnet_weights": str(args.csrnet_weights),
        "yolo": metric_summary(rows, "yolo_count", "yolo_inference_ms"),
        "csrnet": metric_summary(rows, "csrnet_count", "csrnet_inference_ms"),
        "density_groups": groups,
        "notes": [
            "Both models were evaluated on the same image files and point-count ground truth.",
            "YOLO count is the number of person detections after NMS; CSRNet count is the sum of its density output.",
            "Inference time excludes image file reading and model initialization; it includes model preprocessing and prediction.",
            "Count metrics do not identify object-level false positives or false negatives without a declared point-to-box matching protocol.",
        ],
    }
    metrics_path = output_dir / "frame_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_visual_examples(rows, image_dir, output_dir, detector, csrnet, args.visual_examples)

    print(json.dumps(metrics, indent=2))
    logging.info("Saved per-image predictions: %s", csv_path)
    logging.info("Saved density breakdown: %s", group_path)
    logging.info("Saved summary metrics: %s", metrics_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
