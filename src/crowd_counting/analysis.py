from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .csrnet import CrowdPrediction


@dataclass(frozen=True)
class DenseRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    estimated_count: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def density_heatmap(density_map: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Colorize a CSRNet output map to the input image dimensions (BGR)."""
    height, width = image_shape
    values = np.maximum(np.nan_to_num(density_map, nan=0.0), 0.0).astype(np.float32)
    upscaled = cv2.resize(values, (width, height), interpolation=cv2.INTER_CUBIC)
    positive = upscaled[upscaled > 0]
    scale = float(np.percentile(positive, 99)) if positive.size else 0.0
    normalized = np.clip(upscaled / max(scale, 1e-8), 0, 1)
    return cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_JET)


def overlay_density(image_bgr: np.ndarray, density_map: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    heat = density_heatmap(density_map, image_bgr.shape[:2])
    positive = np.maximum(np.nan_to_num(density_map), 0)
    mask_small = positive > 0
    mask = cv2.resize(mask_small.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)
    blended[mask == 0] = image_bgr[mask == 0]
    return blended


def find_dense_regions(
    density_map: np.ndarray,
    image_shape: tuple[int, int],
    grid_rows: int = 4,
    grid_cols: int = 4,
    top_k: int = 3,
) -> list[DenseRegion]:
    """Rank non-overlapping coarse grid zones by estimated people count."""
    if grid_rows < 1 or grid_cols < 1 or top_k < 1:
        raise ValueError("Grid dimensions and top_k must be positive.")
    h, w = density_map.shape[:2]
    image_h, image_w = image_shape
    positive = np.maximum(np.nan_to_num(density_map, nan=0.0), 0.0)
    cells: list[DenseRegion] = []
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    for row in range(grid_rows):
        for col in range(grid_cols):
            ya, yb = y_edges[row], y_edges[row + 1]
            xa, xb = x_edges[col], x_edges[col + 1]
            if ya == yb or xa == xb:
                continue
            score = float(positive[ya:yb, xa:xb].sum())
            cells.append(DenseRegion(
                x1=int(round(xa * image_w / w)),
                y1=int(round(ya * image_h / h)),
                x2=int(round(xb * image_w / w)),
                y2=int(round(yb * image_h / h)),
                estimated_count=score,
            ))
    return sorted(cells, key=lambda region: region.estimated_count, reverse=True)[:top_k]


def draw_dense_regions(image_bgr: np.ndarray, regions: list[DenseRegion]) -> np.ndarray:
    """Draw readable coarse region boxes with short zone IDs."""
    result = image_bgr.copy()
    height, width = result.shape[:2]
    colors = [(0, 230, 255), (0, 165, 255), (255, 190, 0)]
    for rank, region in enumerate(regions, start=1):
        x1 = int(np.clip(region.x1, 0, width - 1))
        y1 = int(np.clip(region.y1, 0, height - 1))
        x2 = int(np.clip(region.x2, 0, width - 1))
        y2 = int(np.clip(region.y2, 0, height - 1))
        color = colors[(rank - 1) % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Keep the in-image tag short so adjacent zone labels cannot run together.
        label = f"Z{rank}"
        font_scale = max(0.42, min(0.62, width / 700))
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tag_w, tag_h = text_w + 10, text_h + baseline + 6
        tag_x = min(x1 + 2, max(0, width - tag_w))
        tag_y = min(y1 + 2, max(0, height - tag_h))
        cv2.rectangle(result, (tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h), color, -1)
        cv2.putText(result, label, (tag_x + 5, tag_y + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
    return result


def annotate_density_frame(
    image_bgr: np.ndarray,
    prediction: CrowdPrediction,
    regions: list[DenseRegion] | None = None,
    sampled_at_seconds: float | None = None,
) -> np.ndarray:
    frame = overlay_density(image_bgr, prediction.density_map)
    if regions:
        frame = draw_dense_regions(frame, regions)
    text = f"CSRNet estimated count: {prediction.count:.1f}"
    if sampled_at_seconds is not None:
        text += f" | updated {sampled_at_seconds:.1f}s"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 570), 38), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
