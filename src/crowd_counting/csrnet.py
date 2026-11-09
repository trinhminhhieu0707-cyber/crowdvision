from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch
from PIL import Image
from torch import nn


def _layers(config: list[int | str], in_channels: int = 3, dilation: bool = False) -> nn.Sequential:
    modules: list[nn.Module] = []
    rate = 2 if dilation else 1
    for value in config:
        if value == "M":
            modules.append(nn.MaxPool2d(2, stride=2))
        else:
            channels = int(value)
            modules.extend([nn.Conv2d(in_channels, channels, 3, padding=rate, dilation=rate), nn.ReLU(inplace=True)])
            in_channels = channels
    return nn.Sequential(*modules)


class CSRNet(nn.Module):
    """CSRNet-B with VGG-style frontend and dilated backend."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend = _layers([64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512])
        self.backend = _layers([512, 512, 512, 256, 128, 64], 512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backend(self.frontend(x)))


@dataclass(frozen=True)
class CrowdPrediction:
    count: float
    density_map: np.ndarray
    image_rgb: np.ndarray


class CSRNetCounter:
    """Single-image CSRNet inference. Independent from YOLO and tracking."""

    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CSRNet checkpoint does not exist: {checkpoint}")
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.isdigit():
            requested_device = f"cuda:{requested_device}"
        self.device = torch.device(requested_device)
        self.model = CSRNet().to(self.device)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was added.
            state = torch.load(checkpoint, map_location="cpu")
        except pickle.UnpicklingError:
            # Some trusted research .pth.tar files bundle metadata with weights.
            # Only load checkpoints from a source you trust.
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Checkpoint must contain a state_dict.")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_rgb(self, image_rgb: np.ndarray) -> CrowdPrediction:
        """Run inference on an RGB uint8 image array."""
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.size == 0:
            raise ValueError("image_rgb must be a non-empty HxWx3 RGB array.")
        rgb = np.asarray(image_rgb, dtype=np.uint8)
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        x = ((x - mean) / std).unsqueeze(0).to(self.device)
        density = self.model(x).squeeze(0).squeeze(0).float().cpu().numpy()
        return CrowdPrediction(float(density.sum()), density, rgb.copy())

    def predict(self, image_path: str | Path) -> CrowdPrediction:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return self.predict_rgb(rgb)
