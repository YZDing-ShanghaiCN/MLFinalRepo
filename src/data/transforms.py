from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image


_INTERPOLATION = {
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "bicubic": Image.Resampling.BICUBIC,
}


@dataclass(frozen=True)
class ResizePadResult:
    """PIL result and metadata for resize-longest-side plus center padding."""

    image: Image.Image
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int]


class ResizeLongestSideAndPad:
    """Resize longest side to a fixed length, center-pad, and return CHW tensor."""

    def __init__(
        self,
        size: int = 128,
        padding_value: Sequence[int] = (0, 0, 0),
        interpolation: str = "bilinear",
    ) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if len(padding_value) != 3:
            raise ValueError("padding_value must contain three RGB values")
        if interpolation not in _INTERPOLATION:
            raise ValueError(f"Unsupported interpolation: {interpolation}")
        self.size = int(size)
        self.padding_value = tuple(int(value) for value in padding_value)
        self.interpolation = interpolation
        self._pil_interpolation = _INTERPOLATION[interpolation]

    def resized_size(self, width: int, height: int) -> tuple[int, int]:
        """Compute resized (width, height) with the longest side equal to size."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: width={width}, height={height}")
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if max(new_width, new_height) != self.size:
            if width >= height:
                new_width = self.size
            else:
                new_height = self.size
        new_width = min(self.size, new_width)
        new_height = min(self.size, new_height)
        return new_width, new_height

    def apply(self, image: Image.Image) -> ResizePadResult:
        """Return a padded PIL image and the geometry used to create it."""
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        new_width, new_height = self.resized_size(width, height)
        resized = rgb_image.resize((new_width, new_height), self._pil_interpolation)

        pad_left = (self.size - new_width) // 2
        pad_top = (self.size - new_height) // 2
        pad_right = self.size - new_width - pad_left
        pad_bottom = self.size - new_height - pad_top

        canvas = Image.new("RGB", (self.size, self.size), self.padding_value)
        canvas.paste(resized, (pad_left, pad_top))
        return ResizePadResult(
            image=canvas,
            original_size=(width, height),
            resized_size=(new_width, new_height),
            padding=(pad_left, pad_top, pad_right, pad_bottom),
        )

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert an RGB PIL image to float32 CHW tensor in [0, 1]."""
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        data = torch.frombuffer(bytearray(rgb_image.tobytes()), dtype=torch.uint8)
        tensor = data.view(height, width, 3).permute(2, 0, 1).contiguous()
        return tensor.to(dtype=torch.float32).div_(255.0)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Transform a PIL image into a 3 x size x size tensor."""
        result = self.apply(image)
        return self.to_tensor(result.image)


def build_transform_from_config(config: dict) -> ResizeLongestSideAndPad:
    """Build the deterministic preprocessing transform from dataset config."""
    preprocess = config.get("preprocess", {})
    return ResizeLongestSideAndPad(
        size=int(preprocess.get("longest_side", config.get("dataset", {}).get("input_size", 128))),
        padding_value=preprocess.get("padding_value", (0, 0, 0)),
        interpolation=str(preprocess.get("interpolation", "bilinear")),
    )
