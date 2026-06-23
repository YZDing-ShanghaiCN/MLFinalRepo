from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.transforms import ResizeLongestSideAndPad


def _image(width: int, height: int, color: tuple[int, int, int] = (220, 40, 10)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


@pytest.mark.parametrize("width,height", [(83, 81), (328, 282), (60, 49), (90, 195), (128, 128), (20, 200)])
def test_resize_pad_outputs_expected_shape(width: int, height: int) -> None:
    transform = ResizeLongestSideAndPad(size=128)
    tensor = transform(_image(width, height))
    assert tuple(tensor.shape) == (3, 128, 128)


def test_square_input_fills_canvas() -> None:
    transform = ResizeLongestSideAndPad(size=128)
    result = transform.apply(_image(64, 64))
    assert result.resized_size == (128, 128)
    assert result.padding == (0, 0, 0, 0)


@pytest.mark.parametrize("width,height", [(83, 81), (328, 282), (60, 49), (90, 195), (10, 200)])
def test_aspect_ratio_is_preserved(width: int, height: int) -> None:
    transform = ResizeLongestSideAndPad(size=128)
    result = transform.apply(_image(width, height))
    scale = 128 / max(width, height)
    assert max(result.resized_size) == 128
    assert result.resized_size[0] == pytest.approx(width * scale, abs=0.5)
    assert result.resized_size[1] == pytest.approx(height * scale, abs=0.5)


@pytest.mark.parametrize("width,height", [(83, 81), (60, 49), (90, 195)])
def test_padding_is_centered(width: int, height: int) -> None:
    transform = ResizeLongestSideAndPad(size=128)
    result = transform.apply(_image(width, height))
    left, top, right, bottom = result.padding
    assert abs(left - right) <= 1
    assert abs(top - bottom) <= 1


def test_tensor_dtype_range_and_channels() -> None:
    transform = ResizeLongestSideAndPad(size=128)
    tensor = transform(_image(83, 81))
    assert tensor.dtype == torch.float32
    assert tensor.shape[0] == 3
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
