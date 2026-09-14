"""Small positional-embedding helpers used by the OCR model."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _interpolate_2d_pos(position: Tensor, height: int, width: int) -> Tensor:
    old_h = old_w = int(position.size(1) ** 0.5)
    if old_h * old_w != position.size(1):
        raise ValueError("spatial position table must contain a square grid")
    pos = position.transpose(1, 2).reshape(1, position.size(2), old_h, old_w)
    pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
    return pos.flatten(2).transpose(1, 2)

