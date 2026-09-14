"""Inference-only batch-1 MPS runtime for MonkeyOCRv2-B + R2 Q129.

The training model keeps its portable PyTorch implementation.  This wrapper
reuses exactly the same parameters while removing batch padding/masks and
using small Metal kernels for operations that are expensive to express as a
sequence of MPS graph operations.  It accepts one native dynamic-patch image
and therefore never waits for, pads to, or groups with another image.

The encoder kernels are adapted from the independently validated
``PP-OCR Fusion/tools/monkeyocrv2_mps.py`` implementation.  Construct this
wrapper only after strict checkpoint loading and after moving the base model
to MPS.  It is forward-only and must never be used for training.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import weakref

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .position import _interpolate_2d_pos
from .model import MonkeyQ129OCR, packed_to_raster


_METAL = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef DTYPE scalar;

kernel void rms_norm(
    device const scalar* x [[buffer(0)]],
    device const scalar* weight [[buffer(1)]],
    device scalar* y [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  float sum = 0.0f;
  for (uint c = tid; c < width; c += 256) {
    float val = float(x[row * width + c]);
    sum += val * val;
  }
  sum = simd_sum(sum);
  if (lane == 0) partial[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = tid < 8 ? partial[tid] : 0.0f;
  total = simd_sum(total);
  if (tid == 0) partial[0] = precise::rsqrt(total / float(width) + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = partial[0];
  for (uint c = tid; c < width; c += 256) {
    scalar normalized = scalar(float(x[row * width + c]) * inv);
    y[row * width + c] = normalized * weight[c];
  }
}

kernel void rope_qkv(
    device const scalar* x [[buffer(0)]],
    device const float* cosines [[buffer(1)]],
    device const float* sines [[buffer(2)]],
    device scalar* y [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  uint total = 3 * heads * seq * dim;
  if (idx >= total) return;
  uint d = idx % dim;
  uint s = (idx / dim) % seq;
  uint h = (idx / (dim * seq)) % heads;
  uint qkv = idx / (dim * seq * heads);
  uint base = s * 3 * heads * dim + qkv * heads * dim + h * dim;
  if (qkv == 2) { y[idx] = x[base + d]; return; }
  uint half_dim = dim / 2;
  uint rotated_d = d < half_dim ? d + half_dim : d - half_dim;
  float rotated = float(x[base + rotated_d]) * (d < half_dim ? -1.0f : 1.0f);
  uint pos = s * half_dim + d % half_dim;
  y[idx] = scalar(float(x[base + d]) * cosines[pos] + rotated * sines[pos]);
}

kernel void swiglu(
    device const scalar* x [[buffer(0)]],
    device scalar* y [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= total) return;
  uint row = idx / width;
  uint c = idx % width;
  float gate = float(x[row * width * 2 + c]);
  scalar activated = scalar(gate / (1.0f + exp(-gate)));
  y[idx] = activated * x[row * width * 2 + width + c];
}
"""


@lru_cache(maxsize=3)
def _shaders(dtype: torch.dtype):
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("this PyTorch MPS build does not expose compile_shader")
    names = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
    if dtype not in names:
        raise ValueError(f"unsupported MPS dtype: {dtype}")
    return torch.mps.compile_shader(_METAL.replace("DTYPE", names[dtype]))


def _rms_norm(x: Tensor, module: nn.Module) -> Tensor:
    x = x if x.is_contiguous() else x.contiguous()
    output = torch.empty_like(x)
    width = x.shape[-1]
    _shaders(x.dtype).rms_norm(
        x,
        module.weight,
        output,
        width,
        float(module.eps),
        threads=(x.numel() // width * 256,),
        group_size=(256,),
    )
    return output


def _swiglu(projected: Tensor) -> Tensor:
    width = projected.shape[-1] // 2
    output = torch.empty((*projected.shape[:-1], width), device=projected.device, dtype=projected.dtype)
    _shaders(projected.dtype).swiglu(
        projected,
        output,
        width,
        output.numel(),
        threads=(output.numel(),),
        group_size=(256,),
    )
    return output


def _tensor_identity(tensor: Tensor) -> tuple[int, str, torch.dtype, int, int | None]:
    """Cheap metadata identity for detecting an invalidated inference cache."""

    try:
        version: int | None = tensor._version
    except RuntimeError:
        # Parameters materialized under inference_mode intentionally omit a
        # version counter. Device/dtype/storage identity still catches moves;
        # such a model is treated as immutable after runtime construction.
        version = None
    return (id(tensor), str(tensor.device), tensor.dtype, tensor.data_ptr(), version)


@dataclass(frozen=True)
class MPSBatch1Options:
    """Numerically conservative switches for the batch-1 runtime."""

    metal_norm: bool = True
    metal_rope: bool = True
    packed_encoder_swiglu: bool = True
    metal_swiglu: bool = True
    cache_grids: int = 32


class _Batch1Encoder(nn.Module):
    """Single-image Monkey encoder with no batching or attention mask."""

    def __init__(self, encoder: nn.Module, options: MPSBatch1Options):
        super().__init__()
        if next(encoder.parameters()).device.type != "mps":
            raise ValueError("encoder weights must already be on MPS")
        # The complete base model already owns the encoder parameters. Keep a
        # weak reference here so the optimized view does not publish duplicate
        # state_dict names for the same tensors.
        object.__setattr__(self, "_encoder_ref", weakref.ref(encoder))
        self.options = options
        self._positions: OrderedDict[tuple[int, int], tuple[Tensor, Tensor]] = OrderedDict()
        self._packed_mlp: list[tuple[Tensor, Tensor | None]] = []
        self._packed_sources: list[tuple[tuple, tuple]] = []
        if options.packed_encoder_swiglu:
            with torch.no_grad():
                for block in encoder.blocks:
                    mlp = block.mlp
                    weight = torch.cat((mlp.fc1.weight, mlp.fc3.weight), dim=0)
                    bias = None if mlp.fc1.bias is None else torch.cat((mlp.fc1.bias, mlp.fc3.bias))
                    self._packed_mlp.append((weight, bias))
                    self._packed_sources.append(
                        (_tensor_identity(mlp.fc1.weight), _tensor_identity(mlp.fc3.weight))
                    )

    @property
    def encoder(self) -> nn.Module:
        encoder = self._encoder_ref()
        if encoder is None:
            raise RuntimeError("base encoder was released")
        return encoder

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def clear_cache(self) -> None:
        self._positions.clear()

    def _validate_packed_weights(self) -> None:
        if not self.options.packed_encoder_swiglu:
            return
        current = [
            (_tensor_identity(block.mlp.fc1.weight), _tensor_identity(block.mlp.fc3.weight))
            for block in self.encoder.blocks
        ]
        if current != self._packed_sources:
            raise RuntimeError(
                "base encoder weights/device/dtype changed after MPSBatch1OCR construction; "
                "construct a new optimized runtime"
            )

    def _norm(self, x: Tensor, module: nn.Module) -> Tensor:
        return _rms_norm(x, module) if self.options.metal_norm else module(x)

    def _position(self, height: int, width: int) -> tuple[Tensor, Tensor]:
        key = (height, width)
        if key not in self._positions:
            grid = torch.tensor([[1, height, width]], dtype=torch.long)
            frequencies = self.encoder.rot_pos_emb(grid)
            self._positions[key] = (
                frequencies.cos().float().contiguous(),
                frequencies.sin().float().contiguous(),
            )
            if len(self._positions) > self.options.cache_grids:
                self._positions.popitem(last=False)
        self._positions.move_to_end(key)
        return self._positions[key]

    def _attention(self, x: Tensor, attention: nn.Module, cosine: Tensor, sine: Tensor) -> Tensor:
        sequence = x.shape[0]
        heads = attention.num_heads
        qkv = attention.qkv(x)
        head_dim = qkv.shape[-1] // (3 * heads)
        if self.options.metal_rope:
            arranged = torch.empty((3, 1, heads, sequence, head_dim), device=x.device, dtype=x.dtype)
            _shaders(x.dtype).rope_qkv(
                qkv,
                cosine,
                sine,
                arranged,
                sequence,
                heads,
                head_dim,
                threads=(arranged.numel(),),
                group_size=(256,),
            )
            q, k, v = arranged.unbind(0)
        else:
            q, k, v = qkv.reshape(sequence, 3, heads, head_dim).permute(1, 2, 0, 3).unsqueeze(1).unbind(0)
            cos = cosine.repeat(1, 2)[None, None]
            sin = sine.repeat(1, 2)[None, None]

            def rotate(tensor: Tensor) -> Tensor:
                floating = tensor.float()
                first, second = floating.chunk(2, dim=-1)
                return (floating * cos + torch.cat((-second, first), dim=-1) * sin).to(tensor.dtype)

            q, k = rotate(q), rotate(k)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return attention.proj(attended.transpose(1, 2).reshape(sequence, -1))

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        if self.training or self.encoder.training:
            raise RuntimeError("MPS batch-1 encoder is inference-only")
        if pixel_values.device.type != "mps" or pixel_values.dtype != self.dtype:
            raise ValueError(f"pixel_values must be MPS {self.dtype}")
        if not isinstance(grid_thw, Tensor) or grid_thw.device.type != "cpu" or tuple(grid_thw.shape) != (1, 3):
            raise ValueError("grid_thw must be a CPU [1,3] tensor")
        temporal, height, width = map(int, grid_thw[0].tolist())
        if temporal != 1 or min(height, width) < 1 or height % 2 or width % 2:
            raise ValueError("expected one still image with a positive even patch grid")
        sequence = height * width
        if tuple(pixel_values.shape) != (sequence, 3 * 14 * 14):
            raise ValueError("pixel_values do not match the declared image grid")
        self._validate_packed_weights()

        patchifier = self.encoder.patch_embed.patchifier
        x = F.linear(pixel_values, patchifier.proj.weight.flatten(1), patchifier.proj.bias)
        x = self._norm(x, patchifier.norm)
        cosine, sine = self._position(height, width)
        for index, block in enumerate(self.encoder.blocks):
            attention = getattr(block.attn, "qkv", None)
            if attention is None:
                raise ValueError("encoder attention does not expose qkv/proj parameters")
            x = x + self._attention(self._norm(x, block.norm1), block.attn, cosine, sine)
            normalized = self._norm(x, block.norm2)
            if self.options.packed_encoder_swiglu:
                weight, bias = self._packed_mlp[index]
                projected = F.linear(normalized, weight, bias)
                hidden = _swiglu(projected) if self.options.metal_swiglu else (
                    F.silu(projected.chunk(2, dim=-1)[0]) * projected.chunk(2, dim=-1)[1]
                )
                residual = block.mlp.fc2(hidden)
            else:
                residual = block.mlp(normalized)
            x = x + residual
        if self.encoder.config.post_norm:
            x = self._norm(x, self.encoder.post_trunk_norm)
        return x


class MPSBatch1OCR(nn.Module):
    """Low-latency one-image view of an already loaded Q129 model."""

    def __init__(self, model: MonkeyQ129OCR, options: MPSBatch1Options | None = None):
        super().__init__()
        if model.training:
            raise ValueError("call eval() on the base model before constructing MPSBatch1OCR")
        if next(model.parameters()).device.type != "mps":
            raise ValueError("move the base model to MPS before constructing MPSBatch1OCR")
        self.model = model
        self.options = options or MPSBatch1Options()
        if self.options.cache_grids < 1:
            raise ValueError("cache_grids must be positive")
        self.encoder = _Batch1Encoder(model.encoder, self.options)
        self._spatial_positions: OrderedDict[tuple[int, int], Tensor] = OrderedDict()
        self._spatial_source = _tensor_identity(model.spatial_position)
        self.eval()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def clear_cache(self) -> None:
        self.encoder.clear_cache()
        self._spatial_positions.clear()

    def _norm(self, x: Tensor, module: nn.Module) -> Tensor:
        return _rms_norm(x, module) if self.options.metal_norm else module(x)

    def _spatial_position(self, height: int, width: int) -> Tensor:
        if _tensor_identity(self.model.spatial_position) != self._spatial_source:
            raise RuntimeError(
                "base spatial position/device/dtype changed after MPSBatch1OCR construction; "
                "construct a new optimized runtime"
            )
        key = (height, width)
        if key not in self._spatial_positions:
            self._spatial_positions[key] = _interpolate_2d_pos(
                self.model.spatial_position, height, width
            )[0].to(dtype=self.dtype).contiguous()
            if len(self._spatial_positions) > self.options.cache_grids:
                self._spatial_positions.popitem(last=False)
        self._spatial_positions.move_to_end(key)
        return self._spatial_positions[key]

    @staticmethod
    def _cross_attention(queries: Tensor, memory: Tensor, module: nn.Module) -> Tensor:
        # There is exactly one unpadded image, so an attention mask would be
        # all True and only add dispatch/mask materialization overhead.
        _, query_count, _ = queries.shape
        memory_count = memory.shape[1]
        heads = module.num_heads
        head_dim = module.head_dim
        q = module.q(queries).reshape(1, query_count, heads, head_dim).transpose(1, 2)
        k, v = module.kv(memory).reshape(
            1, memory_count, 2, heads, head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return module.output(attended.transpose(1, 2).reshape(1, query_count, -1))

    @torch.inference_mode()
    def forward(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        *,
        include_occupancy: bool = False,
    ) -> dict[str, Tensor]:
        if self.training or self.model.training:
            raise RuntimeError("MPSBatch1OCR is inference-only")
        if pixel_values.device.type != "mps" or pixel_values.dtype != self.dtype:
            raise ValueError(f"pixel_values must be MPS {self.dtype}")
        if not isinstance(image_grid_thw, Tensor) or image_grid_thw.device.type != "cpu" or tuple(image_grid_thw.shape) != (1, 3):
            raise ValueError("image_grid_thw must be a CPU [1,3] tensor")
        _, height, width = map(int, image_grid_thw[0].tolist())

        visual = self.encoder(pixel_values, image_grid_thw)
        projected = self.model.memory_projection(visual)
        memory = (
            packed_to_raster(projected, height, width)
            + self._spatial_position(height, width)
        ).unsqueeze(0)
        queries = self.model.query_position.to(dtype=memory.dtype)
        for layer in self.model.decoder:
            queries = queries + layer.self_attention(self._norm(queries, layer.self_norm))
            queries = queries + self._cross_attention(
                self._norm(queries, layer.cross_norm), memory, layer.cross_attention
            )
            normalized = self._norm(queries, layer.ffn_norm)
            projected_ffn = layer.ffn.gate_up(normalized)
            hidden = _swiglu(projected_ffn) if self.options.metal_swiglu else (
                F.silu(projected_ffn.chunk(2, dim=-1)[0]) * projected_ffn.chunk(2, dim=-1)[1]
            )
            queries = queries + layer.ffn.down(hidden)

        decoded = self._norm(queries, self.model.final_norm)
        outputs = {"logits": self.model.classifier(decoded)}
        if include_occupancy:
            outputs["occupancy_logits"] = self.model.occupancy_head(decoded).squeeze(-1)
        return outputs
