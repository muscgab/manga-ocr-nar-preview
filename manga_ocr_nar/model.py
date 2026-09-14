"""MonkeyOCRv2-B encoder with the R2 Q129 non-autoregressive decoder.

The only pretrained component here is the reviewed, local MonkeyOCRv2-B
vision encoder. The decoder is initialized from scratch. ``image_grid_thw``
is deliberately a CPU collate artifact: make :class:`ImageBatchPlan` before
moving image patches to an accelerator, then pass that plan to ``forward``.
That avoids a device-to-CPU scalar transfer in every encoder attention layer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .position import _interpolate_2d_pos
from .upstream.configuration_monkeyocrv2vit import MonkeyOCRv2VisionConfig
from .upstream.modeling_monkeyocrv2_vision import (
    MonkeyOCRv2VisionTransformer,
    apply_rotary_pos_emb_vision,
)


PAD_ID = -100
VOCAB_SIZE = 6816
EOS_ID = VOCAB_SIZE
NUM_CLASSES = VOCAB_SIZE + 1
MAX_QUERIES = 129


@dataclass(frozen=True)
class ImageBatchPlan:
    """Static CPU metadata for one packed Monkey image batch.

    ``sample_ranges`` entries are ``(start, stop, height, width)`` in native
    packed-token order. ``attention_groups`` collects same-length ranges so
    the encoder can use dense SDPA without any cross-image attention or a
    concatenated quadratic mask. This object contains only Python integers
    and a CPU grid, so using it in every layer cannot synchronize an accelerator.
    """

    grid_thw: Tensor
    sample_ranges: tuple[tuple[int, int, int, int], ...]
    attention_groups: tuple[tuple[tuple[int, int], ...], ...]
    max_memory_tokens: int

    @property
    def batch_size(self) -> int:
        return len(self.sample_ranges)

    @property
    def total_tokens(self) -> int:
        return self.sample_ranges[-1][1]


def make_image_batch_plan(image_grid_thw: Tensor) -> ImageBatchPlan:
    """Build CPU-only packed-image metadata once in the collate function.

    The processor returns one ``[T, H, W]`` row per input image. R2 accepts
    still images (``T == 1``) and Qwen's native 2x2 packing requires positive,
    even H and W. GPU grids are rejected instead of silently copied back and
    synchronizing the training stream.
    """

    if not isinstance(image_grid_thw, Tensor):
        raise TypeError("image_grid_thw must be a CPU torch.Tensor")
    if image_grid_thw.device.type != "cpu":
        raise ValueError(
            "image_grid_thw must remain on CPU; call make_image_batch_plan in collate "
            "before moving pixel_values to the accelerator"
        )
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3 or image_grid_thw.shape[0] == 0:
        raise ValueError("image_grid_thw must be a nonempty CPU [B,3] tensor")
    if image_grid_thw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("image_grid_thw must use an integer dtype")

    grid = image_grid_thw.detach().to(dtype=torch.long).contiguous()
    rows = grid.tolist()  # Explicit CPU-only collate work, never a layer-time readback.
    ranges: list[tuple[int, int, int, int]] = []
    grouped: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for temporal, height, width in rows:
        if temporal != 1 or height <= 0 or width <= 0 or height % 2 or width % 2:
            raise ValueError("Q129 accepts still images with positive, even native patch grids")
        count = height * width
        ranges.append((offset, offset + count, height, width))
        grouped.setdefault(count, []).append((offset, offset + count))
        offset += count
    return ImageBatchPlan(
        grid_thw=grid,
        sample_ranges=tuple(ranges),
        attention_groups=tuple(tuple(group) for group in grouped.values()),
        max_memory_tokens=max(stop - start for start, stop, _, _ in ranges),
    )


class ImageIsolatedSDPA(nn.Module):
    """Mathematically equivalent per-image Monkey attention using functional SDPA.

    The reviewed encoder accepts concatenated visual tokens. This adapter
    batches only images with the same token count and calls SDPA for each
    static group. Unlike the old prototype, it never derives Python lengths
    from ``cu_seqlens`` inside every block.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.qkv = original.qkv
        self.proj = original.proj
        self.num_heads = original.num_heads
        self._batch_plan: ImageBatchPlan | None = None

    def set_batch_plan(self, batch_plan: ImageBatchPlan) -> None:
        self._batch_plan = batch_plan

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        rotary_pos_emb: Tensor | None = None,
    ) -> Tensor:
        del cu_seqlens
        plan = self._batch_plan
        if plan is None:
            raise RuntimeError("configure ImageIsolatedSDPA with an ImageBatchPlan before encoder forward")
        if hidden_states.shape[0] != plan.total_tokens:
            raise ValueError("packed token count does not match the current ImageBatchPlan")

        sequence_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(
            sequence_length, 3, self.num_heads, -1
        ).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        by_range: dict[tuple[int, int], Tensor] = {}
        for group in plan.attention_groups:
            q_group = torch.stack([q[start:stop] for start, stop in group], dim=0).transpose(1, 2)
            k_group = torch.stack([k[start:stop] for start, stop in group], dim=0).transpose(1, 2)
            v_group = torch.stack([v[start:stop] for start, stop in group], dim=0).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q_group, k_group, v_group, dropout_p=0.0, is_causal=False
            ).transpose(1, 2)
            by_range.update(zip(group, attended.unbind(0)))
        ordered = [by_range[(start, stop)] for start, stop, _, _ in plan.sample_ranges]
        return self.proj(torch.cat(ordered, dim=0).reshape(sequence_length, -1))


def use_image_isolated_sdpa(encoder: nn.Module) -> nn.Module:
    """Replace reviewed encoder attention while preserving its parameter names."""

    for block in getattr(encoder, "blocks", []):
        if not isinstance(block.attn, ImageIsolatedSDPA):
            block.attn = ImageIsolatedSDPA(block.attn)
    return encoder


def _set_encoder_batch_plan(encoder: nn.Module, batch_plan: ImageBatchPlan) -> None:
    for block in getattr(encoder, "blocks", []):
        if isinstance(block.attn, ImageIsolatedSDPA):
            block.attn.set_batch_plan(batch_plan)


class LinearPatchEmbed(nn.Module):
    """Exact patch-projection algebra without MIOpen's dynamic-N Conv2d find.

    Monkey's vendored patchifier first reshapes every packed 14x14 RGB patch to
    ``[N, 3, 14, 14]`` and applies a stride-one convolution whose kernel covers
    the complete patch.  That operation is exactly a linear projection of the
    flattened patch.  Keeping ``proj`` as the original ``nn.Conv2d`` preserves
    all checkpoint parameter names, shapes, and loaded values; only its forward
    dispatch changes to a GEMM-friendly ``F.linear``.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.num_channels = original.num_channels
        self.patch_size = original.patch_size
        self.temporal_patch_size = original.temporal_patch_size
        self.embed_dim = original.embed_dim
        self.config = original.config
        self.proj = original.proj
        self.norm = original.norm

    def forward(self, pixel_values: Tensor, grid_thw: Tensor | None = None) -> Tensor:
        del grid_thw
        patches = pixel_values.view(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        projected = F.linear(patches.flatten(1), self.proj.weight.flatten(1), self.proj.bias)
        return self.norm(projected)


def use_linear_patch_embed(encoder: nn.Module) -> nn.Module:
    """Swap only Monkey's patchifier implementation, retaining its state keys."""

    patch_embed = getattr(encoder, "patch_embed", None)
    patchifier = getattr(patch_embed, "patchifier", None)
    if patchifier is None:
        raise ValueError("encoder must expose patch_embed.patchifier")
    if not isinstance(patchifier, LinearPatchEmbed):
        patch_embed.patchifier = LinearPatchEmbed(patchifier)
    return encoder


def build_local_encoder_from_config(config_path: str | Path) -> MonkeyOCRv2VisionTransformer:
    """Construct the reviewed encoder from config without loading base weights.

    Model-only R2 safetensors include the encoder state themselves.  This
    constructor lets a read-only inference tool instantiate the same adapted
    architecture from the checked-in runtime config before strict state load.
    """

    raw = json.loads(Path(config_path).read_text())
    expected = dict(
        embed_dim=768,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        num_channels=3,
        use_bias=False,
        post_norm=True,
        is_causal=False,
        rms_norm_eps=1e-5,
    )
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("local config is not the reviewed MonkeyOCRv2-B encoder")
    raw["vision_attn_implementation"] = "eager_v2"
    encoder = MonkeyOCRv2VisionTransformer(MonkeyOCRv2VisionConfig(**raw))
    if sum(parameter.numel() for parameter in encoder.parameters()) != 113_718_528:
        raise ValueError("unexpected standalone encoder parameter count")
    return use_linear_patch_embed(use_image_isolated_sdpa(encoder))


def load_local_encoder(directory: str | Path, *, expected_sha256: str) -> MonkeyOCRv2VisionTransformer:
    """Strictly load the reviewed local MonkeyOCRv2-B encoder and its SHA256."""

    from safetensors.torch import load_file

    directory = Path(directory)
    weight = directory / "model.safetensors"
    with weight.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"MonkeyOCRv2-B weight hash mismatch: {actual}")
    encoder = build_local_encoder_from_config(directory / "config.json")
    encoder.load_state_dict(load_file(str(weight), device="cpu"), strict=True)
    return encoder


def make_image_processor(*, min_pixels: int = 224 * 224, max_pixels: int = 448 * 448):
    """Return the native Qwen patch packer and Monkey-compatible normalization."""

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    if not 28 * 28 <= min_pixels <= max_pixels:
        raise ValueError("invalid image area limits")
    return Qwen2VLImageProcessor(
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        patch_size=14,
        temporal_patch_size=1,
        merge_size=2,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
    )


def packed_to_raster(tokens: Tensor, height: int, width: int) -> Tensor:
    """Undo Qwen's 2x2 packed patch order without merging visual tokens."""

    return tokens.reshape(height // 2, width // 2, 2, 2, -1).permute(
        0, 2, 1, 3, 4
    ).reshape(height * width, -1)


class RMSNorm(nn.Module):
    """Pre-normalization with FP32 variance reduction for CUDA and MPS."""

    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=input_dtype) * self.weight.to(dtype=input_dtype)


class SelfAttention(nn.Module):
    """Bias-free, fused-QKV, non-causal multi-head SDPA."""

    def __init__(self, d_model: int, num_heads: int, attention_dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = attention_dropout

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.output(attended.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class CrossAttention(nn.Module):
    """Bias-free Q plus fused-KV attention over one image's padded memory."""

    def __init__(self, d_model: int, num_heads: int, attention_dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = attention_dropout

    def forward(self, queries: Tensor, memory: Tensor, memory_valid: Tensor) -> Tensor:
        batch_size, query_count, _ = queries.shape
        memory_count = memory.shape[1]
        q = self.q(queries).reshape(
            batch_size, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.kv(memory).reshape(
            batch_size, memory_count, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        # Functional SDPA bool masks use True for positions that participate.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=memory_valid[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.output(attended.transpose(1, 2).reshape(batch_size, query_count, -1))


class SwiGLU(nn.Module):
    """Bias-free fused gate/up projection with 2048 hidden features by default."""

    def __init__(self, d_model: int, hidden_features: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden_features, bias=False)
        self.down = nn.Linear(hidden_features, d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = self.gate_up(hidden_states).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class DecoderBlock(nn.Module):
    """Three bias-free pre-RMSNorm residual sublayers."""

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.self_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.self_attention = SelfAttention(
            config.d_model, config.decoder_heads, config.attention_dropout
        )
        self.cross_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.cross_attention = CrossAttention(
            config.d_model, config.decoder_heads, config.attention_dropout
        )
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.decoder_ffn_dim)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, queries: Tensor, memory: Tensor, memory_valid: Tensor) -> Tensor:
        queries = queries + self.residual_dropout(self.self_attention(self.self_norm(queries)))
        queries = queries + self.residual_dropout(
            self.cross_attention(self.cross_norm(queries), memory, memory_valid)
        )
        return queries + self.residual_dropout(self.ffn(self.ffn_norm(queries)))


@dataclass
class ModelConfig:
    """Fixed R2 decoder contract; defaults are the trainable 8x768 model."""

    d_model: int = 768
    decoder_layers: int = 8
    decoder_heads: int = 12
    decoder_ffn_dim: int = 2048
    residual_dropout: float = 0.1
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    position_grid: int = 32
    vocab_size: int = VOCAB_SIZE
    max_queries: int = MAX_QUERIES
    head_bias: bool = True
    occupancy_weight: float = 0.1

    @property
    def eos_id(self) -> int:
        return self.vocab_size

    @property
    def num_classes(self) -> int:
        return self.vocab_size + 1


# Keep the old exported name import-compatible while its defaults now implement R2.
Q129Config = ModelConfig


class MonkeyQ129OCR(nn.Module):
    """MonkeyOCRv2-B plus the fixed-length, ordered R2 Q129 decoder."""

    def __init__(self, encoder: nn.Module, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.d_model % cfg.decoder_heads:
            raise ValueError("d_model must be divisible by decoder_heads")
        if cfg.max_queries != MAX_QUERIES:
            raise ValueError("R2 is fixed to 129 ordered queries")
        if cfg.vocab_size != VOCAB_SIZE:
            raise ValueError("R2 is fixed to the 6816-character NAR vocabulary")
        if cfg.decoder_layers <= 0 or cfg.decoder_ffn_dim <= 0 or cfg.position_grid <= 0:
            raise ValueError("decoder dimensions must be positive")
        if not 0.0 <= cfg.residual_dropout < 1.0 or not 0.0 <= cfg.attention_dropout < 1.0:
            raise ValueError("dropout values must be in [0, 1)")

        self.encoder = encoder
        encoder_width = _encoder_embed_dim(encoder)
        self.memory_projection = nn.Linear(encoder_width, cfg.d_model, bias=False)
        self.spatial_position = nn.Parameter(
            torch.empty(1, cfg.position_grid * cfg.position_grid, cfg.d_model)
        )
        self.query_position = nn.Parameter(torch.empty(1, MAX_QUERIES, cfg.d_model))
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes, bias=cfg.head_bias)
        self.occupancy_head = nn.Linear(cfg.d_model, 1, bias=cfg.head_bias)
        self._reset_decoder_parameters()

    @property
    def backbone(self) -> nn.Module:
        """Compatibility read-only alias; use ``encoder`` for optimizer groups."""

        return self.encoder

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.parameters()

    def parameter_report(self) -> dict[str, int]:
        encoder_parameters = sum(parameter.numel() for parameter in self.encoder.parameters())
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "encoder_parameters": encoder_parameters,
            "decoder_and_heads_parameters": total_parameters - encoder_parameters,
            "total_parameters": total_parameters,
        }

    def _reset_decoder_parameters(self) -> None:
        components = (
            self.memory_projection,
            self.decoder,
            self.final_norm,
            self.classifier,
            self.occupancy_head,
        )
        for component in components:
            for module in component.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.query_position, std=0.02)

    def forward(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        labels: Tensor | None = None,
        *,
        batch_plan: ImageBatchPlan | None = None,
    ) -> dict[str, Tensor]:
        """Return Q129 logits and, with labels, ``loss``, ``ce``, and ``co``.

        ``pixel_values`` remains the Qwen packed patch tensor ``[sum(H*W),588]``.
        Keep ``image_grid_thw`` on CPU. A precomputed ``batch_plan`` is the
        preferred training API and makes all grouping decisions before GPU work.
        """

        if pixel_values.ndim != 2 or pixel_values.shape[1] != 3 * 14 * 14:
            raise ValueError("pixel_values must be native packed [sum(H*W),588] patches")
        if batch_plan is None:
            batch_plan = make_image_batch_plan(image_grid_thw)
        else:
            _validate_plan_matches_grid(batch_plan, image_grid_thw)
        if pixel_values.shape[0] != batch_plan.total_tokens:
            raise ValueError("pixel_values token count does not match image_grid_thw")

        _set_encoder_batch_plan(self.encoder, batch_plan)
        # ``grid_thw`` deliberately remains CPU. The vendored encoder builds
        # RoPE coordinates from it once; adapted attention layers use the static
        # plan and contain no .cpu(), .item(), or .tolist() operations.
        visual = self.encoder(pixel_values, grid_thw=batch_plan.grid_thw, bf16=False)
        if visual.ndim != 2 or visual.shape[0] != batch_plan.total_tokens:
            raise ValueError("encoder must return one [embed_dim] feature per packed patch")
        memory = self._make_memory(self.memory_projection(visual), batch_plan)
        lengths = torch.as_tensor(
            [stop - start for start, stop, _, _ in batch_plan.sample_ranges],
            device=memory.device,
            dtype=torch.long,
        )
        memory_valid = torch.arange(memory.shape[1], device=memory.device)[None, :] < lengths[:, None]
        queries = self.query_position.to(dtype=memory.dtype).expand(batch_plan.batch_size, -1, -1)
        for layer in self.decoder:
            queries = layer(queries, memory, memory_valid)
        decoded = self.final_norm(queries)
        outputs: dict[str, Tensor] = {
            "logits": self.classifier(decoded),
            "occupancy_logits": self.occupancy_head(decoded).squeeze(-1),
            "memory_valid_mask": memory_valid,
            "patch_key_padding_mask": ~memory_valid,
        }
        if labels is not None:
            outputs.update(
                compute_ocr_loss(
                    outputs,
                    labels,
                    pad_id=PAD_ID,
                    eos_id=self.config.eos_id,
                    max_queries=self.config.max_queries,
                    occupancy_weight=self.config.occupancy_weight,
                )
            )
        return outputs

    def _make_memory(self, projected: Tensor, batch_plan: ImageBatchPlan) -> Tensor:
        memories: list[Tensor] = []
        for start, stop, height, width in batch_plan.sample_ranges:
            raster = packed_to_raster(projected[start:stop], height, width)
            position = _interpolate_2d_pos(self.spatial_position, height, width)[0]
            memories.append(raster + position.to(dtype=raster.dtype))
        return pad_sequence(memories, batch_first=True)


def _encoder_embed_dim(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    value = getattr(config, "embed_dim", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("encoder must expose config.embed_dim")
    return value


def _validate_plan_matches_grid(batch_plan: ImageBatchPlan, image_grid_thw: Tensor) -> None:
    if not isinstance(image_grid_thw, Tensor) or image_grid_thw.device.type != "cpu":
        raise ValueError("image_grid_thw must remain on CPU even when batch_plan is supplied")
    incoming = image_grid_thw.detach().to(dtype=torch.long).contiguous()
    if not torch.equal(incoming, batch_plan.grid_thw):
        raise ValueError("batch_plan was built for a different image_grid_thw")


def validate_q129_labels(
    labels: Tensor,
    *,
    pad_id: int = PAD_ID,
    eos_id: int = EOS_ID,
    max_queries: int = MAX_QUERIES,
) -> None:
    """Validate the ordered body, one EOS, then PAD target contract.

    Call this in CPU data validation or a debug assertion. The training loss
    itself is vectorized and does not issue target-validation readbacks.
    """

    if labels.ndim != 2 or labels.shape[1] != max_queries:
        raise ValueError(f"labels must be [B,{max_queries}]")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("labels must be an integer tensor")
    invalid = (labels != pad_id) & ((labels < 0) | (labels > eos_id))
    if bool(invalid.any()):
        raise ValueError("labels contain an invalid character id")
    is_eos = labels == eos_id
    if not bool(torch.all(is_eos.sum(dim=1) == 1)):
        raise ValueError("each target must contain exactly one EOS")
    positions = torch.arange(max_queries, device=labels.device).unsqueeze(0)
    eos_position = is_eos.to(torch.long).argmax(dim=1, keepdim=True)
    if bool(torch.any((labels == pad_id) & (positions < eos_position))):
        raise ValueError("PAD cannot appear before EOS")
    if bool(torch.any((labels != pad_id) & (positions > eos_position))):
        raise ValueError("only PAD may appear after EOS")


def compute_ocr_loss(
    outputs: Mapping[str, Tensor],
    labels: Tensor,
    *,
    pad_id: int = PAD_ID,
    eos_id: int = EOS_ID,
    max_queries: int = MAX_QUERIES,
    occupancy_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Per-sample ordered CE plus class-balanced occupancy BCE.

    CE includes character tokens and EOS but excludes PAD. Occupancy regards
    every non-PAD position, including EOS, as positive; PAD slots are negative.
    Positive and negative slots contribute equally within each sample whenever
    both are present, which prevents short labels' padded tail from dominating.
    """

    logits = outputs["logits"]
    occupancy_logits = outputs["occupancy_logits"]
    if logits.ndim != 3 or logits.shape[1] != max_queries or logits.shape[2] != eos_id + 1:
        raise ValueError(f"logits must be [B,{max_queries},{eos_id + 1}]")
    if occupancy_logits.shape != logits.shape[:2] or labels.shape != logits.shape[:2]:
        raise ValueError("occupancy logits and labels must agree with Q129 logits")
    if occupancy_weight < 0.0:
        raise ValueError("occupancy_weight must be nonnegative")

    targets = labels.to(device=logits.device, dtype=torch.long)
    valid = targets != pad_id
    safe_targets = targets.masked_fill(~valid, 0)
    token_ce = F.cross_entropy(logits.transpose(1, 2), safe_targets, reduction="none")
    valid_counts = valid.sum(dim=1).clamp_min(1)
    ce = (token_ce * valid).sum(dim=1).div(valid_counts).mean()

    occupied = valid.to(dtype=occupancy_logits.dtype)
    positive = F.softplus(-occupancy_logits)
    negative = F.softplus(occupancy_logits)
    positive_count = occupied.sum(dim=1)
    negative_count = (1.0 - occupied).sum(dim=1)
    positive_mean = (positive * occupied).sum(dim=1) / positive_count.clamp_min(1)
    negative_mean = (negative * (1.0 - occupied)).sum(dim=1) / negative_count.clamp_min(1)
    has_positive = (positive_count > 0).to(positive_mean.dtype)
    has_negative = (negative_count > 0).to(negative_mean.dtype)
    co = (
        positive_mean * has_positive + negative_mean * has_negative
    ).div((has_positive + has_negative).clamp_min(1)).mean()
    total = ce + occupancy_weight * co
    return {
        "loss": total,
        "ce": ce,
        "co": co,
        "ordered_ce_loss": ce,
        "occupancy_bce_loss": co,
        "valid_token_count": valid.sum(),
    }


__all__ = [
    "CrossAttention",
    "EOS_ID",
    "ImageBatchPlan",
    "ImageIsolatedSDPA",
    "MAX_QUERIES",
    "ModelConfig",
    "MonkeyQ129OCR",
    "NUM_CLASSES",
    "PAD_ID",
    "Q129Config",
    "RMSNorm",
    "SelfAttention",
    "VOCAB_SIZE",
    "compute_ocr_loss",
    "build_local_encoder_from_config",
    "load_local_encoder",
    "make_image_batch_plan",
    "make_image_processor",
    "packed_to_raster",
    "use_image_isolated_sdpa",
    "validate_q129_labels",
]
