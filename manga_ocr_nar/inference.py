"""Single-image inference API for manga-ocr-nar-preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from safetensors.torch import load_file

from .model import EOS_ID, ModelConfig, MonkeyQ129OCR, build_local_encoder_from_config, make_image_processor
from .mps_batch1 import MPSBatch1OCR


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype(device: torch.device, name: str) -> torch.dtype:
    if name != "auto":
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class MangaOCRNar:
    """Load one release directory and recognize cropped text regions."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        precision: str = "auto",
        optimized_mps: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        manifest = json.loads((self.model_dir / "model_manifest.json").read_text())
        weights = self.model_dir / manifest["model"]["filename"]
        if weights.stat().st_size != manifest["model"]["bytes"] or _sha256(weights) != manifest["model"]["sha256"]:
            raise ValueError("model.safetensors does not match model_manifest.json")
        vocab_path = self.model_dir / "nar_vocabulary.json"
        if _sha256(vocab_path) != manifest["contract"]["vocab_sha256"]:
            raise ValueError("nar_vocabulary.json does not match the model contract")
        vocabulary = json.loads(vocab_path.read_text())["characters"]
        if len(vocabulary) != EOS_ID or len(set(vocabulary)) != EOS_ID:
            raise ValueError("invalid 6816-character vocabulary")

        self.device = _device(device)
        self.dtype = _dtype(self.device, precision)
        if self.device.type == "mps" and self.dtype == torch.bfloat16:
            raise ValueError("use fp32 or fp16 on MPS")
        if self.device.type == "cpu" and self.dtype != torch.float32:
            raise ValueError("CPU inference uses fp32")

        encoder = build_local_encoder_from_config(self.model_dir / "config.json")
        model = MonkeyQ129OCR(encoder, ModelConfig())
        state = load_file(str(weights), device="cpu")
        model.load_state_dict(state, strict=True)
        del state
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.runner = self.model
        self.mps_optimized = False
        if self.device.type == "mps" and optimized_mps:
            if not hasattr(torch.mps, "compile_shader"):
                raise RuntimeError("optimized MPS inference requires torch.mps.compile_shader")
            self.runner = MPSBatch1OCR(self.model)
            self.mps_optimized = True
        self.processor = make_image_processor(
            min_pixels=int(manifest["contract"]["image_min_pixels"]),
            max_pixels=int(manifest["contract"]["image_max_pixels"]),
        )
        self.characters = vocabulary
        self.manifest = manifest

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path) -> str:
        close = False
        if isinstance(image, (str, Path)):
            image = Image.open(image)
            close = True
        try:
            encoded = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        finally:
            if close:
                image.close()
        pixels = encoded["pixel_values"].to(device=self.device, dtype=self.dtype)
        grid = encoded["image_grid_thw"]
        logits = self.runner(pixels, grid)["logits"]
        ids = logits.argmax(-1)[0].cpu().tolist()
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID)]
        return "".join(self.characters[index] for index in ids if 0 <= index < EOS_ID)

    def predict_all(self, images: Iterable[Image.Image | str | Path]) -> list[str]:
        return [self.predict(image) for image in images]

