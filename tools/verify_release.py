#!/usr/bin/env python3
"""Verify release identities and optionally run one OCR smoke image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from safetensors import safe_open


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    args = parser.parse_args()
    root = args.model_dir.resolve()
    sys.path.insert(0, str(root))
    manifest = json.loads((root / "model_manifest.json").read_text())
    model = root / manifest["model"]["filename"]
    if model.stat().st_size != manifest["model"]["bytes"] or sha256(model) != manifest["model"]["sha256"]:
        raise ValueError("model identity mismatch")
    if manifest["source_checkpoint"]["step"] != 30_000:
        raise ValueError("release does not use the final step-30000 checkpoint")
    with safe_open(model, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        parameters = sum(handle.get_tensor(key).numel() for key in keys)
    if len(keys) != manifest["model"]["tensor_count"] or parameters != 195_953_570:
        raise ValueError("model tensor or parameter count mismatch")
    result = {
        "status": "passed",
        "model_sha256": manifest["model"]["sha256"],
        "model_bytes": manifest["model"]["bytes"],
        "checkpoint_step": 30_000,
        "tensor_count": len(keys),
        "parameters": parameters,
    }
    if args.image is not None:
        from manga_ocr_nar import MangaOCRNar

        ocr = MangaOCRNar(root, device=args.device, precision=args.precision)
        result["smoke_image"] = str(args.image)
        result["prediction"] = ocr.predict(args.image)
        result["device"] = str(ocr.device)
        result["precision"] = str(ocr.dtype)
        result["mps_optimized"] = ocr.mps_optimized
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

