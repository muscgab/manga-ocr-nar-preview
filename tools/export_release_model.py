#!/usr/bin/env python3
"""Export the model state from a full training checkpoint as safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("step", -1)) != 30_000:
        raise ValueError("release checkpoint must be the final step-30000 artifact")
    contract = state.get("contract")
    if not isinstance(contract, dict) or contract.get("query_count") != 129:
        raise ValueError("checkpoint does not contain the Q129 release contract")
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in state["model"].items()}
    output = args.output_dir / "model.safetensors"
    save_file(tensors, str(output), metadata={"format": "pt", "model": "manga-ocr-nar-preview"})
    roundtrip = load_file(str(output), device="cpu")
    if roundtrip.keys() != tensors.keys() or any(not torch.equal(roundtrip[k], tensors[k]) for k in tensors):
        raise RuntimeError("safetensors round-trip mismatch")
    manifest = {
        "schema": "manga-ocr-nar-preview-model-v1",
        "source_checkpoint": {
            "step": int(state["step"]),
            "samples_seen": int(state.get("samples_seen", 0)),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
        },
        "model": {
            "filename": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "parameters": sum(t.numel() for t in tensors.values()),
            "tensor_count": len(tensors),
            "dtype": "float32",
        },
        "contract": contract,
    }
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

