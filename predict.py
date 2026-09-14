#!/usr/bin/env python3
"""Recognize one or more cropped manga text images."""

from __future__ import annotations

import argparse
from pathlib import Path

from manga_ocr_nar import MangaOCRNar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--portable-mps", action="store_true", help="disable the batch-1 Metal path")
    args = parser.parse_args()
    ocr = MangaOCRNar(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        optimized_mps=not args.portable_mps,
    )
    for path, text in zip(args.images, ocr.predict_all(args.images), strict=True):
        print(f"{path}\t{text}")


if __name__ == "__main__":
    main()

