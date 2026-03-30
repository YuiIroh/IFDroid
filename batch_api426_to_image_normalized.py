#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch convert each APK folder's api426_centrality_max.csv into a normalized 42x42 grayscale image.

Input layout:
INPUT_ROOT/
    apk_a/
        api426_centrality_max.csv
    apk_b/
        api426_centrality_max.csv
    ...

Output layout:
OUTPUT_ROOT/
    apk_a/
        api426_matrix_raw.csv
        api426_matrix_normalized.csv
        api426_flat_vector_normalized.csv
        image_42x42_normalized_float.csv
        image_42x42_normalized_uint8.csv
        image_42x42_normalized.png
        image_preview_420x420.png
        image_summary.json
    apk_b/
        ...
    batch_image_summary.csv

Normalization strategy:
- degree_centrality: per-column min-max
- katz_twohop_centrality: log1p -> per-column min-max
- closeness_centrality: per-column min-max
- harmonic_centrality: log1p -> per-column min-max

Then:
426x4 -> flatten(1704) -> pad 60 zeros -> reshape(42x42) -> *255 -> uint8 PNG
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# Hard-coded paths
# =========================================================
INPUT_ROOT = r"E:\Text to image\IFDroid\APK_sensitive426_output"
OUTPUT_ROOT = r"E:\Text to image\IFDroid\APK_image_output_normalized"

INPUT_FILENAME = "api426_centrality_max.csv"

CENTRALITY_COLS = [
    "degree_centrality",
    "katz_twohop_centrality",
    "closeness_centrality",
    "harmonic_centrality",
]

EXPECTED_API_COUNT = 426
PAD_LENGTH = 60
IMAGE_SIZE = 42
PREVIEW_SCALE = 10


# =========================================================
# Helpers
# =========================================================
def minmax_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - xmin) / (xmax - xmin)


def normalize_426x4_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_df.index)

    # degree: direct min-max
    out["degree_centrality"] = minmax_01(raw_df["degree_centrality"].to_numpy(dtype=np.float64))

    # katz: log1p + min-max
    katz = np.log1p(np.maximum(raw_df["katz_twohop_centrality"].to_numpy(dtype=np.float64), 0.0))
    out["katz_twohop_centrality"] = minmax_01(katz)

    # closeness: direct min-max
    out["closeness_centrality"] = minmax_01(raw_df["closeness_centrality"].to_numpy(dtype=np.float64))

    # harmonic: log1p + min-max
    harmonic = np.log1p(np.maximum(raw_df["harmonic_centrality"].to_numpy(dtype=np.float64), 0.0))
    out["harmonic_centrality"] = minmax_01(harmonic)

    return out


def process_one(sample_dir: Path, output_root: Path) -> Optional[Dict[str, object]]:
    in_csv = sample_dir / INPUT_FILENAME
    if not in_csv.exists():
        return None

    out_dir = output_root / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    missing = [c for c in CENTRALITY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != EXPECTED_API_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_API_COUNT} rows in {in_csv.name}, got {len(df)}"
        )

    # Raw 426x4
    raw_matrix_df = df[CENTRALITY_COLS].copy()
    raw_matrix_df.to_csv(out_dir / "api426_matrix_raw.csv", index=False, encoding="utf-8-sig")

    # Normalized 426x4
    norm_matrix_df = normalize_426x4_matrix(raw_matrix_df)
    norm_matrix_df.to_csv(out_dir / "api426_matrix_normalized.csv", index=False, encoding="utf-8-sig")

    # Flatten normalized matrix
    flat = norm_matrix_df.to_numpy(dtype=np.float64).reshape(-1)
    pd.DataFrame({"value": flat}).to_csv(
        out_dir / "api426_flat_vector_normalized.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Pad 60 zeros
    padded = np.concatenate([flat, np.zeros(PAD_LENGTH, dtype=np.float64)])

    # Reshape 42x42
    img_float = padded.reshape(IMAGE_SIZE, IMAGE_SIZE)
    pd.DataFrame(img_float).to_csv(
        out_dir / "image_42x42_normalized_float.csv",
        index=False,
        header=False,
        encoding="utf-8-sig"
    )

    # Convert to uint8 grayscale
    img_uint8 = np.rint(img_float * 255.0).clip(0, 255).astype(np.uint8)
    pd.DataFrame(img_uint8).to_csv(
        out_dir / "image_42x42_normalized_uint8.csv",
        index=False,
        header=False,
        encoding="utf-8-sig"
    )

    # Save PNG
    img = Image.fromarray(img_uint8, mode="L")
    img.save(out_dir / "image_42x42_normalized.png")

    # Save enlarged preview
    preview = img.resize(
        (IMAGE_SIZE * PREVIEW_SCALE, IMAGE_SIZE * PREVIEW_SCALE),
        resample=Image.NEAREST
    )
    preview.save(out_dir / f"image_preview_{IMAGE_SIZE * PREVIEW_SCALE}x{IMAGE_SIZE * PREVIEW_SCALE}.png")

    summary = {
        "apk_folder": sample_dir.name,
        "input_csv": str(in_csv),
        "row_count": int(len(df)),
        "centrality_cols": CENTRALITY_COLS,
        "normalization": {
            "degree_centrality": "minmax",
            "katz_twohop_centrality": "log1p_then_minmax",
            "closeness_centrality": "minmax",
            "harmonic_centrality": "log1p_then_minmax",
        },
        "matrix_shape_before_pad": [EXPECTED_API_COUNT, 4],
        "flat_length": int(len(flat)),
        "pad_length": PAD_LENGTH,
        "final_shape": [IMAGE_SIZE, IMAGE_SIZE],
        "output_raw_matrix_csv": str(out_dir / "api426_matrix_raw.csv"),
        "output_normalized_matrix_csv": str(out_dir / "api426_matrix_normalized.csv"),
        "output_float_csv": str(out_dir / "image_42x42_normalized_float.csv"),
        "output_uint8_csv": str(out_dir / "image_42x42_normalized_uint8.csv"),
        "output_png": str(out_dir / "image_42x42_normalized.png"),
        "output_preview_png": str(out_dir / f"image_preview_{IMAGE_SIZE * PREVIEW_SCALE}x{IMAGE_SIZE * PREVIEW_SCALE}.png"),
        "raw_min": {
            "degree_centrality": float(raw_matrix_df["degree_centrality"].min()),
            "katz_twohop_centrality": float(raw_matrix_df["katz_twohop_centrality"].min()),
            "closeness_centrality": float(raw_matrix_df["closeness_centrality"].min()),
            "harmonic_centrality": float(raw_matrix_df["harmonic_centrality"].min()),
        },
        "raw_max": {
            "degree_centrality": float(raw_matrix_df["degree_centrality"].max()),
            "katz_twohop_centrality": float(raw_matrix_df["katz_twohop_centrality"].max()),
            "closeness_centrality": float(raw_matrix_df["closeness_centrality"].max()),
            "harmonic_centrality": float(raw_matrix_df["harmonic_centrality"].max()),
        },
    }

    (out_dir / "image_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return summary


def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {input_root}")

    sample_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: p.name.lower())

    rows = []
    for sample_dir in sample_dirs:
        try:
            row = process_one(sample_dir, output_root)
            if row is None:
                print(f"[SKIP] {sample_dir.name}: missing {INPUT_FILENAME}")
            else:
                rows.append(row)
                print(f"[OK] {sample_dir.name}")
        except Exception as e:
            print(f"[ERROR] {sample_dir.name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(
            output_root / "batch_image_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print(f"\nDone. Batch summary saved to: {output_root / 'batch_image_summary.csv'}")
    else:
        print("\nNo valid APK folders were processed.")


if __name__ == "__main__":
    main()
