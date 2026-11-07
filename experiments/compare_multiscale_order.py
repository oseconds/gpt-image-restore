"""Compare Spline36 multiscale BEFORE vs AFTER the existing Balanced preset.

Produces local-only images for native docs/media samples. The upstream demo
images are explicitly excluded from reuse permission: do not commit or publish
their derivatives. The experiment doesn't modify any shipped restoration code.

From repository root:
    python experiments/compare_multiscale_order.py
    python experiments/compare_multiscale_order.py --sample portrait --roi 555 100 330 280
    open docs/local_media/order-portrait/comparison.png
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multiscale_restore import (
    MultiscaleOptions, multiscale_fusion, repo_scale_denoiser,
)
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import restore_array


def _roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    height, width = image.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI {roi} is outside {width}x{height}")
    return image[y:y + h, x:x + w]


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    magnitude = diff.mean(axis=2)
    return {
        "mean_abs_change_dn": round(float(diff.mean()), 6),
        "max_abs_change_dn": round(float(diff.max()), 4),
        "fraction_pixels_mean_change_ge_0_5dn": round(float((magnitude >= 0.5).mean()), 6),
        "fraction_pixels_mean_change_ge_1dn": round(float((magnitude >= 1).mean()), 6),
    }


def _comparison(original: np.ndarray, baseline: np.ndarray,
                pre: np.ndarray, post: np.ndarray,
                roi: tuple[int, int, int, int], path: Path) -> None:
    variants = [
        ("Original", original),
        ("Balanced", baseline),
        ("Pre: Multiscale > Balanced", pre),
        ("Post: Balanced > Multiscale", post),
    ]
    crops = [_roi(rgb, roi) for _, rgb in variants]
    h, w = crops[0].shape[:2]
    label_h = 28
    # Second row: differences against identical Balanced baseline, shown at x12.
    differences = [
        ("Original - Balanced (abs x12)", np.abs(crops[0] - crops[1])),
        ("Pre - Balanced (abs x12)", np.abs(crops[2] - crops[1])),
        ("Post - Balanced (abs x12)", np.abs(crops[3] - crops[1])),
        ("Pre - Post (abs x12)", np.abs(crops[2] - crops[3])),
    ]
    canvas = Image.new("RGB", (4 * w, 2 * (h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, items in enumerate((variants, differences)):
        for col, (title, val) in enumerate(items):
            left, top = col * w, row * (h + label_h)
            draw.text((left + 6, top + 7), title, fill="black")
            pixels = crops[col] if row == 0 else np.clip(val * 12, 0, 255)
            canvas.paste(Image.fromarray(_u8(pixels)), (left, top + label_h))
    canvas.save(path)


def run_one(item: dict, opts: MultiscaleOptions,
            roi_override: tuple[int, int, int, int] | None) -> Path:
    sample_id = item["id"]
    input_path = ROOT / "docs" / item["images"]["original"]
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    loaded = load_image(input_path)
    target = ROOT / "docs" / "local_media" / f"order-{sample_id}"
    target.mkdir(parents=True, exist_ok=True)
    original = loaded.rgb.copy()
    roi = tuple(roi_override) if roi_override else tuple(item["view"])

    print(f"[{sample_id}] Balanced baseline...", flush=True)
    t0 = time.perf_counter()
    baseline, baseline_stages, _ = restore_array(
        loaded, preset="balanced", bit_depth=16)
    t_baseline = time.perf_counter() - t0

    print(f"[{sample_id}] Pre: multiscale(original) then Balanced...", flush=True)
    t0 = time.perf_counter()
    prefiltered, pre_multiscale_report, _ = multiscale_fusion(
        original, repo_scale_denoiser, opts)
    # Retain floating-point prefilter output instead of rounding it to uint8.
    # restore_array treats a 16-bit LoadedImage as already decoded float RGB.
    preloaded = replace(loaded, rgb=prefiltered, bit_depth=16)
    pre, pre_stages, _ = restore_array(preloaded, preset="balanced", bit_depth=16)
    t_pre = time.perf_counter() - t0

    print(f"[{sample_id}] Post: Balanced then multiscale(baseline)...", flush=True)
    t0 = time.perf_counter()
    post, post_multiscale_report, _ = multiscale_fusion(
        baseline, repo_scale_denoiser, opts)
    t_post = time.perf_counter() - t0

    images = {"original": original, "balanced": baseline,
              "pre": pre, "post": post}
    for key, rgb in images.items():
        save_image(target / f"{key}.png", rgb, loaded.alpha,
                   image_format="png", bit_depth=8, icc_profile=loaded.icc_profile)
    _comparison(original, baseline, pre, post, roi, target / "comparison.png")

    report = {
        "sample": sample_id,
        "input": str(input_path.relative_to(ROOT)),
        "roi_native_xywh": list(roi),
        "resampler": opts.resampler,
        "kernel": opts.kernel,
        "scales": list(opts.scales),
        "strengths": list(opts.scale_strengths),
        "variants": {
            "baseline": "original -> balanced",
            "pre": "original -> multiscale -> balanced",
            "post": "original -> balanced -> multiscale",
        },
        "timing_seconds": {
            "baseline": round(t_baseline, 2),
            "pre_total": round(t_pre, 2),
            "post_extra_after_baseline": round(t_post, 2),
        },
        "difference_vs_baseline": {
            "pre": _stats(pre, baseline),
            "post": _stats(post, baseline),
            "pre_vs_post": _stats(pre, post),
        },
        "stage_reports": {
            "balanced_baseline": baseline_stages,
            "balanced_after_prefilter": pre_stages,
            "multiscale_before_balanced": pre_multiscale_report,
            "multiscale_after_balanced": post_multiscale_report,
        },
        "notes": [
            "The multiscale stage uses the SAME upstream grime/mottle modules in both orders.",
            "Downsampled RGB is used for residual estimation, not to replace full-res RGB.",
            "Difference magnitude is not a measure of quality or artifact removal.",
            "Upstream docs/media examples and derived images may not be redistributed.",
        ],
    }
    (target / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[{sample_id}] {target / 'comparison.png'}", flush=True)
    print(json.dumps(report["difference_vs_baseline"], ensure_ascii=False,
                     indent=2), flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs="+", default=["portrait", "teapot"],
                        help="IDs from docs/showcase.json (defaults: portrait teapot)")
    parser.add_argument("--roi", nargs=4, type=int,
                        metavar=("X", "Y", "W", "H"),
                        help="Override native pixel crop; use for one sample")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.75, 0.5])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.6, 0.4])
    parser.add_argument("--resampler", choices=["zscale", "opencv"], default="zscale")
    args = parser.parse_args()
    if args.roi is not None and len(args.sample) != 1:
        parser.error("Specify one --sample when using --roi")
    if len(args.scales) != len(args.strengths):
        parser.error("--scales and --strengths must have the same length")
    opts = MultiscaleOptions(
        scales=tuple(args.scales), scale_strengths=tuple(args.strengths),
        resampler=args.resampler, kernel="spline36",
    )
    showcase = json.loads((ROOT / "docs" / "showcase.json").read_text(encoding="utf-8"))
    items = {item["id"]: item for item in showcase["items"]}
    for sample in args.sample:
        if sample not in items:
            parser.error(f"Unknown sample {sample!r}; available: {list(items)}")
        run_one(items[sample], opts, tuple(args.roi) if args.roi else None)
    print("All images are local-only and gitignored. Do not publish derivatives.")


if __name__ == "__main__":
    main()
