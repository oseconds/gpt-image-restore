"""Standalone tests and a synthetic benchmark for the fusion/resampling core.

These tests do NOT claim to test upstream restoration filters. For actual restoration,
copy experiments/multiscale_restore.py into a checkout of gpt-image-restore.
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from multiscale_restore import (
    MultiscaleOptions, multiscale_fusion, preview_scale_denoiser,
    protection_mask, resize_rgb, save_uint8,
)


def make_synthetic(width=640, height=416) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2030)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    clean = np.empty((height, width, 3), np.float32)
    clean[..., 0] = 180 + 18 * x / width + 10 * np.sin(y / 90)
    clean[..., 1] = 188 + 13 * x / width + 11 * np.cos(y / 120)
    clean[..., 2] = 192 + 16 * x / width + 8 * np.sin((x + y) / 130)
    cv2.rectangle(clean, (45, 42), (295, 168), (60, 86, 131), thickness=-1)
    cv2.circle(clean, (460, 138), 72, (195, 91, 48), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(clean, (460, 138), 43, (237, 167, 96), thickness=-1, lineType=cv2.LINE_AA)
    cv2.line(clean, (45, 200), (594, 200), (34, 37, 43), thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(clean, "RESTORE 2.6", (65, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 248, 230), 3, cv2.LINE_AA)
    for j in range(260, 375, 5):
        cv2.line(clean, (54, j), (268, j + 9), (88, 111, 130), 1, cv2.LINE_AA)
    clean = np.clip(clean, 0, 255).astype(np.float32)

    # Synthetic mix: periodic crosshatch + sparse hot pixels + RGB-independent mottle.
    periodic = (3.6 * np.sin(2 * np.pi * (x + y) / 9)
                + 2.3 * np.sin(2 * np.pi * (x - y) / 14))
    mottle_small = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32),
                                     (0, 0), 8)
    mottle_small = mottle_small / (mottle_small.std() + 1e-6) * 2.8
    noisy = clean + periodic[..., None] + mottle_small[..., None]
    noisy += rng.normal(0, 1.7, (height, width, 3)).astype(np.float32)
    for yi, xi in zip(rng.integers(5, height-5, 100), rng.integers(5, width-5, 100)):
        noisy[yi-1:yi+2, xi-1:xi+2] += 45
    return clean, np.clip(noisy, 0, 255).astype(np.float32)


def psnr(a, b) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float(10 * np.log10(255 * 255 / max(mse, 1e-20)))


def visual_comparison(clean, noisy, restored, outfile: Path):
    from PIL import ImageDraw
    a = [np.rint(np.clip(img, 0, 255)).astype(np.uint8) for img in (clean, noisy, restored)]
    h, w = clean.shape[:2]
    canvas = Image.new("RGB", (3 * w, h+36), "white")
    for index, (img, title) in enumerate(zip(a, ("REFERENCE (synthetic)", "NOISY INPUT", "MULTISCALE (proxy filter)"))):
        canvas.paste(Image.fromarray(img), (w*index, 36))
        ImageDraw.Draw(canvas).text((w*index+10, 9), title, fill="black")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(outfile)


def run_benchmark(outdir: Path, opts: MultiscaleOptions) -> dict:
    clean, noisy = make_synthetic()
    restored, report, diag = multiscale_fusion(noisy, preview_scale_denoiser, opts)
    img_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gradient_x, gradient_y) > 35
    flat = ~cv2.dilate(edge.astype(np.uint8), np.ones((7,7), np.uint8)).astype(bool)
    def mae(a, b, mask):
        return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
    metrics = {
        "experiment": "synthetic_data_proving_plumbing_not_upstream_quality",
        "baseline_psnr_db": psnr(clean, noisy),
        "multiscale_psnr_db": psnr(clean, restored),
        "psnr_change_db": psnr(clean, restored) - psnr(clean, noisy),
        "flat_mae_before": mae(noisy,clean,flat),
        "flat_mae_after": mae(restored,clean,flat),
        "edge_mae_before": mae(noisy,clean,edge),
        "edge_mae_after": mae(restored,clean,edge),
        "upstream_filters_used": False,
        "resampling": opts.resampler,
        "kernel": opts.kernel,
        "pipeline": report,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    save_uint8(outdir / "synthetic_reference.png", clean)
    save_uint8(outdir / "synthetic_noisy.png", noisy)
    save_uint8(outdir / "synthetic_multiscale.png", restored)
    visual_comparison(clean, noisy, restored, outdir / "comparison.png")
    for tag, d in diag.items():
        if tag.startswith("correction"):
            save_uint8(outdir / f"{tag}.png", 128 + d * 8)
    (outdir / "benchmark.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


class SmokeTests(unittest.TestCase):
    def test_constant_16bit_zscale(self):
        opts = MultiscaleOptions(scales=(0.5,), scale_strengths=(0.5,), resampler="zscale")
        rgb = np.full((45, 57, 3), 130, np.float32)
        small = resize_rgb(rgb, 28, 22, opts)
        self.assertEqual(small.shape, (22, 28, 3))
        self.assertLess(float(np.max(abs(small-130))), 0.02)

    def test_identity(self):
        rng = np.random.default_rng(2)
        original = rng.integers(3, 252, (62, 75, 3)).astype(np.float32)
        opts = MultiscaleOptions(resampler="opencv")
        out, report, _ = multiscale_fusion(original, lambda x, s: (x.copy(), {}), opts)
        self.assertTrue(np.array_equal(original, out))
        self.assertEqual(len(report["scales"]), 2)

    def test_sharp_edge_protection(self):
        rgb = np.full((96, 96, 3), 110, dtype=np.float32)
        rgb[:, 48:] = 245
        opts = MultiscaleOptions(edge_threshold=3.0, edge_softness=7.0)
        m = protection_mask(rgb, opts)
        self.assertGreater(float(m[30, 20]), 0.95)
        self.assertLess(float(m[30, 48]), 0.5)

    def test_options_validation(self):
        rgb = np.full((100, 100, 3), 100, dtype=np.float32)
        with self.assertRaises(ValueError):
            multiscale_fusion(rgb, lambda x, s: (x, {}), MultiscaleOptions(scales=(0.5,), scale_strengths=(0.5, 0.1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--resampler", choices=["zscale", "opencv"], default="zscale")
    args, remaining = parser.parse_known_args()
    if args.demo:
        result = run_benchmark(Path(args.outdir), MultiscaleOptions(resampler=args.resampler))
        print(json.dumps(result, indent=2))
    else:
        unittest.main(argv=[sys.argv[0]]+remaining)
