"""Experimental multiscale residual fusion for hehee9/gpt-image-restore.

Drop into repo/experiments/ and run from repo root:
    python experiments/multiscale_restore.py input.png --preset balanced --resampler zscale

This is an optional experiment: it does not change upstream's package or presets.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Running "python experiments/multiscale_restore.py" does not always place the
# repository root on sys.path. Add it only when this is inside an upstream clone.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "gptimg_restore").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np


@dataclass
class MultiscaleOptions:
    scales: tuple[float, ...] = (0.5, 0.25)
    scale_strengths: tuple[float, ...] = (0.6, 0.4)
    resampler: str = "zscale"  # zscale or opencv
    kernel: str = "spline36"  # spline36 or lanczos
    edge_threshold: float = 3.0
    edge_softness: float = 7.0
    max_change_dn: float = 12.0


def validate_options(opts: MultiscaleOptions) -> None:
    if len(opts.scales) != len(opts.scale_strengths):
        raise ValueError("Scales and strengths need the same number of entries")
    if not all(0 < s < 1 for s in opts.scales):
        raise ValueError("Scales must be strictly between 0 and 1")
    if not all(0 <= w <= 1.5 for w in opts.scale_strengths):
        raise ValueError("Strengths must be in [0, 1.5]")
    if opts.resampler not in ("zscale", "opencv"):
        raise ValueError("Unsupported resampler")
    if opts.kernel not in ("spline36", "lanczos"):
        raise ValueError("Unsupported kernel")
    if opts.max_change_dn <= 0:
        raise ValueError("max_change_dn must be positive")


def _zscale_rgb(image: np.ndarray, width: int, height: int, kernel: str) -> np.ndarray:
    """Use FFmpeg/libzimg in planar 16-bit full-range RGB; returns float 0..255."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg with zscale is required for --resampler zscale")
    src = np.rint(np.clip(image, 0, 255) * 257).astype(np.uint16)
    planes = np.moveaxis(src[..., [1, 2, 0]], -1, 0).copy()  # GBR, not RGB!
    h, w = image.shape[:2]
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", f"{w}x{h}",
        "-i", "pipe:0", "-vf", f"zscale=w={width}:h={height}:filter={kernel}:rin=full:r=full",
        "-pix_fmt", "gbrp16le", "-frames:v", "1", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=planes.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg zscale failed: {proc.stderr.decode(errors='replace')[-1000:]}")
    expected = width * height * 3
    samples = np.frombuffer(proc.stdout, dtype="<u2")
    if len(samples) != expected:
        raise RuntimeError(f"zscale produced {len(samples)} samples, expected {expected}")
    return np.moveaxis(samples.reshape(3, height, width)[[2, 0, 1]], 0, -1).astype(np.float32) / 257.0


def resize_rgb(image: np.ndarray, width: int, height: int, opts: MultiscaleOptions) -> np.ndarray:
    if width == image.shape[1] and height == image.shape[0]:
        return np.asarray(image, dtype=np.float32).copy()
    if opts.resampler == "zscale":
        return _zscale_rgb(image, width, height, opts.kernel)
    # OpenCV's Lanczos4 is not exactly zscale's Spline36; fallback is for quick tests only.
    interpol = cv2.INTER_AREA if (width < image.shape[1] and height < image.shape[0]) else cv2.INTER_LANCZOS4
    return cv2.resize(np.asarray(image, np.float32), (width, height), interpolation=interpol)


def protection_mask(image: np.ndarray, opts: MultiscaleOptions) -> np.ndarray:
    """1 means correction is permitted; 0 means preserve edges/textures."""
    smooth = cv2.GaussianBlur(np.asarray(image, np.float32), (0, 0), 0.75)
    lum = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    gradient = cv2.magnitude(gx, gy)
    # Soft anti-edge map, progressively removing weight near high gradients.
    inhibition = np.clip((gradient - opts.edge_threshold) / opts.edge_softness, 0, 1)
    inhibition = cv2.GaussianBlur(inhibition, (0, 0), 1.75)
    return 1 - np.clip(inhibition, 0, 1)


def multiscale_fusion(
    image: np.ndarray,
    scale_denoiser: Callable[[np.ndarray, float], tuple[np.ndarray, dict]],
    opts: MultiscaleOptions,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Iterative low-resolution residual estimation and full-resolution correction.

    The denoiser is run on each *currently restored* downsample. Corrected
    images are NEVER directly upsampled to replace original-resolution detail.
    Corrections are upsampled as differences and protected near high gradients.
    """
    validate_options(opts)
    current = np.asarray(image, dtype=np.float32).copy()
    h, w = current.shape[:2]
    diagnostics: dict[str, np.ndarray] = {}
    reports = []
    if not opts.scales:
        return current, {"scales": []}, diagnostics
    for scale, strength in zip(opts.scales, opts.scale_strengths):
        sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
        if sw >= w or sh >= h or strength == 0:
            reports.append({"scale": scale, "skipped": True})
            continue
        reduced = resize_rgb(current, sw, sh, opts)
        corrected, stage_info = scale_denoiser(reduced, scale)
        corrected = np.asarray(corrected, np.float32)
        if corrected.shape != reduced.shape:
            raise ValueError(f"Denoiser output shape {corrected.shape} != {reduced.shape}")
        # Upsample each 0..255 image separately. Avoids clipping signed residuals.
        estimate = (resize_rgb(reduced, w, h, opts) - resize_rgb(corrected, w, h, opts))
        # Multiscale corrections contain no genuinely new high-frequency detail.
        # Supress interpolation ringing in the correction itself.
        estimate = cv2.GaussianBlur(estimate, (0, 0), max(0.6, (1 / scale) * 0.4))
        keep = protection_mask(current, opts)
        delta = np.clip(estimate * keep[..., None] * strength,
                        -opts.max_change_dn, opts.max_change_dn)
        current = np.clip(current - delta, 0, 255)
        tag = f"{scale:.2f}x"
        diagnostics[f"correction_{tag}"] = delta
        diagnostics[f"protected_{tag}"] = keep
        reports.append({
            "scale": scale,
            "size": [sw, sh],
            "strength": strength,
            "mean_abs_correction_dn": float(np.mean(np.abs(delta))),
            "maximum_abs_correction_dn": float(np.max(np.abs(delta))),
            "stage_reports": stage_info,
        })
    return current, {"scales": reports}, diagnostics


def repo_scale_denoiser(image: np.ndarray, scale: float, backend=None) -> tuple[np.ndarray, dict]:
    """Uses actual upstream modules; requires this script inside upstream checkout."""
    from gptimg_restore import grime, mottle

    out = image.copy()
    report = {}
    if scale >= 0.4:
        out, report["grime"] = grime.apply(out, strength=0.45, backend=backend)
    out, report["mottle"] = mottle.apply(out, strength=0.65 if scale >= 0.4 else 0.45, backend=backend)
    return out, report


def preview_scale_denoiser(image: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    """Synthetic benchmark *only*: bilateral smoother, not upstream gptimg_restore."""
    out = cv2.bilateralFilter(image, 7, 15, 3)
    return out, {"method": "synthetic_bilateral_proxy", "not_upstream": True}


def save_uint8(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), "RGB").save(path)


def run_repo_cli(args: argparse.Namespace, opts: MultiscaleOptions) -> dict:
    from gptimg_restore.io_utils import load_image, save_image
    from gptimg_restore.pipeline import restore_array

    loaded = load_image(args.input)
    backend = None
    if args.device == "gpu":
        from gptimg_restore._gpu import GPUBackend
        backend = GPUBackend()
    base, stage_reports, timings = restore_array(loaded, preset=args.preset,
                                                  bit_depth=16, device=args.device, backend=backend)
    restored, multi_report, diagnostic = multiscale_fusion(
        base, lambda img, scale: repo_scale_denoiser(img, scale, backend=backend), opts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    def save(path: Path, value: np.ndarray) -> None:
        save_image(path, value, loaded.alpha, image_format="png", bit_depth=16,
                   icc_profile=loaded.icc_profile)

    save(outdir / "baseline_16bit.png", base)
    save(outdir / "multiscale_16bit.png", restored)
    diff = 128 + np.clip(restored - base, -16, 16) * 8
    save_uint8(outdir / "difference_x8.png", diff)
    if args.diagnostics:
        for tag, value in diagnostic.items():
            if tag.startswith("correction"):
                save_uint8(outdir / f"{tag}_x8.png", 128 + value * 8)
            else:
                save_uint8(outdir / f"{tag}.png", value[..., None] * np.ones((1,1,3)) * 255)
    analysis = None
    if args.analyze:
        from gptimg_restore import periodic, grime, mottle, dots
        def snapshot(rgb):
            return {"periodic": periodic.analyze(rgb, backend=backend),
                    "grime": grime.analyze(rgb, backend=backend),
                    "mottle": mottle.analyze(rgb, backend=backend),
                    "dots": dots.analyze(rgb)}
        analysis = {"baseline": snapshot(base), "multiscale": snapshot(restored)}
    report = {"source": str(args.input), "preset": args.preset,
              "device": args.device, "analysis": analysis,
              "resampler": opts.resampler, "kernel": opts.kernel,
              "baseline_stage_reports": stage_reports, "baseline_timing": timings,
              "multiscale": multi_report}
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Input image, from an upstream repository checkout")
    p.add_argument("--outdir", default="multiscale_results")
    p.add_argument("--preset", default="balanced", choices=("safe", "balanced", "strong"))
    p.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    p.add_argument("--analyze", action="store_true", help="Run original analyzer on baseline and final results")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.25])
    p.add_argument("--strengths", type=float, nargs="+", default=[0.6, 0.4])
    p.add_argument("--resampler", choices=("zscale", "opencv"), default="zscale")
    p.add_argument("--kernel", choices=("spline36", "lanczos"), default="spline36")
    p.add_argument("--edge-threshold", type=float, default=3.0)
    p.add_argument("--edge-softness", type=float, default=7.0)
    p.add_argument("--max-change", type=float, default=12.0)
    p.add_argument("--diagnostics", action="store_true")
    return p


if __name__ == "__main__":
    ns = _parser().parse_args()
    conf = MultiscaleOptions(scales=tuple(ns.scales), scale_strengths=tuple(ns.strengths),
        resampler=ns.resampler, kernel=ns.kernel, edge_threshold=ns.edge_threshold,
        edge_softness=ns.edge_softness, max_change_dn=ns.max_change)
    try:
        result = run_repo_cli(ns, conf)
        print(json.dumps({"output": str(ns.outdir), "multiscale": result["multiscale"]},
                         indent=2, ensure_ascii=False))
    except ImportError as e:
        sys.exit("Run this script inside a gpt-image-restore checkout (missing package): " + str(e))
