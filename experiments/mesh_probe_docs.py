"""Local diagnostic only: crop docs/media native pixels and compare periodic mesh evidence.

The sample images in docs/media have an explicit no-reuse notice. This script
reads them locally for analysis and writes results ONLY to gitignored
docs/local_media/; it never commits, uploads, or publishes derived images.

Example:
  python experiments/mesh_probe_docs.py --sample portrait --resampler zscale
  python experiments/mesh_probe_docs.py --sample portrait --roi 470 100 420 320

These maps show candidate quasi-periodic textures, NOT proven artifacts.
No restoration or modification of input images is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.multiscale_restore import MultiscaleOptions, resize_rgb

SCALES = (1.0, 0.75, 0.5)


def load_crop(path: Path, roi: tuple[int, int, int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        x, y, w, h = roi
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.width or y + h > image.height:
            raise ValueError(f"ROI {roi} exceeds image dimensions {image.size}")
        # Crop before any resampling; preserve file-native pixel coordinates.
        return np.asarray(image.crop((x, y, x + w, y + h)), dtype=np.uint8)


def _grid_starts(length: int, patch: int, stride: int) -> list[int]:
    starts = list(range(0, length - patch + 1, stride))
    if not starts or starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _periodicity_map(image: np.ndarray, patch: int, stride: int,
                     threshold: float) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Heuristic local FFT: thresholded candidate and unthresholded peak score."""
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY) / 255
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 8.0)
    height, width = gray.shape
    if min(height, width) < patch:
        raise ValueError(f"ROI too small for patch={patch}: {width}x{height}")

    window = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)
    coord = np.arange(patch) - patch // 2
    fy, fx = np.meshgrid(coord, coord, indexing="ij")
    radius = np.hypot(fx, fy)
    # Wavelengths from ~3px to ~patch/3 px at this resolution.
    valid = (radius >= 3) & (radius <= patch / 3) & ((fx > 0) | ((fx == 0) & (fy > 0)))
    accumulation = np.zeros((height, width), dtype=np.float32)
    peak_accumulation = np.zeros_like(accumulation)
    denom = np.zeros_like(accumulation)
    tiles: list[dict] = []

    for y in _grid_starts(height, patch, stride):
        for x in _grid_starts(width, patch, stride):
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2(
                highpass[y:y + patch, x:x + patch] * window))).astype(np.float32)
            local_mean = cv2.GaussianBlur(spectrum, (0, 0), 2.8)
            prominence = spectrum / (local_mean + max(1e-7, float(np.median(local_mean)) * 0.1))
            peaks = cv2.dilate(prominence, np.ones((5, 5), np.uint8))
            ys, xs = np.where(valid & (prominence >= peaks - 1e-6))
            ranked = sorted(zip(ys, xs), key=lambda ij: prominence[ij], reverse=True)
            chosen: list[tuple[float, float]] = []
            for py, px in ranked:
                angle = float(np.arctan2(py - patch // 2, px - patch // 2)) % np.pi
                angular_distance = lambda a, b: abs((a - b + np.pi / 2) % np.pi - np.pi / 2)
                if all(angular_distance(angle, prev) > np.deg2rad(28) for prev, _ in chosen):
                    chosen.append((angle, float(prominence[py, px])))
                if len(chosen) == 2:
                    break
            strength = min(chosen[0][1], chosen[1][1]) if len(chosen) == 2 else 0
            confidence = float(np.clip((strength - threshold) / 1.25, 0, 1))
            accumulation[y:y + patch, x:x + patch] += confidence * window
            peak_accumulation[y:y + patch, x:x + patch] += strength * window
            denom[y:y + patch, x:x + patch] += window
            tiles.append({"x": int(x), "y": int(y),
                          "confidence": round(confidence, 4),
                          "dual_peak_prominence": round(float(strength), 3)})
    confidence = np.divide(accumulation, denom, out=np.zeros_like(accumulation), where=denom > 1e-6)
    peak_score = np.divide(peak_accumulation, denom,
                           out=np.zeros_like(peak_accumulation), where=denom > 1e-6)
    return np.clip(confidence, 0, 1), tiles, peak_score


def _heatmap(mask: np.ndarray) -> np.ndarray:
    heat_bgr = cv2.applyColorMap(np.rint(mask * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path, format="PNG")


def _summary(mask: np.ndarray) -> dict:
    return {"mean_candidate_confidence": round(float(mask.mean()), 4),
            "fraction_above_0.25": round(float((mask > 0.25).mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="portrait", help="docs/showcase.json sample ID")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="native image pixel coordinates; default: showcase.json view")
    parser.add_argument("--resampler", choices=("zscale", "opencv"), default="zscale")
    parser.add_argument("--threshold", type=float, default=2.35,
                        help="Diagnostic peak threshold; lower values raise false positives")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "docs" / "showcase.json").read_text(encoding="utf-8"))
    items = {item["id"]: item for item in manifest["items"]}
    if args.sample not in items:
        parser.error(f"Unknown sample: {args.sample}. Options: {', '.join(items)}")
    item = items[args.sample]
    roi = tuple(args.roi if args.roi is not None else item["view"])
    media_root = ROOT / "docs"
    originals = {}
    for key in ("original", "balanced", "strong"):
        path = media_root / item["images"][key]
        originals[key] = load_crop(path, roi)

    outdir = ROOT / "docs" / "local_media" / f"mesh-{args.sample}"
    outdir.mkdir(parents=True, exist_ok=True)
    for key, image in originals.items():
        _write_image(outdir / f"{key}_roi.png", image)

    original = originals["original"]
    h, w = original.shape[:2]
    opts = MultiscaleOptions(resampler=args.resampler, kernel="spline36")
    maps = {}
    peak_maps = {}
    stats = {}
    for scale in SCALES:
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        scaled = (original.astype(np.float32) if scale == 1 else
                  resize_rgb(original.astype(np.float32), sw, sh, opts))
        # Keep patch size modest at the reduced scales.
        patch = min(128 if scale == 1 else 96, sw, sh)
        patch = patch // 2 * 2
        stride = max(16, patch // 4)
        confidence, tiles, peak_score = _periodicity_map(scaled, patch, stride, args.threshold)
        full = (confidence if scale == 1 else
                cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR))
        maps[scale] = np.clip(full, 0, 1)
        peak_maps[scale] = (peak_score if scale == 1 else
                            cv2.resize(peak_score, (w, h), interpolation=cv2.INTER_LINEAR))
        key = {1.: "1x", .75: "075x", .5: "05x"}[scale]
        _write_image(outdir / f"mesh_{key}.png", _heatmap(maps[scale]))
        # Unthresholded peak evidence stays visible even if candidate detection is empty.
        _write_image(outdir / f"peak_score_{key}.png",
                     _heatmap(np.clip((peak_maps[scale] - 1.8) / 2.0, 0, 1)))
        stats[key] = {"scale": scale, "patch": patch, "stride": stride,
                      **_summary(maps[scale]),
                      "peak_score_p50": round(float(np.percentile(peak_maps[scale], 50)), 3),
                      "peak_score_p95": round(float(np.percentile(peak_maps[scale], 95)), 3),
                      "top_tiles": sorted(tiles, key=lambda v: v["confidence"], reverse=True)[:10]}

    # Cross-scale agreement is diagnostic only; a repeated real texture can agree too.
    agreement = np.sqrt(maps[1.0] * maps[0.5])
    _write_image(outdir / "mesh_agreement.png", _heatmap(agreement))
    heat = _heatmap(agreement).astype(np.float32)
    opacity = (0.6 * agreement)[..., None]
    overlay = np.clip(original.astype(np.float32) * (1 - opacity) + heat * opacity,
                      0, 255).astype(np.uint8)
    _write_image(outdir / "mesh_overlay.png", overlay)

    diff_bal = np.abs(original.astype(np.float32) - originals["balanced"].astype(np.float32))
    diff_str = np.abs(original.astype(np.float32) - originals["strong"].astype(np.float32))
    # Grayscale removal energy: bright means stronger changes, not guaranteed artifact removal.
    for key, diff in (("balanced", diff_bal), ("strong", diff_str)):
        magnitude = np.clip(diff.mean(axis=2) * 12, 0, 255).astype(np.uint8)
        _write_image(outdir / f"removed_by_{key}_x12.png", np.repeat(magnitude[..., None], 3, axis=2))

    cells = [("Original", original), ("Balanced", originals["balanced"]),
             ("Strong", originals["strong"]),
             ("Mesh candidate 1x", _heatmap(maps[1.])),
             ("Mesh candidate 0.75x", _heatmap(maps[.75])),
             ("Mesh candidate 0.5x", _heatmap(maps[.5])),
             ("Cross-scale agreement", _heatmap(agreement)),
             ("Overlay", overlay),
             ("Removed by Balanced x12",
              np.repeat(np.clip(diff_bal.mean(axis=2) * 12, 0, 255).astype(np.uint8)[..., None], 3, axis=2)),
             ("Raw peak score 1x", _heatmap(np.clip((peak_maps[1.] - 1.8) / 2.0, 0, 1))),
             ("Raw peak score 0.75x", _heatmap(np.clip((peak_maps[.75] - 1.8) / 2.0, 0, 1))),
             ("Raw peak score 0.5x", _heatmap(np.clip((peak_maps[.5] - 1.8) / 2.0, 0, 1)))]
    label_height = 27
    sheet = Image.new("RGB", (w * 3, (h + label_height) * 4), "#ededed")
    draw = ImageDraw.Draw(sheet)
    for idx, (name, im) in enumerate(cells):
        xx, yy = (idx % 3) * w, (idx // 3) * (h + label_height)
        draw.text((xx + 8, yy + 6), name, fill="#111111")
        sheet.paste(Image.fromarray(im), (xx, yy + label_height))
    sheet.save(outdir / "comparison.png", format="PNG")

    report = {
        "sample": args.sample, "source": "docs/media native original.webp",
        "roi_native_pixels_xywh": list(roi), "resampler": args.resampler,
        "threshold": args.threshold, "scales": stats,
        "agreement": _summary(agreement),
        "mean_abs_original_vs_balanced_dn": round(float(diff_bal.mean()), 4),
        "mean_abs_original_vs_strong_dn": round(float(diff_str.mean()), 4),
        "interpretation": "Heuristic local FFT periodicity only: unthresholded peak_score maps are diagnostic, and thresholded mesh maps have false positives. Real textures and resampling artifacts may trigger the detector.",
        "rights": "Local diagnostic outputs only; upstream docs/media samples exclude reuse permission."
    }
    (outdir / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved local-only diagnostic maps: {outdir}")
    print(f"ROI: {roi} / {w}x{h}px")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "top_tiles"}
                      for key, value in stats.items()}, indent=2))
    print("Open comparison.png; no image files are committed or uploaded.")


if __name__ == "__main__":
    main()
