"""Measure specific FFT peaks in the *native* docs/media samples.

This is a diagnostic, NOT a validated GPT-image mesh detector.
Output lives only in gitignored docs/local_media/; never publish derivatives
of the upstream docs/media examples.

Usage:
  python experiments/mesh_peak_probe.py --sample portrait
  open docs/local_media/mesh-peaks-portrait/comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PEAKS = (
    ("diagonal_16_-16", 16, -16),
    ("opposite_16_16", 16, 16),
    ("axis_31_0", 31, 0),
    ("axis_0_31", 0, 31),
)
PATCH = 64


def native_crop(path: Path, roi: tuple[int, int, int, int]) -> np.ndarray:
    with Image.open(path) as im:
        x, y, w, h = roi
        if x < 0 or y < 0 or w < PATCH or h < PATCH or x+w > im.width or y+h > im.height:
            raise ValueError(f"Invalid ROI {roi}, image dimensions {im.size}")
        return np.asarray(im.convert("RGB").crop((x, y, x+w, y+h)), np.uint8)


def starts(length: int, stride: int) -> list[int]:
    values = list(range(0, length-PATCH+1, stride))
    if values[-1] != length-PATCH:
        values.append(length-PATCH)
    return values


def peak_metrics(spectrum: np.ndarray, fx: int, fy: int) -> tuple[float, float, float]:
    """Relative narrowband prominence and absolute amplitude; no classification."""
    i, j = fy % PATCH, fx % PATCH
    ys = np.arange(i-3, i+4) % PATCH
    xs = np.arange(j-3, j+4) % PATCH
    neighborhood = spectrum[np.ix_(ys, xs)]
    surround = np.ones((7, 7), bool)
    surround[2:5, 2:5] = False
    baseline = float(np.median(neighborhood[surround]))
    amplitude = float(spectrum[i, j])
    return amplitude / (baseline + 1e-7), amplitude, baseline


def maps_for_image(rgb: np.ndarray, stride: int) -> tuple[dict, dict]:
    height, width = rgb.shape[:2]
    lum = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY) / 255.
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    total_weight = np.zeros((height, width), np.float32)
    accumulation = {name: np.zeros((height, width), np.float32) for name, _, _ in PEAKS}
    amp_acc = {name: 0.0 for name, _, _ in PEAKS}
    bg_acc = {name: 0.0 for name, _, _ in PEAKS}
    count = 0
    for y in starts(height, stride):
        for x in starts(width, stride):
            tile = lum[y:y+PATCH, x:x+PATCH]
            detail = tile - cv2.GaussianBlur(tile, (0, 0), 6)
            spectrum = np.abs(np.fft.fft2(detail * window)).astype(np.float32)
            for name, fx, fy in PEAKS:
                score, amplitude, baseline = peak_metrics(spectrum, fx, fy)
                accumulation[name][y:y+PATCH, x:x+PATCH] += score * window
                amp_acc[name] += amplitude
                bg_acc[name] += baseline
            total_weight[y:y+PATCH, x:x+PATCH] += window
            count += 1
    maps = {name: np.divide(acc, total_weight, out=np.zeros_like(acc),
                            where=total_weight > 1e-6)
            for name, acc in accumulation.items()}
    stats = {name: {"median_prominence": round(float(np.median(m[total_weight > 1e-6])), 3),
                    "p90_prominence": round(float(np.percentile(m[total_weight > 1e-6], 90)), 3),
                    "mean_amplitude": round(amp_acc[name] / count, 5),
                    "mean_local_baseline": round(bg_acc[name] / count, 5)}
             for name, m in maps.items()}
    return maps, stats


def heatmap(values: np.ndarray, upper: float = 12.) -> np.ndarray:
    # Identical fixed scale for all presets, never independent normalization.
    level = np.uint8(np.clip((values - 1.) / upper, 0., 1.) * 255)
    return cv2.cvtColor(cv2.applyColorMap(level, cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="portrait")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                   default=[555, 100, 330, 280])
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()
    if args.stride < 1 or args.stride > 32:
        p.error("--stride must be between 1 and 32")
    roi = tuple(args.roi)
    manifest = json.loads((ROOT / "docs" / "showcase.json").read_text())
    items = {entry["id"]: entry for entry in manifest["items"]}
    if args.sample not in items:
        p.error(f"Unknown sample: {args.sample}")
    item = items[args.sample]
    originals = {
        key: native_crop(ROOT / "docs" / item["images"][key], roi)
        for key in ("original", "balanced", "strong")
    }
    maps = {}
    stats = {}
    for key, rgb in originals.items():
        maps[key], stats[key] = maps_for_image(rgb, args.stride)

    outdir = ROOT / "docs" / "local_media" / f"mesh-peaks-{args.sample}"
    outdir.mkdir(parents=True, exist_ok=True)
    # Columns are image presets; rows are identical frequency coordinates.
    rows = [
        ("Native crop", lambda k: originals[k]),
        ("Diagonal (16,-16)", lambda k: heatmap(maps[k]["diagonal_16_-16"])),
        ("Opposite (16,16) control", lambda k: heatmap(maps[k]["opposite_16_16"])),
        ("~2px x axis (31,0)", lambda k: heatmap(maps[k]["axis_31_0"], upper=18)),
        ("~2px y axis (0,31)", lambda k: heatmap(maps[k]["axis_0_31"], upper=18)),
        ("Diagonal minus control", lambda k: heatmap(
            np.clip(maps[k]["diagonal_16_-16"] -
                    maps[k]["opposite_16_16"], 0, None), upper=8)),
    ]
    x0, y0, w, h = roi
    label_h = 28
    sheet = Image.new("RGB", (3*w, len(rows)*(h+label_h)), "#ececec")
    draw = ImageDraw.Draw(sheet)
    for row, (title, get_image) in enumerate(rows):
        for col, key in enumerate(originals):
            px, py = col*w, row*(h+label_h)
            draw.text((px+5, py+6), f"{key} | {title}", fill="black")
            tile = get_image(key)
            sheet.paste(Image.fromarray(tile), (px, py+label_h))
    sheet.save(outdir / "comparison.png")
    report = {
        "source": "native docs/media samples, no upscaling",
        "roi_xywh": list(roi), "patch": PATCH, "stride": args.stride,
        "peaks": {name: {"fx_bin": fx, "fy_bin": fy} for name, fx, fy in PEAKS},
        "presets": stats,
        "important": "High spectral prominence does not prove an artifact or "
                     "an increase in absolute intensity. Compare mean_amplitude "
                     "AND mean_local_baseline. WebP may itself introduce structure.",
        "rights": "Local-only diagnostics; do not redistribute upstream sample derivatives."
    }
    (outdir / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Comparison: {outdir / 'comparison.png'}")
    print(f"Stats: {outdir / 'stats.json'}")


if __name__ == "__main__":
    main()
