"""Native-pixel local FFT wavelength diagnostics for repository demo media.

Diagnostic only, NOT an artifact classifier. All derived images are saved
locally under docs/local_media/, which is excluded from Git.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BANDS = ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0))
SCALES = (1.0, 0.75, 0.5)


def _starts(length, patch, stride):
    values = list(range(0, length - patch + 1, stride))
    if not values or values[-1] != length - patch:
        values.append(length - patch)
    return values


def band_score_map(rgb: np.ndarray, scale: float, orig_band: tuple[float, float],
                   patch: int = 64, stride: int = 12):
    original_h, original_w = rgb.shape[:2]
    rgb = rgb.astype(np.float32)
    if scale != 1:
        rgb = cv2.resize(rgb, (round(original_w * scale), round(original_h * scale)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) / 255.0
    gray = gray - cv2.GaussianBlur(gray, (0, 0), 6)
    h, w = gray.shape
    patch = min(patch, h, w) // 2 * 2
    stride = min(stride, patch // 4)
    per_min, per_max = (p * scale for p in orig_band)
    # The 2-pixel Nyquist limit is insufficient for reliable detection.
    if per_max <= 2.1 or per_min > patch / 2:
        return None, {"valid": False, "min_period_scaled": round(per_min, 2),
                      "max_period_scaled": round(per_max, 2)}
    partial = per_min < 2.1 or per_max > patch / 2
    per_min, per_max = max(2.1, per_min), min(patch / 2, per_max)
    window = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)
    bins = np.fft.fftfreq(patch) * patch
    fx, fy = np.meshgrid(bins, bins)
    rad = np.hypot(fx, fy)
    valid = ((rad >= patch / per_max) & (rad <= patch / per_min) &
             ((fx > 0) | ((fx == 0) & (fy > 0))))
    angles = np.arctan2(fy, fx) % np.pi
    accumulator = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    scores = []
    for y in _starts(h, patch, stride):
        for x in _starts(w, patch, stride):
            block = gray[y:y + patch, x:x + patch] * window
            amp = np.abs(np.fft.fft2(block)).astype(np.float32)
            smooth = cv2.GaussianBlur(amp, (0, 0), 1.8)
            ratio = amp / (smooth + max(float(np.median(smooth)) * 0.2, 1e-6))
            local_max = cv2.dilate(ratio, np.ones((3, 3), np.uint8))
            ys, xs = np.where(valid & (ratio >= local_max - 1e-5))
            ranked = sorted(zip(ys, xs), key=lambda ij: ratio[ij], reverse=True)
            chosen = []
            for py, px in ranked:
                angle = angles[py, px]
                if all(abs((angle - prev + np.pi / 2) % np.pi - np.pi / 2) >
                       np.deg2rad(32) for prev, _ in chosen):
                    chosen.append((angle, float(ratio[py, px])))
                if len(chosen) == 2:
                    break
            strength = min(val for _, val in chosen) if len(chosen) == 2 else 0.0
            scores.append(strength)
            accumulator[y:y + patch, x:x + patch] += strength * window
            weight[y:y + patch, x:x + patch] += window
    sm = np.divide(accumulator, weight, out=np.zeros_like(accumulator),
                   where=weight > 1e-6)
    if scale != 1:
        sm = cv2.resize(sm, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return sm, {"valid": True, "partially_resolved": partial,
                "p50": round(float(np.percentile(scores, 50)), 2),
                "p90": round(float(np.percentile(scores, 90)), 2),
                "p99": round(float(np.percentile(scores, 99)), 2)}


def render(input_path: Path, roi, outdir: Path):
    im = Image.open(input_path).convert("RGB")
    x, y, w, h = roi
    if not (x >= 0 and y >= 0 and w > 0 and h > 0 and
            x + w <= im.width and y + h <= im.height):
        raise ValueError(f"ROI {roi} outside {im.size}")
    native = np.asarray(im.crop((x, y, x + w, y + h)))
    outdir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(native).save(outdir / "original_roi.png")

    def scale_preview(scale):
        if scale == 1:
            return native.copy()
        resized = cv2.resize(native, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

    cells = [(f"Input {scale:g}x (resized back)", scale_preview(scale))
             for scale in SCALES]
    measurements = {}
    for pmin, pmax in BANDS:
        for scale in SCALES:
            key = f"period_{int(pmin)}_{int(pmax)}px_scale_{str(scale).replace('.', '_')}"
            mask, stats = band_score_map(native, scale, (pmin, pmax))
            measurements[key] = stats
            if mask is None:
                view = np.full_like(native, 230)
                label = f"{pmin:g}-{pmax:g}px | {scale:g}x (below Nyquist)"
            else:
                intensity = np.clip((mask - 1.3) / 2.5, 0, 1)
                heat_bgr = cv2.applyColorMap(np.uint8(np.round(255 * intensity)),
                                             cv2.COLORMAP_INFERNO)
                view = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
                label = f"{pmin:g}-{pmax:g}px | {scale:g}x"
                if stats.get("partially_resolved"):
                    label += " (partial)"
            Image.fromarray(view).save(outdir / f"{key}.png")
            cells.append((label, view))

    sheet = Image.new("RGB", (3 * w, 5 * (h + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, arr) in enumerate(cells):
        xx, yy = (i % 3) * w, (i // 3) * (h + 24)
        draw.text((xx + 5, yy + 5), label, fill="black")
        sheet.paste(Image.fromarray(arr), (xx, yy + 24))
    sheet.save(outdir / "frequency_bands.png")
    report = {"input": str(input_path), "roi": roi, "bands": measurements,
              "note": "Diagnostic FFT peakiness, NOT validated mesh classification. "
                      "Hair, eyebrows, and legitimate textures can produce strong peaks. "
                      "OpenCV INTER_AREA is used for downsampling, not Spline36."}
    (outdir / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Generated {outdir / 'frequency_bands.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="portrait")
    p.add_argument("--roi", type=int, nargs=4, default=[555, 100, 330, 280],
                   metavar=("X", "Y", "W", "H"))
    p.add_argument("--input", type=Path, help="Override native docs media source")
    args = p.parse_args()
    source = args.input or ROOT / "docs" / "media" / args.sample / "original.webp"
    render(source, args.roi, ROOT / "docs" / "local_media" / f"frequency-{args.sample}")


if __name__ == "__main__":
    main()
