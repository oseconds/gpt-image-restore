"""Add the existing periodic module to 2x Delta 0.50 and inspect with Contrast Boost.

Reuses the already-rendered (8-bit PNG) result from sweep_resampler_paths.py
so Balanced at 2x is NOT run again. This tests a quick independent module
combination: Original vs Balanced vs 2x Delta 0.50 vs Delta + Periodic.

Contrast Boost follows docs/site.js: histogram of V=max(R,G,B) from original,
the same piecewise slope-constrained LUT for *all* variants, and grayscale
output. It is applied to VIEWING ONLY, not restoration or metrics.

The upstream docs/media samples exclude permission for redistribution. All
generated sample derivatives remain local in ignored docs/local_media/.

Usage:
  python experiments/compare_delta_periodic.py --sample teapot
  open docs/local_media/delta-periodic-teapot/latest_boost.png
  open docs/local_media/delta-periodic-teapot/latest_detail_boost.png
  python experiments/compare_delta_periodic.py --sample portrait
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptimg_restore import periodic
from gptimg_restore.io_utils import save_image

NAMES = ("Original", "Balanced", "2x Delta 0.50", "2x Delta 0.50 + Periodic")

# Source-image native pixel ROIs. None of these are input-processing crops.
TEAPOT_ROIS = {
    "lid": (520, 185, 550, 265),
    "spout_join": (265, 310, 350, 350),
    "handle": (985, 330, 385, 465),
    "floor": (160, 765, 720, 245),
}
PORTRAIT_ROIS = {
    "forehead": (555, 100, 330, 280),
}


def _rgb_file(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def _uint8(data: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(data, 0, 255)).astype(np.uint8)


def _load_sweep(sample: str, selected: Path | None) -> Path:
    if selected is not None:
        folder = selected.expanduser().resolve()
    else:
        parent = ROOT / "docs" / "local_media" / f"resampler-sweep-{sample}"
        if not parent.exists():
            raise FileNotFoundError(
                f"No saved sweep in {parent}. Run first:\n"
                f"  python experiments/sweep_resampler_paths.py --sample {sample}")
        required = ("original.png", "balanced.png", "2x_delta_050.png")
        candidates = [p for p in parent.iterdir() if p.is_dir() and
                      all((p / name).is_file() for name in required)]
        if not candidates:
            raise FileNotFoundError(f"No sweep containing {required} under {parent}")
        folder = sorted(candidates, key=lambda p: p.name)[-1]
    for filename in ("original.png", "balanced.png", "2x_delta_050.png"):
        if not (folder / filename).is_file():
            raise FileNotFoundError(folder / filename)
    return folder


def _boost_lut(rgb: np.ndarray) -> np.ndarray:
    """Match docs/site.js _boostCurve (no sample-specific BOOST_BANDS for teapot)."""
    values = _uint8(rgb).max(axis=2)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    tail = values.size * .2 / 100.0  # BOOST_TAIL_PERCENT = 0.2
    lower_cumulative = np.cumsum(hist)
    upper_cumulative = np.cumsum(hist[::-1])
    low = int(np.searchsorted(lower_cumulative, tail, side="right"))
    high = 255 - int(np.searchsorted(upper_cumulative, tail, side="right"))
    if low > high:
        raise ValueError("Cannot construct contrast curve for this image")
    sigma = 3.0  # BOOST_SMOOTH_SIGMA
    offsets = np.arange(-9, 10)
    smooth_weights = np.exp(-offsets * offsets / (2*sigma*sigma))
    # JS uses histogram[clamp(value+offset, 0, 255)] at every offset.
    domain = np.arange(256)
    density = np.zeros(256, np.float64)
    for off, weight in zip(offsets, smooth_weights):
        density += hist[np.clip(domain + int(off), 0, 255)] * weight

    floors = np.full(256, .3, np.float64)  # BOOST_MIN_GAIN
    def slopes(scale: float) -> np.ndarray:
        out = np.zeros(256, np.float64)
        out[low:high+1] = np.clip(scale*density[low:high+1],
                                 floors[low:high+1], 4.0)  # BOOST_MAX_GAIN
        return out

    lower, upper = 0.0, 1.0
    while float(slopes(upper).sum()) < 255 and upper < 1e12:
        upper *= 2
    for _ in range(60):
        middle = (lower + upper) / 2
        if float(slopes(middle).sum()) < 255:
            lower = middle
        else:
            upper = middle
    gradients = slopes(upper)
    shift = max(0.0, (255.0 - float(gradients.sum())) / 2)
    curve = shift + np.cumsum(gradients) - gradients/2
    # Browser Uint8ClampedArray tie handling differs only at exact .5.
    return np.rint(np.clip(curve, 0, 255)).astype(np.uint8)


def _boost_view(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    values = _uint8(rgb).max(axis=2)
    gray = lut[values]
    return np.repeat(gray[:, :, None], 3, axis=2)


def _crop(arr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    ih, iw = arr.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > iw or y+h > ih:
        raise ValueError(f"Invalid ROI {roi} for image {iw}x{ih}")
    return arr[y:y+h, x:x+w]


def _sheet(images: dict[str, np.ndarray], outfile: Path, *,
           roi: tuple[int,int,int,int] | None = None, lut: np.ndarray | None = None,
           max_width: int | None = None) -> None:
    views = {}
    for name, data in images.items():
        view = _crop(data, roi) if roi is not None else data
        if lut is not None:
            view = _boost_view(view, lut)
        else:
            view = _uint8(view)
        if max_width is not None and view.shape[1] > max_width:
            ratio = max_width/view.shape[1]
            view = cv2.resize(view, (max_width, round(view.shape[0]*ratio)),
                              interpolation=cv2.INTER_AREA)
        views[name] = view
    w, h = next(iter(views.values())).shape[1], next(iter(views.values())).shape[0]
    label = 30
    sheet = Image.new("RGB", (w*4, h+label), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, view) in enumerate(views.items()):
        xx = index*w
        draw.text((xx+6, 7), name, fill="black")
        sheet.paste(Image.fromarray(view, "RGB"), (xx, label))
    sheet.save(outfile)


def _diff_panel(images: dict[str, np.ndarray], outfile: Path, *,
                roi: tuple[int, int, int, int] | None = None) -> None:
    original, balanced, delta, combo = images.values()
    comparisons = {
        "Original - Balanced (abs x12)": np.abs(original - balanced),
        "Original - Delta 0.50 (abs x12)": np.abs(original - delta),
        "Delta 0.50 - Combo (abs x12)": np.abs(delta - combo),
        "Balanced - Combo (abs x12)": np.abs(balanced - combo),
    }
    amplified = {name: np.clip(v * 12, 0, 255)
                 for name, v in comparisons.items()}
    _sheet(amplified, outfile, roi=roi)


def _rms_highpass(rgb: np.ndarray, sigma: float = 1.15) -> float:
    gray = cv2.cvtColor(np.asarray(rgb, np.float32), cv2.COLOR_RGB2GRAY)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
    return float(np.sqrt(np.mean(detail*detail)))


def _difference(a: np.ndarray, b: np.ndarray) -> dict:
    abs_diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "mean_absolute_change_dn": round(float(np.mean(abs_diff)), 5),
        "maximum_change_dn": round(float(abs_diff.max()), 4),
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("teapot","portrait"), default="teapot")
    parser.add_argument("--sweep-folder", type=Path,
                        help="Explicit saved sweep directory; default: newest matching run")
    parser.add_argument("--periodic-strength", type=float, default=1.0)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                        help="Optional native-pixel crop for an extra comparison sheet")
    args = parser.parse_args()
    if not 0 <= args.periodic_strength <= 2:
        parser.error("--periodic-strength must be within 0..2")
    sweep = _load_sweep(args.sample, args.sweep_folder)
    originals = [_rgb_file(sweep / filename)
                 for filename in ("original.png", "balanced.png", "2x_delta_050.png")]
    if not all(img.shape == originals[0].shape for img in originals):
        raise ValueError("Saved sweep images have mismatched shapes")
    original, baseline, delta = originals

    print(f"Using existing sweep: {sweep}", flush=True)
    print("Applying only the existing native-resolution periodic module...", flush=True)
    periodic_result, stage_report = periodic.apply(delta.copy(),
                                                   strength=args.periodic_strength)
    periodic_result = np.clip(periodic_result, 0, 255)
    variants = dict(zip(NAMES, (original, baseline, delta, periodic_result)))
    lut = _boost_lut(original)

    parent = ROOT / "docs" / "local_media" / f"delta-periodic-{args.sample}"
    parent.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = parent / date_tag
    number = 1
    while outdir.exists():
        outdir = parent / f"{date_tag}-{number}"
        number += 1
    outdir.mkdir()

    # All filenames follow the same fixed position/order; no overwritten runs.
    _sheet(variants, outdir / "normal.png", max_width=768)
    _sheet(variants, outdir / "boost.png", lut=lut, max_width=768)
    _diff_panel(variants, outdir / "difference_x12.png")
    for label, region in (TEAPOT_ROIS if args.sample=="teapot" else PORTRAIT_ROIS).items():
        _sheet(variants, outdir / f"{label}_normal.png", roi=region)
        _sheet(variants, outdir / f"{label}_boost.png", roi=region, lut=lut)
        _diff_panel(variants, outdir / f"{label}_difference_x12.png", roi=region)
    if args.roi is not None:
        roi = tuple(args.roi)
        _sheet(variants, outdir / "custom_normal.png", roi=roi)
        _sheet(variants, outdir / "custom_boost.png", roi=roi, lut=lut)

    save_image(outdir / "delta_plus_periodic.png", periodic_result,
               image_format="png", bit_depth=8)
    details = {}
    regions = TEAPOT_ROIS if args.sample=="teapot" else PORTRAIT_ROIS
    for region_name, box in regions.items():
        region_images = {name: _crop(rgb, box) for name, rgb in variants.items()}
        original_energy = _rms_highpass(region_images[NAMES[0]])
        details[region_name] = {
            name: {
                "highpass_rms": round(_rms_highpass(rgb), 5),
                "highpass_vs_original_percent": round(
                    100*_rms_highpass(rgb)/(original_energy+1e-8), 2),
                "difference_from_original": _difference(rgb, region_images[NAMES[0]]),
                "difference_from_balanced": _difference(rgb, region_images[NAMES[1]]),
            }
            for name, rgb in region_images.items()
        }

    report = {
        "source_sweep": str(sweep.relative_to(ROOT)),
        "source_precision": "8-bit PNG output of prior 2x sweep; no 2x rerun",
        "comparison": list(variants.keys()),
        "periodic_stage": stage_report,
        "region_metrics": details,
        "contrast": {
            "reference": "same original V=max(R,G,B) histogram for all variants",
            "method": "docs/site.js _boostCurve; grayscale V, tail 0.2%, slope 0.3..4",
            "visualization_only": True,
        },
        "caveats": [
            "Highpass energy includes both real texture and unwanted artifacts.",
            "Periodic may detect no component; check periodic_stage status and removed_rms.",
            "8-bit source roundtrips slightly differ from the unsaved float pipeline.",
            "Contrast Boost is only a visualization and can amplify innocuous variations.",
            "Upstream example media must remain local and cannot be redistributed.",
        ],
    }
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                   default=_json_default), encoding="utf-8")
    for key in ("normal","boost","difference_x12",
                "floor_boost","floor_difference_x12","lid_boost",
                "lid_difference_x12","handle_boost","handle_difference_x12",
                "forehead_boost","forehead_difference_x12"):
        source = outdir / f"{key}.png"
        if source.is_file():
            copyfile(source, parent / f"latest_{key}.png")
    print(f"Periodic: status={stage_report.get('status')}, "
          f"components={len(stage_report.get('components', []))}, "
          f"removed_rms={stage_report.get('removed_rms',0):.5f}", flush=True)
    print(f"Open: {parent/'latest_boost.png'}")
    if args.sample=="teapot":
        print(f"Floor: {parent/'latest_floor_boost.png'}")
        print(f"Lid: {parent/'latest_lid_boost.png'}")
        print(f"Handle: {parent/'latest_handle_boost.png'}")
    print(f"Archived result: {outdir}")
    print("Do not publish the generated sample images.")


if __name__ == "__main__":
    main()
