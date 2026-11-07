"""Compare two simple resampling additions to the original Balanced pipeline.

Experiment A: 0.5x Lanczos -> 1x Spline36 roundtrip, blend 10/25/50%
              with original, then run native-resolution Balanced.
Experiment B: 2x Spline36 -> Balanced at 2x -> 1x Lanczos;
              remove the pure roundtrip delta, then add only the
              filter-derived correction to original at 50/100/150%.

No shipped preset is modified. Results are diagnostic, not a new filter.
The upstream docs/media examples explicitly forbid reuse: outputs stay in
gitignored docs/local_media and must never be committed or published.

From the repo root:
  python experiments/sweep_resampler_paths.py --sample portrait
  open docs/local_media/resampler-sweep-portrait/latest.png
  python experiments/sweep_resampler_paths.py --sample teapot
  open docs/local_media/resampler-sweep-teapot/latest.png

Each run is also archived in a timestamped subdirectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_resampler_paths import resize, stats
from experiments.mesh_peak_probe import maps_for_image
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import restore_array

HALF_BLEND = (0.10, 0.25, 0.50)
DOUBLE_DELTA = (0.50, 1.00, 1.50)
PEAKS = ("diagonal_16_-16", "opposite_16_16", "axis_31_0", "axis_0_31")


def _crop(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = region
    h, w = image.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > w or y + height > h:
        raise ValueError(f"Invalid native crop {region} for image {w}x{h}")
    return image[y:y+height, x:x+width]


def _image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(image, 0, 255)).astype(np.uint8), "RGB")


def _comparison(variants: dict[str, np.ndarray], baseline: np.ndarray,
                region: tuple[int, int, int, int], outfile: Path) -> None:
    items = list(variants.items())
    if len(items) != 9:
        raise AssertionError("Expected nine experiment variants")
    _, _, w, h = region
    label_h = 27
    # Top 2 rows: native pixel crops. Bottom 2 rows: abs difference from Balanced x12.
    sheet = Image.new("RGB", (w*3, (h+label_h)*6), "white")
    draw = ImageDraw.Draw(sheet)
    b = _crop(baseline, region)
    for idx, (name, full) in enumerate(items):
        cropped = _crop(full, region)
        for is_difference in (False, True):
            row = (idx // 3) + (3 if is_difference else 0)
            col = idx % 3
            x, y = col*w, row*(h+label_h)
            title = name if not is_difference else f"{name} - Balanced (abs x12)"
            draw.text((x+5, y+5), title, fill="black")
            pixels = np.abs(cropped - b) * 12 if is_difference else cropped
            sheet.paste(_image(pixels), (x, y+label_h))
    sheet.save(outfile)


def _peak_stats(variant: np.ndarray, region: tuple[int, int, int, int]) -> dict:
    cropped = _crop(variant, region)
    if min(cropped.shape[:2]) < 64:
        return {"skipped": "ROI is smaller than 64 native pixels"}
    _, summary = maps_for_image(cropped, stride=16)
    return {
        key: {field: summary[key][field]
              for field in ("median_prominence", "p90_prominence",
                            "mean_amplitude", "mean_local_baseline")}
        for key in PEAKS
    }


def _run(args: argparse.Namespace) -> None:
    showcase = json.loads((ROOT / "docs" / "showcase.json").read_text(encoding="utf-8"))
    samples = {item["id"]: item for item in showcase["items"]}
    if args.sample not in samples:
        raise ValueError(f"Unknown sample {args.sample!r}. Options: {list(samples)}")
    item = samples[args.sample]
    loaded = load_image(ROOT / "docs" / item["images"]["original"])
    original = np.asarray(loaded.rgb, dtype=np.float32)
    h, w = original.shape[:2]
    region = tuple(args.roi or
                   ([555, 100, 330, 280] if args.sample == "portrait"
                    else item["view"]))

    # Float32 -> restore_array without an intermediate 8-bit quantization.
    # This keeps the three blends comparable to the same native baseline.
    def balanced(rgb: np.ndarray) -> np.ndarray:
        data = replace(loaded, rgb=np.asarray(rgb, dtype=np.float32),
                       bit_depth=16, size=(rgb.shape[1], rgb.shape[0]))
        result, _, _ = restore_array(data, preset="balanced", bit_depth=16)
        return np.asarray(result, dtype=np.float32)

    print("Balanced original...", flush=True)
    base = balanced(original)

    print("Half-size Lanczos -> Spline36 roundtrip...", flush=True)
    half = resize(original, max(32, round(w/2)), max(32, round(h/2)), "lanczos")
    half_back = resize(half, w, h, "spline36")

    variants = {"Original": original, "Balanced": base}
    for weight in HALF_BLEND:
        print(f"Half roundtrip {weight:.2f} blend -> Balanced...", flush=True)
        pref = np.clip(original + weight*(half_back - original), 0, 255)
        variants[f"Half blend {weight:.2f}"] = balanced(pref)

    print("Spline36 2x enlargement...", flush=True)
    doubled = resize(original, 2*w, 2*h, "spline36")
    pure_roundtrip = resize(doubled, w, h, "lanczos")
    print("Balanced at 2x, then Lanczos downsampling...", flush=True)
    restored_doubled = balanced(doubled)
    restored_down = resize(restored_doubled, w, h, "lanczos")

    # Subtract the control that underwent precisely the same interpolation.
    # This difference contains the 2x filter correction rather than the
    # resampling-only change. It is NOT the equivalent of ordinary Balanced.
    filter_delta = restored_down - pure_roundtrip
    for gain in DOUBLE_DELTA:
        variants[f"2x delta {gain:.2f}"] = np.clip(original + gain*filter_delta, 0, 255)

    # Final reference: the full 2x restoration with resampling effects included.
    variants["2x Balanced full"] = restored_down

    parent = ROOT / "docs" / "local_media" / f"resampler-sweep-{args.sample}"
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = parent / timestamp
    suffix = 1
    while target.exists():
        target = parent / f"{timestamp}-{suffix}"
        suffix += 1
    target.mkdir()

    _comparison(variants, base, region, target / "comparison.png")
    for name, pixels in variants.items():
        slug = name.lower().replace(" ", "_").replace(".", "")
        save_image(target / f"{slug}.png", pixels, loaded.alpha, image_format="png",
                   bit_depth=8, icc_profile=loaded.icc_profile)

    report = {
        "sample": args.sample,
        "native_roi_xywh": list(region),
        "half_roundtrip": "0.5x Lanczos -> 1x Spline36, mixed BEFORE Balanced",
        "double_delta": "Spline36 2x, Balanced at 2x, Lanczos down; "
                        "subtract interpolation-only roundtrip; add correction to original",
        "half_blends": list(HALF_BLEND),
        "double_delta_gains": list(DOUBLE_DELTA),
        "statistics": {
            name: {
                "vs_original": stats(value, original),
                "vs_balanced": stats(value, base),
                "roi_vs_original": stats(_crop(value, region), _crop(original, region)),
                "roi_vs_balanced": stats(_crop(value, region), _crop(base, region)),
            } for name, value in variants.items()
        },
        # Fixed-frequency controls are meaningful chiefly for the portrait;
        # they are not a validated measure of image quality.
        "fft_peak_statistics_roi": {
            name: _peak_stats(value, region) for name, value in variants.items()
        },
        "limitations": [
            "These are not objective image quality scores.",
            "FFT peaks can include actual details and WebP artifacts, not only GPT artifacts.",
            "2x delta is an alternative treatment of Original, not an extra pass after Balanced.",
            "Upscaling does not add recoverable ground-truth texture.",
            "Upstream example images exclude reuse permission; results must stay local.",
        ],
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep all historical runs while allowing a stable command to open the newest.
    copyfile(target / "comparison.png", parent / "latest.png")
    (parent / "latest-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Archived comparison: {target / 'comparison.png'}")
    print(f"Open latest: {parent / 'latest.png'}")
    for name, value in report["statistics"].items():
        print(f"  {name:20s}: mean_abs_vs_Balanced={value['vs_balanced']['mean_abs_change_dn']:.4f} DN")
    print("No media is added to Git. Do not publish derived upstream demo images.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="portrait", help="Sample from docs/showcase.json")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    args = p.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
