"""Try resampling order around Balanced without altering the upstream pipeline.

All generated outputs stay local in gitignored docs/local_media. The examples
in docs/media have a no-redistribution notice; never commit these derivatives.

Quick:
  python experiments/compare_resampler_paths.py --sample portrait
  open docs/local_media/resampler-portrait/comparison.png

Optional slower test (Balanced itself runs at twice the native resolution):
  python experiments/compare_resampler_paths.py --sample portrait --balanced-at-2x

The doubled image cannot contain new detail. Compare the 2x resampling-only
control with the 2x Balanced result before attributing changes to restoration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multiscale_restore import MultiscaleOptions, resize_rgb
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import restore_array


def resize(image: np.ndarray, width: int, height: int, kernel: str) -> np.ndarray:
    return resize_rgb(image, width, height,
                      MultiscaleOptions(resampler="zscale", kernel=kernel))


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    difference = np.abs(a.astype(np.float32) - b.astype(np.float32))
    mean_channels = difference.mean(axis=2)
    return {
        "mean_abs_change_dn": round(float(difference.mean()), 6),
        "max_change_dn": round(float(difference.max()), 4),
        "fraction_pixels_ge_1dn": round(float((mean_channels >= 1).mean()), 6),
    }


def crop(img: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, w, h = roi
    ih, iw = img.shape[:2]
    if not (0 <= x < iw and 0 <= y < ih and w > 0 and h > 0
            and x + w <= iw and y + h <= ih):
        raise ValueError(f"ROI {roi} is outside {iw}x{ih}")
    return img[y:y+h, x:x+w]


def uint8(image: np.ndarray) -> np.ndarray:
    return np.uint8(np.rint(np.clip(image, 0, 255)))


def comparison(images: dict[str, np.ndarray], baseline: np.ndarray,
               roi: list[int], out: Path) -> None:
    columns = 4
    names = list(images)
    # Render all variants, then an amplified difference row for each.
    rows = (len(names) + columns - 1) // columns
    height, width = crop(baseline, roi).shape[:2]
    label_h = 27
    sheet = Image.new("RGB", (columns * width,
                             2 * rows * (height + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_type in range(2):
        for index, (name, rgb) in enumerate(images.items()):
            row = row_type * rows + index // columns
            col = index % columns
            left, top = col * width, row * (height + label_h)
            label = name if row_type == 0 else f"abs({name} - Balanced) x12"
            draw.text((left + 6, top + 6), label, fill="black")
            view = crop(rgb, roi)
            if row_type == 1:
                view = np.abs(view - crop(baseline, roi)) * 12
            sheet.paste(Image.fromarray(uint8(view)), (left, top + label_h))
    sheet.save(out, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="portrait")
    parser.add_argument("--half-blend", type=float, default=0.25,
                        help="How much 0.5x roundtrip to mix into original before Balanced (0..1)")
    parser.add_argument("--balanced-at-2x", action="store_true",
                        help="Also run full Balanced at 2x resolution; slower and more memory")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"),
                        help="Native-pixel view crop for comparison, not processing")
    args = parser.parse_args()
    if not 0 <= args.half_blend <= 1:
        parser.error("--half-blend must be in [0,1]")
    showcase = json.loads((ROOT / "docs" / "showcase.json").read_text(encoding="utf-8"))
    items = {item["id"]: item for item in showcase["items"]}
    if args.sample not in items:
        parser.error(f"Unknown sample {args.sample!r}; choose {list(items)}")
    item = items[args.sample]
    loaded = load_image(ROOT / "docs" / item["images"]["original"])
    original = np.asarray(loaded.rgb, dtype=np.float32)
    height, width = original.shape[:2]
    roi = list(args.roi if args.roi is not None else item["view"])
    outputs = ROOT / "docs" / "local_media" / f"resampler-{args.sample}"
    outputs.mkdir(parents=True, exist_ok=True)
    timings = {}

    def balanced(image: np.ndarray, label: str) -> np.ndarray:
        started = time.perf_counter()
        # Pass float32 without any intermediate uint8 quantization.
        data = replace(loaded, rgb=np.asarray(image, np.float32), bit_depth=16,
                       size=(image.shape[1], image.shape[0]))
        rgb, _, _ = restore_array(data, preset="balanced", bit_depth=16)
        timings[label] = round(time.perf_counter() - started, 2)
        return rgb

    print("Balanced baseline...", flush=True)
    base = balanced(original, "balanced_native")

    print("0.5x Lanczos -> 1x Spline36 roundtrip...", flush=True)
    half = resize(original, max(32, round(width * .5)),
                  max(32, round(height * .5)), "lanczos")
    half_back = resize(half, width, height, "spline36")
    half_mixed = original + args.half_blend * (half_back - original)
    half_pre = balanced(half_mixed, "half_roundtrip_before_balanced")

    print("2x Spline36 -> 1x Lanczos roundtrip...", flush=True)
    double = resize(original, 2 * width, 2 * height, "spline36")
    double_back = resize(double, width, height, "lanczos")
    two_pre = balanced(double_back, "double_roundtrip_before_balanced")

    variants = {
        "Original": original, "Balanced": base,
        "Half roundtrip (pure)": half_back,
        "Half blend > Balanced": half_pre,
        "2x roundtrip (pure)": double_back,
        "2x roundtrip > Balanced": two_pre,
    }
    if args.balanced_at_2x:
        print("Balanced at 2x then downsample (slow)...", flush=True)
        processed2x = balanced(double, "balanced_at_double_resolution")
        down_restored = resize(processed2x, width, height, "lanczos")
        # Control-subtracted delta separates filter change from resampling-only change.
        filtered_delta = down_restored - double_back
        variants["2x Balanced > down"] = down_restored
        variants["2x filter delta only"] = np.clip(original + filtered_delta, 0, 255)

    report = {
        "sample": args.sample,
        "settings": {"half_blend": args.half_blend,
                     "full_2x_balanced": args.balanced_at_2x,
                     "half_path": "Lanczos down to 0.5x; Spline36 up to 1x",
                     "double_path": "Spline36 up to 2x; Lanczos down to 1x"},
        "roi_xywh": roi,
        "timings_seconds": timings,
        "differences_vs_balanced": {name: stats(value, base)
                                    for name, value in variants.items()},
        "roundtrip_only_vs_original": {
            "half": stats(half_back, original),
            "double": stats(double_back, original),
        },
        "warning": "Simple roundtrip is low-pass filtering and may remove real "
                   "hair/skin texture. Difference magnitudes do not measure quality. "
                   "This experiment does not demonstrate recovery of lost detail.",
        "rights": "Local-only outputs; upstream sample derivative reuse is prohibited.",
    }
    for name, rgb in variants.items():
        slug = name.lower().replace(" ", "_").replace(">", "to").replace("(", "").replace(")", "")
        save_image(outputs / f"{slug}.png", rgb, loaded.alpha, image_format="png",
                   bit_depth=8, icc_profile=loaded.icc_profile)
    comparison(variants, base, roi, outputs / "comparison.png")
    (outputs / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparison: {outputs / 'comparison.png'}")
    print(json.dumps({"roundtrip_only_vs_original": report["roundtrip_only_vs_original"],
                      "differences_vs_balanced": report["differences_vs_balanced"]},
                     indent=2, ensure_ascii=False))
    print("Local output only. Do not publish derived upstream demo images.")


if __name__ == "__main__":
    main()
