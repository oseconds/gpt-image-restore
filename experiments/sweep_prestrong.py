"""Sweep pre-cleaning strengths BEFORE the native Strong preset.

Uses new user-controlled images only and appends results to the existing,
gitignored docs/test_media/<image>/ outputs and showcase.tests.json.

Default controls:
  Strong (reused if saved)
  Half-roundtrip Lanczos 0.05 / 0.10 / 0.25 -> Strong
  2x Balanced filter delta 0.25 / 0.50 / 1.00
      -> native axis-only 2px periodic (strength 1.0) -> Strong

The costly 2x Balanced operation runs only ONCE across all delta gains.
The 0.5x Lanczos -> 1x Spline36 roundtrip also runs only once.
Strong runs independently on each pre-clean variant; no Strong coefficients
or masks are reused across images. No original/saved Balanced/Experimental
images are modified. Existing upstream demos are excluded.

Usage:
  python experiments/sweep_prestrong.py test_inputs/design-poster.png
  python experiments/sweep_prestrong.py test_inputs/design-poster.png \
    --half 0.10 --delta 0.50 1.00
  python3 -m http.server 8000 --directory docs
  # Open http://localhost:8000/?tests=1
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_owned_images import (
    PRIVATE_MANIFEST, PRIVATE_MEDIA, _change, _slug, collect_images,
)
from experiments.experimental_restore import axis_only_periodic
from experiments.multiscale_restore import MultiscaleOptions, resize_rgb
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import restore_array


def gain_key(kind: str, gain: float) -> str:
    """Stable, human-readable keys like pre_half_010 or pre_delta_050."""
    scaled = round(gain * 100)
    if not np.isclose(gain * 100, scaled, atol=1e-6):
        raise ValueError("Strengths must be multiples of 0.01")
    return f"pre_{kind}_{scaled:03d}"


def _strong(loaded, precleaned: np.ndarray) -> np.ndarray:
    """Run the same Strong preset with the input's bit-depth semantics."""
    prepared = replace(loaded, rgb=np.asarray(precleaned, np.float32))
    result, _, _ = restore_array(prepared, preset="strong", bit_depth=16)
    return np.clip(result, 0, 255)


def _resize(image: np.ndarray, w: int, h: int, method: str,
            resampler: str) -> np.ndarray:
    opts = MultiscaleOptions(resampler=resampler, kernel=method)
    return resize_rgb(image, w, h, opts)


def _compute_delta(loaded, original: np.ndarray, resampler: str) -> np.ndarray:
    """2x Spline36 -> Balanced at 2x -> Lanczos 1x, minus pure roundtrip.

    The returned signed correction is shared by ALL delta strengths. The
    correction is not an upscaled image or hallucinated higher-resolution detail.
    """
    h, w = original.shape[:2]
    up = _resize(original, 2*w, 2*h, "spline36", resampler)
    print("  2x Balanced calculation (once for all delta gains)...", flush=True)
    data = replace(loaded, rgb=up, size=(2*w, 2*h), bit_depth=16)
    restored_up, _, _ = restore_array(data, preset="balanced", bit_depth=16)
    down_restored = _resize(restored_up, w, h, "lanczos", resampler)
    down_pure = _resize(up, w, h, "lanczos", resampler)
    return down_restored - down_pure


def _compute_half_roundtrip(original: np.ndarray, resampler: str) -> np.ndarray:
    """Lanczos 0.5x -> Spline36 1x, matching the earlier resampler experiment."""
    h, w = original.shape[:2]
    down = _resize(original, max(32, round(w*0.5)), max(32, round(h*0.5)),
                   "lanczos", resampler)
    return _resize(down, w, h, "spline36", resampler)


def _save_variant(target: Path, key: str, rgb: np.ndarray, loaded) -> None:
    save_image(target / f"{key}.png", rgb, loaded.alpha, image_format="png",
               bit_depth=8, icc_profile=loaded.icc_profile)


def run_one(path: Path, args, manifest: dict) -> None:
    slug = _slug(path)
    items = {item["id"]: item for item in manifest.get("items", [])}
    if slug not in items:
        raise ValueError(
            f"{slug} is not in {PRIVATE_MANIFEST}. "
            "Run compare_owned_images.py on the same input first.")
    item = items[slug]
    target = PRIVATE_MEDIA / slug
    for required in ("original.png", "balanced.png", "experimental.png", "report.json"):
        if not (target / required).is_file():
            raise FileNotFoundError(f"Expected existing A/B test output: {target / required}")

    loaded = load_image(path)
    if loaded.size != (item["width"], item["height"]):
        raise ValueError(f"Input dimensions changed for {path.name}; rerun baseline A/B.")

    source = np.asarray(loaded.rgb, np.float32)
    report_path = target / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("source_name") != path.name:
        raise ValueError(
            f"Existing test source name differs ({report.get('source_name')!r}); "
            "rerun the baseline A/B test before comparison.")

    new_item = dict(item)
    new_item["images"] = dict(item["images"])
    new_item["labels"] = dict(item.get("labels", {}))
    sweep = {
        "comparison": "pre-clean -> Strong vs Strong-only",
        "source": path.name, "resampler": args.resampler,
        "native_output_bit_depth": 8,
        "strong_processing_bit_depth": 16,
        "preclean_intermediate": "float32 0..255, no 8-bit intermediate quantization",
        "axis_periodic_strength": args.xy_strength,
        "half_path": "Lanczos down 0.5x then Spline36 up to 1x",
        "delta_path": "Spline36 up 2x -> Balanced@2x -> Lanczos down 1x; subtract pure roundtrip",
        "variants": {},
        "warning": "Mean absolute change is not a metric of perceptual quality. "
                   "The Strong preset remains unchanged, so pre-cleaning need "
                   "not prevent text damage.",
    }

    if (target / "strong.png").is_file():
        print("  Using existing native Strong baseline.", flush=True)
        new_item["images"]["strong"] = f"test_media/{slug}/strong.png"
    else:
        print("  Strong baseline...", flush=True)
        started = time.perf_counter()
        baseline = _strong(loaded, source)
        _save_variant(target, "strong", baseline, loaded)
        new_item["images"]["strong"] = f"test_media/{slug}/strong.png"
        sweep["strong_baseline_seconds"] = round(time.perf_counter() - started, 3)

    if args.half:
        print("  0.5x Lanczos / 1x Spline36 roundtrip (once)...", flush=True)
        half = _compute_half_roundtrip(source, args.resampler)
        diff = half - source
        for amount in args.half:
            key = gain_key("half", amount)
            print(f"  Half blend {amount:.2f} -> Strong...", flush=True)
            started = time.perf_counter()
            pre = np.clip(source + amount * diff, 0, 255)
            result = _strong(loaded, pre)
            elapsed = time.perf_counter() - started
            _save_variant(target, key, result, loaded)
            new_item["images"][key] = f"test_media/{slug}/{key}.png"
            new_item["labels"][key] = f"Half {amount:.2f} -> Strong"
            sweep["variants"][key] = {
                "half_blend": amount, "elapsed_strong_seconds": round(elapsed, 3),
                "vs_original": _change(result, source),
            }
        del half, diff

    if args.delta:
        print("  Preparing shared 2x Balanced delta...", flush=True)
        started_delta = time.perf_counter()
        delta = _compute_delta(loaded, source, args.resampler)
        sweep["shared_2x_balanced_seconds"] = round(time.perf_counter()-started_delta, 3)
        for amount in args.delta:
            key = gain_key("delta", amount)
            print(f"  2x Delta {amount:.2f} + XY {args.xy_strength:.2f} -> Strong...", flush=True)
            started = time.perf_counter()
            pre = np.clip(source + amount * delta, 0, 255)
            if args.xy_strength > 0:
                pre, detected = axis_only_periodic(pre, strength=args.xy_strength)
                pre = np.clip(pre, 0, 255)
            else:
                detected = {"status": "disabled"}
            result = _strong(loaded, pre)
            elapsed = time.perf_counter() - started
            _save_variant(target, key, result, loaded)
            new_item["images"][key] = f"test_media/{slug}/{key}.png"
            xy_suffix = f" + XY {args.xy_strength:g}" if args.xy_strength > 0 else ""
            new_item["labels"][key] = f"2x Delta {amount:.2f}{xy_suffix} -> Strong"
            sweep["variants"][key] = {
                "delta_gain": amount, "axis_periodic": detected,
                "elapsed_strong_seconds": round(elapsed, 3),
                "vs_original": _change(result, source),
            }

    report["prestrong_sweep"] = sweep
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    manifest["items"] = [
        new_item if x["id"] == slug else x for x in manifest["items"]
    ]
    tmp = PRIVATE_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(PRIVATE_MANIFEST)
    print(f"  Complete: {len(sweep['variants'])} pre-clean variants in {target}", flush=True)


def _unique_strengths(values: list[float], *, minimum: float, maximum: float,
                      name: str, parser: argparse.ArgumentParser) -> list[float]:
    result = list(dict.fromkeys(values))
    for value in result:
        if not np.isfinite(value) or not minimum <= value <= maximum:
            parser.error(f"{name}: strengths must lie in [{minimum}, {maximum}]")
        if not np.isclose(value * 100, round(value * 100), atol=1e-6):
            parser.error(f"{name}: use up to two decimal places")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Previously processed user-owned images or directories")
    parser.add_argument("--half", type=float, nargs="*", default=[0.05, 0.10, 0.25],
                        help="Half roundtrip blend gains, default 0.05 0.10 0.25")
    parser.add_argument("--delta", type=float, nargs="*", default=[0.25, 0.50, 1.00],
                        help="2x Balanced filter delta gains, default 0.25 0.50 1.00")
    parser.add_argument("--xy-strength", type=float, default=1.0,
                        help="Axis-only 2px periodic strength after 2x delta; 0 disables it")
    parser.add_argument("--resampler", choices=("zscale", "opencv"), default="zscale",
                        help="zscale is actual Spline36/Lanczos; opencv is approximate")
    args = parser.parse_args()

    args.half = _unique_strengths(args.half, minimum=0, maximum=1,
                                  name="--half", parser=parser)
    args.delta = _unique_strengths(args.delta, minimum=0, maximum=2,
                                   name="--delta", parser=parser)
    if not np.isfinite(args.xy_strength) or not 0 <= args.xy_strength <= 2:
        parser.error("--xy-strength must be in [0,2]")
    if not args.half and not args.delta:
        parser.error("Choose at least one nonempty --half or --delta sweep")

    images = collect_images(args.inputs)
    if not PRIVATE_MANIFEST.is_file():
        parser.error(f"Missing {PRIVATE_MANIFEST}; first run compare_owned_images.py")
    manifest = json.loads(PRIVATE_MANIFEST.read_text(encoding="utf-8"))

    for path in images:
        print(f"[{path.name}] Pre-clean -> Strong sweep:", flush=True)
        run_one(path, args, manifest)

    print("\nOpen: http://localhost:8000/?tests=1", flush=True)
    print("Use the A/B dropdowns (Strong vs Pre-clean variants), zoom and Contrast Boost.")
    print("This is a local-only exploratory experiment; nothing is published.")


if __name__ == "__main__":
    main()
