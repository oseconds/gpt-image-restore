"""Check whether 2px X/Y-only periodic removal is preferable to all components.

Uses the previously saved native-resolution 2x_delta_050.png from the local
resampler sweep. Does not rerun 2x Balanced. Keeps upstream source code intact.
Generates Original / Balanced / 2x Delta 0.50 / X-Y only / Full Periodic,
with original-based web-equivalent Contrast Boost for visual inspection.

Upstream docs/media examples cannot be redistributed; generated results are
saved only under gitignored docs/local_media/.

  python experiments/compare_periodic_axes.py --sample teapot
  open docs/local_media/periodic-axes-teapot/latest_floor_boost.png
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_delta_periodic import (
    _boost_lut, _crop, _difference, _load_sweep, _rgb_file,
    _rms_highpass, _sheet, TEAPOT_ROIS, PORTRAIT_ROIS,
)
from experiments.mesh_peak_probe import maps_for_image
from gptimg_restore import periodic
from gptimg_restore.io_utils import save_image

AXIS_FAMILIES = {(4, 0), (0, 4)}
NAMES = (
    "Original",
    "Balanced",
    "2x Delta 0.50",
    "2x Delta + X/Y 2px",
    "2x Delta + Full Periodic",
)


def axis_only_periodic(rgb: np.ndarray, strength: float = 1.0):
    """Reuse periodic's *same* fit/model, but reconstruct only 2px X/Y carriers.

    The internal estimator is private; this wrapper is intentionally kept in
    experiments instead of changing the upstream public API.
    """
    report, model = periodic._estimate(rgb)
    if model is None:
        return rgb.copy(), {
            "status": report["status"], "selected_components": [],
            "removed_rms": 0.0,
        }
    h, w = model["shape"]
    removal = np.zeros((h, w, 3), np.float32)
    selection = []
    for candidate in model["candidates"]:
        family = tuple(candidate.get("family") or ())
        if family not in AXIS_FAMILIES:
            continue
        fx, fy = candidate["frequency"]
        start, stop = candidate["columns"]
        cosine, sine = periodic._carrier_basis(
            fx, fy, candidate["origin_phase"], h, w)
        channels = []
        for channel in range(3):
            coefficients = model["coefficients"][start:stop, channel]
            if not np.any(coefficients):
                continue
            carrier = coefficients[0] * cosine
            if sine is not None:
                carrier = carrier + coefficients[1] * sine
            if model["spatial_gain"]:
                carrier *= periodic._slow_gain(model, candidate, channel)
            removal[..., channel] += carrier
            channels.append(channel)
        selection.append({"family": list(family), "frequency": [fx, fy],
                          "active_channels": channels})
    if not selection:
        return rgb.copy(), {"status": "no_axis_component",
                            "selected_components": [], "removed_rms": 0.0}
    # Match periodic.apply's clipped-pixel protection and opponent->RGB step.
    protection = 1 - ndimage.gaussian_filter(
        model["clipped"].astype(np.float32), 2)
    protection[model["clipped"]] = 0
    removal *= protection[..., None]
    removed_rgb = periodic._rgb(removal) * np.float32(strength)
    output = np.asarray(rgb, np.float32) - removed_rgb
    return output.astype(np.float32, copy=False), {
        "status": "detected",
        "selected_components": selection,
        "removed_rms": float(np.sqrt(np.mean(removed_rgb ** 2))),
        "mean_abs_removed_dn": float(np.mean(np.abs(removed_rgb))),
        "strength": strength,
    }


def _peak_metrics(rgb: np.ndarray, roi: tuple[int, int, int, int]) -> dict:
    region = _crop(rgb, roi)
    if min(region.shape[:2]) < 64:
        return {"skipped": "ROI shorter than 64px"}
    _, result = maps_for_image(region, stride=16)
    return {
        label: {
            "mean_amplitude": result[label]["mean_amplitude"],
            "median_prominence": result[label]["median_prominence"],
        }
        for label in (
            "axis_31_0", "axis_0_31",
            "diagonal_16_-16", "opposite_16_16"
        )
    }


def _diff_sheet(variants: dict[str, np.ndarray], out: Path,
                roi: tuple[int, int, int, int]) -> None:
    # Three changes on exactly the same absolute DN display scale.
    from PIL import Image, ImageDraw
    delta = variants["2x Delta 0.50"]
    axis = variants["2x Delta + X/Y 2px"]
    full = variants["2x Delta + Full Periodic"]
    differences = {
        "X/Y-only removed (abs x16)": np.abs(delta - axis) * 16,
        "Full Periodic removed (abs x16)": np.abs(delta - full) * 16,
        "Axis-only vs Full (abs x16)": np.abs(axis - full) * 16,
    }
    views = [_crop(img, roi) for img in differences.values()]
    h, w = views[0].shape[:2]
    image = Image.new("RGB", (w*len(views), h+28), "white")
    draw = ImageDraw.Draw(image)
    for i, (name, pixels) in enumerate(zip(differences, views)):
        draw.text((i*w+4, 6), name, fill="black")
        image.paste(Image.fromarray(
            np.rint(np.clip(pixels, 0, 255)).astype(np.uint8), "RGB"),
            (i*w, 28))
    image.save(out)


def run(args) -> None:
    folder = _load_sweep(args.sample, args.sweep_folder)
    original, balanced, delta = (
        _rgb_file(folder / filename)
        for filename in ("original.png", "balanced.png", "2x_delta_050.png")
    )
    print(f"Reusing saved sweep: {folder}", flush=True)
    print("Fitting the same periodic model with X/Y carriers only...", flush=True)
    axis, axis_report = axis_only_periodic(delta, strength=args.strength)
    print("Applying full existing periodic module for control...", flush=True)
    full, full_report = periodic.apply(delta.copy(), strength=args.strength)
    variants = dict(zip(NAMES, (
        original, balanced, delta,
        np.clip(axis, 0, 255),
        np.clip(full, 0, 255),
    )))
    lut = _boost_lut(original)  # one full-image, original-based LUT for all
    rois = TEAPOT_ROIS if args.sample == "teapot" else PORTRAIT_ROIS

    parent = ROOT / "docs" / "local_media" / f"periodic-axes-{args.sample}"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = parent / stamp
    i = 1
    while out.exists():
        out = parent / f"{stamp}-{i}"
        i += 1
    out.mkdir()

    _sheet(variants, out / "normal.png", max_width=640)
    _sheet(variants, out / "boost.png", lut=lut, max_width=640)
    for label, roi in rois.items():
        _sheet(variants, out / f"{label}_normal.png", roi=roi)
        _sheet(variants, out / f"{label}_boost.png", roi=roi, lut=lut)
        _diff_sheet(variants, out / f"{label}_removed_x16.png", roi)
    save_image(out / "axis_only.png", axis, image_format="png", bit_depth=8)
    save_image(out / "periodic_full.png", full, image_format="png", bit_depth=8)

    region_metrics = {}
    for region_name, roi in rois.items():
        reference = _crop(original, roi)
        ref_energy = _rms_highpass(reference)
        region_metrics[region_name] = {}
        for name, rgb in variants.items():
            patch = _crop(rgb, roi)
            region_metrics[region_name][name] = {
                "highpass_rms": round(_rms_highpass(patch), 5),
                "highpass_vs_original_pct": round(
                    100*_rms_highpass(patch)/(ref_energy+1e-8), 2),
                "difference_vs_original": _difference(patch, reference),
                "fft_peaks": _peak_metrics(rgb, roi),
            }

    result = {
        "source_sweep": str(folder),
        "source_precision": "Saved 8-bit sweep images; no rerun at 2x",
        "periodic_strength": args.strength,
        "axis_only": axis_report,
        "full_periodic": {
            "status": full_report.get("status"),
            "component_families": [
                c.get("eighth_grid_vector")
                for c in full_report.get("components", [])
            ],
            "removed_rms": full_report.get("removed_rms", 0.0),
        },
        "regions": region_metrics,
        "contrast_boost": "Web-style original V histogram; same LUT for all, "
                          "visualization only, not used for metrics.",
        "cautions": [
            "Resampled 2x filter delta is an alternative to Balanced, not an extra pass.",
            "Periodic axis selector uses upstream private internals and may need "
            "updates if periodic.py changes.",
            "Fixed FFT bins and highpass energy include real textures too.",
            "All upstream sample derivatives must remain local.",
        ],
    }
    (out / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("normal", "boost", *[
        f"{region}_{kind}" for region in rois
        for kind in ("boost", "normal", "removed_x16")
    ]):
        copyfile(out / f"{name}.png", parent / f"latest_{name}.png")
    (parent / "latest_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Axis-only: {len(axis_report['selected_components'])} components, "
          f"removed RMS {axis_report['removed_rms']:.4f} DN")
    print(f"Full: {len(full_report.get('components', []))} components, "
          f"removed RMS {full_report.get('removed_rms', 0.0):.4f} DN")
    print(f"Contrast view: {parent/'latest_boost.png'}")
    for region in rois:
        print(f"{region}: {parent/f'latest_{region}_boost.png'}")
    print(f"Report: {parent/'latest_report.json'}")
    print("Sample derivatives remain local; do not publish.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="teapot", choices=("teapot", "portrait"))
    p.add_argument("--sweep-folder", type=Path)
    p.add_argument("--strength", type=float, default=1.0)
    args = p.parse_args()
    if not 0 <= args.strength <= 2:
        p.error("--strength must be between 0 and 2")
    run(args)


if __name__ == "__main__":
    main()
