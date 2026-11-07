"""Experimental detail-preserving restoration (not part of upstream presets).

One alternative pipeline explored on user-controlled test inputs:
  Original --Spline36 2x--> Balanced at 2x --Lanczos 1x-->
  subtract the interpolation-only roundtrip and keep 0.50 of the
  2x-restoration delta on Original --> native 2px X/Y Periodic only.

Unlike a second Balanced pass, this is an alternative restoration of Original.
The 2x operation does not recover information absent in the source.

The isolated periodic implementation reuses upstream's periodic detector/fit
but selects only the native (4,0) and (0,4) eighth-grid carrier families.
Upstream API / defaults are untouched.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy import ndimage

from experiments.multiscale_restore import MultiscaleOptions, resize_rgb
from gptimg_restore import periodic
from gptimg_restore.pipeline import restore_array

AXIS_FAMILIES = {(4, 0), (0, 4)}


def axis_only_periodic(rgb: np.ndarray, strength: float = 1.0):
    """Remove only detected native 2-pixel X/Y periodic carriers.

    Reuses the same upstream model, fitted coefficients and spatial gains as
    periodic.apply(), which may change when upstream's private helpers change.
    This is deliberately experiment-scoped until its behavior is validated.
    """
    report, model = periodic._estimate(np.asarray(rgb, np.float32))
    if model is None:
        return np.asarray(rgb, np.float32).copy(), {
            "status": report["status"], "selected_components": [], "removed_rms": 0.0
        }
    h, w = model["shape"]
    correction = np.zeros((h, w, 3), np.float32)
    selected = []
    for candidate in model["candidates"]:
        family = tuple(candidate.get("family") or ())
        if family not in AXIS_FAMILIES:
            continue
        fx, fy = candidate["frequency"]
        start, stop = candidate["columns"]
        cosine, sine = periodic._carrier_basis(fx, fy, candidate["origin_phase"], h, w)
        channels = []
        for c in range(3):
            values = model["coefficients"][start:stop, c]
            if not np.any(values):
                continue
            carrier = values[0] * cosine
            if sine is not None:
                carrier = carrier + values[1] * sine
            if model["spatial_gain"]:
                carrier *= periodic._slow_gain(model, candidate, c)
            correction[..., c] += carrier
            channels.append(c)
        selected.append({"family": list(family), "frequency": [float(fx), float(fy)],
                         "active_channels": channels})

    if not selected:
        return np.asarray(rgb, np.float32).copy(), {
            "status": "no_axis_component", "selected_components": [], "removed_rms": 0.0
        }
    protection = 1 - ndimage.gaussian_filter(model["clipped"].astype(np.float32), 2)
    protection[model["clipped"]] = 0
    correction *= protection[..., None]
    removed = periodic._rgb(correction) * np.float32(strength)
    result = np.asarray(rgb, np.float32) - removed
    return result.astype(np.float32, copy=False), {
        "status": "detected", "selected_components": selected,
        "removed_rms": float(np.sqrt(np.mean(removed ** 2))),
        "mean_abs_removed_dn": float(np.mean(np.abs(removed))),
        "strength": strength,
    }


def restore_experimental(loaded, *, delta_gain: float = 0.5,
                         periodic_strength: float = 1.0,
                         resampler: str = "zscale"):
    """Return custom 0..255 float RGB + processing report.

    loaded is io_utils.LoadedImage. Exactly matching the earlier saved
    8-bit PNG-based experiment requires quantization before the Periodic pass;
    here we deliberately keep floating-point data until final save.
    """
    if not 0 <= delta_gain <= 2:
        raise ValueError("delta_gain must be in [0,2]")
    if not 0 <= periodic_strength <= 2:
        raise ValueError("periodic_strength must be in [0,2]")
    if resampler not in ("zscale", "opencv"):
        raise ValueError("resampler must be zscale or opencv")
    original = np.asarray(loaded.rgb, np.float32)
    h, w = original.shape[:2]

    def scaled(image: np.ndarray, target_w: int, target_h: int, kernel: str):
        return resize_rgb(image, target_w, target_h,
                          MultiscaleOptions(resampler=resampler, kernel=kernel))

    enlarged = scaled(original, w * 2, h * 2, "spline36")
    try:
        # A float32 image with bit_depth=16 ensures restore_array does not
        # prematurely round or quantize the intermediate correction.
        input_2x = replace(loaded, rgb=enlarged, size=(w*2, h*2), bit_depth=16)
        restored_2x, _, _ = restore_array(input_2x, preset="balanced", bit_depth=16)
        restored_down = scaled(restored_2x, w, h, "lanczos")
        # Identical resampling path without the restoration stage.
        unprocessed_down = scaled(enlarged, w, h, "lanczos")
        correction = restored_down - unprocessed_down
    finally:
        del enlarged

    pre_periodic = np.clip(original + delta_gain * correction, 0, 255)
    del correction, restored_2x, restored_down, unprocessed_down

    custom, periodic_report = axis_only_periodic(pre_periodic,
                                                 strength=periodic_strength)
    return np.clip(custom, 0, 255), {
        "method": "2x_balanced_delta_plus_axis_only_periodic",
        "scale_up": "2x Spline36", "scale_down": "1x Lanczos",
        "delta_gain": delta_gain,
        "periodic_strength": periodic_strength,
        "resampler": resampler,
        "axis_periodic": periodic_report,
        "note": ("Alternative correction applied to Original, not an additional "
                 "pass after Balanced; float intermediate may differ slightly "
                 "from saved 8-bit prior diagnostics."),
    }
