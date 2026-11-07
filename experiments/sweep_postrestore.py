"""Post-restoration detail rescue, without changing Balanced / Strong internals.

Uses already generated, user-owned showcase images. No model reruns.
Formula:
  repaired = base + strength * text_detail_mask * (donor - base)
Where base is Strong or Balanced, donor is Original / Balanced / Experimental,
and the mask comes from the ORIGINAL (not the damaged result).

This is an exploratory selective reversal of a restoration residual, not an
AI reconstruction. Its main failure mode is reintroducing source noise where
text strokes are falsely detected. Inspect post_mask.png and contrast-boosted
A/B comparisons before drawing conclusions.

Examples:
  python experiments/sweep_postrestore.py test_inputs/design-poster.png
  python experiments/sweep_postrestore.py test_inputs/design-poster.png \
      --target strong --donors original balanced experimental --strengths 0.5 0.8 1
  python experiments/sweep_postrestore.py test_inputs/design-poster.png \
      --target strong --donors original --roi 455 580 560 640

  python3 -m http.server 8000 --directory docs
  # http://localhost:8000/?tests=1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_owned_images import (
    PRIVATE_MANIFEST, PRIVATE_MEDIA, _change, _slug, collect_images,
)
from gptimg_restore.glyphring import _text_rows
from gptimg_restore.io_utils import load_image, save_image


def _source_text_mask(original: np.ndarray, *, detail_min: float = 9.0,
                      detail_range: float = 22.0, expand: int = 3,
                      roi: tuple[int, int, int, int] | None = None):
    """Construct a conservative, feathered glyph protection map from Original.

    The upstream text-row detector supplies semantic *locations*. A local
    median-residual threshold limits blending to high-contrast letter pixels
    and their immediate boundaries, not every background pixel in the row.
    """
    h, w = original.shape[:2]
    src = np.clip(np.rint(original), 0, 255).astype(np.uint8)
    row_support = np.zeros((h, w), np.uint8)

    # Detect both dark-on-light and light-on-dark text from Original, not Strong.
    rows = _text_rows(np.asarray(original, np.float32))
    inverted_rows = _text_rows(255.0 - np.asarray(original, np.float32))
    matched = []
    for row in [*rows, *inverted_rows]:
        left, top, right, bottom = [int(v) for v in row[:4]]
        left, right = max(0, left - 2), min(w, right + 2)
        top, bottom = max(0, top - 2), min(h, bottom + 2)
        if right <= left or bottom <= top:
            continue
        row_support[top:bottom, left:right] = 255
        matched.append([left, top, right, bottom])

    if roi is not None:
        x, y, rw, rh = roi
        if x < 0 or y < 0 or rw <= 0 or rh <= 0 or x + rw > w or y + rh > h:
            raise ValueError(f"Invalid --roi {roi} for {w}x{h}")
        # A manually supplied ROI intentionally adds text support in an area
        # where the detector misses a small/light typeface, but still requires
        # high-contrast source details for per-pixel protection.
        row_support[y:y+rh, x:x+rw] = 255

    luma = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    background = cv2.medianBlur(luma, 17).astype(np.float32)
    detail = np.abs(luma.astype(np.float32) - background)
    confidence = np.clip((detail - detail_min) / detail_range, 0, 1).astype(np.float32)
    # Recover thin stroke boundaries and single-pixel gaps. Use soft feather
    # to avoid sharp rectangular edges in the recombined image.
    kernel_size = max(1, int(expand) | 1)
    if kernel_size > 1:
        confidence = cv2.dilate(confidence, np.ones((kernel_size, kernel_size), np.uint8))
    confidence = cv2.GaussianBlur(confidence, (0, 0), 0.8)
    support = cv2.GaussianBlur(row_support.astype(np.float32) / 255, (0, 0), 1.0)
    mask = np.clip(confidence * support, 0, 1)
    return mask, {
        "detected_text_rows": len(matched),
        "detected_boxes": matched[:300],
        "mask_mean": round(float(np.mean(mask)), 6),
        "mask_over_half_pct": round(100 * float(np.mean(mask > 0.5)), 4),
        "roi_xywh": list(roi) if roi is not None else None,
        "detail_min": detail_min, "detail_range": detail_range,
        "dilation_kernel": kernel_size,
    }


def _donor_residual_gate(base: np.ndarray, donor: np.ndarray,
                         min_change: float) -> np.ndarray:
    """Avoid unnecessarily blending where the two results already agree."""
    diff = np.max(np.abs(donor - base), axis=2)
    return np.clip((diff - min_change) / max(1.0, min_change * 2.0), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="User-owned image(s) already registered in ?tests=1")
    parser.add_argument("--target", choices=("strong", "balanced"), default="strong")
    parser.add_argument("--donors", nargs="+",
                        choices=("original", "balanced", "experimental"),
                        default=("original", "balanced", "experimental"))
    parser.add_argument("--strengths", nargs="+", type=float,
                        default=(0.50, 0.80, 1.00))
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Optional extra text support rectangle in source pixels")
    parser.add_argument("--detail-min", type=float, default=9.0)
    parser.add_argument("--detail-range", type=float, default=22.0)
    parser.add_argument("--expand", type=int, default=3)
    parser.add_argument("--min-change", type=float, default=5.0,
                        help="Minimum source/donor correction before post blending")
    args = parser.parse_args()
    if not PRIVATE_MANIFEST.is_file():
        parser.error("Run compare_owned_images.py first to create the local showcase.")
    if args.target in args.donors:
        args.donors = [d for d in args.donors if d != args.target]
    if not args.donors:
        parser.error("At least one donor must differ from the target.")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in args.strengths):
        parser.error("--strengths must be within [0,1]")
    if any(not np.isfinite(v) or v < 0 for v in
           (args.detail_min, args.detail_range, args.min_change)):
        parser.error("Detail and minimum-change thresholds must be nonnegative.")
    if args.detail_range <= 0 or not 1 <= args.expand <= 15:
        parser.error("--detail-range must be positive and --expand within [1,15]")
    if args.roi and len(args.inputs) != 1:
        parser.error("--roi requires a single input image.")

    manifest = json.loads(PRIVATE_MANIFEST.read_text(encoding="utf-8"))
    lookup = {item["id"]: i for i, item in enumerate(manifest["items"])}
    for path in collect_images(args.inputs):
        slug = _slug(path)
        if slug not in lookup:
            parser.error(f"{slug} is not in the local showcase. Generate A/B first.")
        target = PRIVATE_MEDIA / slug
        item = dict(manifest["items"][lookup[slug]])
        original_loaded = load_image(path)
        if original_loaded.size != (item["width"], item["height"]):
            parser.error(f"Image dimensions changed for {path.name}.")

        base_path = target / f"{args.target}.png"
        if not base_path.is_file():
            parser.error(f"Missing {base_path}. Add Strong with --add-strong first.")
        base = np.asarray(Image.open(base_path).convert("RGB"), np.float32)
        original = np.asarray(original_loaded.rgb, np.float32)
        mask, mask_report = _source_text_mask(
            original, detail_min=args.detail_min, detail_range=args.detail_range,
            expand=args.expand, roi=tuple(args.roi) if args.roi else None)
        Image.fromarray(np.uint8(np.rint(mask * 255)), "L").save(target / "post_mask.png")
        if mask_report["detected_text_rows"] == 0 and not args.roi:
            print(f"[{slug}] WARNING: no text rows detected; inspect post_mask.png")
        print(f"[{slug}] source glyph mask: {mask_report['detected_text_rows']} text rows, "
              f"{mask_report['mask_over_half_pct']:.2f}% strong coverage", flush=True)

        item["images"] = dict(item["images"])
        item["labels"] = dict(item.get("labels", {}))
        report_path = target / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run_info = {
            "base": args.target, "mask_from": "Original",
            "mask": mask_report,
            "note": "Selective masked residual blending. No base/donor model reruns.",
            "variants": {},
        }
        for donor_name in args.donors:
            donor_path = target / f"{donor_name}.png"
            if donor_name == "original":
                donor = original
            elif donor_path.is_file():
                donor = np.asarray(Image.open(donor_path).convert("RGB"), np.float32)
            else:
                parser.error(f"Missing donor {donor_path}")

            # Apply a soft confidence floor: only restore meaningful differences
            # and do not blend noise into pixels where Strong didn't remove it.
            gate = _donor_residual_gate(base, donor, args.min_change)
            weight = mask * gate
            for amount in args.strengths:
                key = f"post_{args.target[0]}_{donor_name[0]}_{int(round(amount * 100)):03d}"
                output = np.clip(base + amount * weight[..., None] * (donor - base),
                                 0, 255)
                save_image(target / f"{key}.png", output, original_loaded.alpha,
                           image_format="png", bit_depth=8,
                           icc_profile=original_loaded.icc_profile)
                item["images"][key] = f"test_media/{slug}/{key}.png"
                item["labels"][key] = f"{args.target.title()} + {donor_name.title()} strokes {amount:g}"
                run_info["variants"][key] = {
                    "donor": donor_name, "strength": amount,
                    "mean_effective_mask": round(float(np.mean(weight * amount)), 6),
                    "mean_change_from_base_dn": _change(output, base),
                    "mean_change_from_original_dn": _change(output, original),
                }
                print(f"  {key}: {item['labels'][key]}", flush=True)

        report.setdefault("postrestore_experiments", []).append(run_info)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        manifest["items"][lookup[slug]] = item
        temp = PRIVATE_MANIFEST.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        temp.replace(PRIVATE_MANIFEST)

    print("\nWeb comparison: http://localhost:8000/?tests=1")
    print("Compare Strong against the post variants, with contrast boost on and off.")
    print("Inspect docs/test_media/<sample>/post_mask.png for false text positives.")
    print("No input, Balanced, Strong, or Experimental output was overwritten.")


if __name__ == "__main__":
    main()
