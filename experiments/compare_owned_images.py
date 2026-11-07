"""Generate A/B comparisons using new, user-owned images only.

Local by default (gitignored):
  python experiments/compare_owned_images.py ~/Pictures/gpt-tests
  python3 -m http.server 8000 --directory docs
  # http://localhost:8000/?tests=1

Explicit opt-in for new images approved for public sharing:
  python experiments/compare_owned_images.py ~/Pictures/gpt-tests --publish \
      --source-model "GPT Image" --license "demo-only" --credit "0seconds"
  # http://localhost:8000/?contributed=1

Published images and manifest are tracked by Git. Do not use sources from
upstream docs/media: those examples have separate restrictive rights.

Right side is 2x Delta 0.50 + X/Y 2px correction, an alternative to Balanced.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from shutil import copyfile

from PIL import Image
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experimental_restore import restore_experimental
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import restore_array

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PRIVATE_MEDIA = ROOT / "docs" / "test_media"
PUBLIC_MEDIA = ROOT / "docs" / "contributed_media"
PRIVATE_MANIFEST = ROOT / "docs" / "showcase.tests.json"
PUBLIC_MANIFEST = ROOT / "docs" / "showcase.contributed.json"
FORBIDDEN_MEDIA = (ROOT / "docs" / "media").resolve()
FORBIDDEN_PREVIEW = (ROOT / "docs" / "previews").resolve()


def collect_images(inputs: list[Path]) -> list[Path]:
    found = []
    for path in inputs:
        path = path.expanduser().resolve()
        if path.is_dir():
            found.extend(sorted(p for p in path.iterdir()
                                if p.is_file() and p.suffix.lower() in EXTENSIONS))
        elif path.is_file():
            if path.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {path}")
            found.append(path)
        else:
            raise FileNotFoundError(path)
    # No implicit processing of upstream sample media, whose reuse is forbidden.
    unique = list(dict.fromkeys(p.resolve() for p in found))
    for path in unique:
        if path.is_relative_to(FORBIDDEN_MEDIA) or path.is_relative_to(FORBIDDEN_PREVIEW):
            raise ValueError(f"Refusing to use upstream demo sample: {path}")
    if not unique:
        raise ValueError(f"No input images found (supported: {sorted(EXTENSIONS)})")
    return unique


def _slug(path: Path) -> str:
    # Match upstream's tidy sample IDs: teapot/, portrait/, poster/.
    # The filename determines the stable public-facing directory name.
    name = re.sub(r"[^a-z0-9-]+", "-", path.stem.lower()).strip("-")[:48]
    if name:
        return name
    # For non-Latin filenames, avoid collapsing every item into 'image'.
    digest = hashlib.sha256(path.stem.encode("utf-8")).hexdigest()[:8]
    return f"image-{digest}"


def _change(a: np.ndarray, b: np.ndarray) -> dict:
    difference = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    return {"mean_abs_dn": round(float(difference.mean()), 5),
            "max_abs_dn": round(float(difference.max()), 3)}


def process_one(path: Path, args) -> dict:
    slug = _slug(path)
    target = (PUBLIC_MEDIA if args.publish else PRIVATE_MEDIA) / slug
    target.mkdir(parents=True, exist_ok=True)
    loaded = load_image(path)
    width, height = loaded.size
    view = list(args.roi if args.roi is not None else (0, 0, width, height))
    x, y, w, h = view
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
        raise ValueError(f"Invalid native-pixel ROI {view} for {path.name} ({width}x{height})")
    print(f"[{path.name}] Balanced at original resolution...", flush=True)
    started = time.perf_counter()
    balanced, _, _ = restore_array(loaded, preset="balanced", bit_depth=16)
    baseline_seconds = round(time.perf_counter()-started, 3)

    print(f"[{path.name}] Experimental: 2x Delta + X/Y Periodic...", flush=True)
    started = time.perf_counter()
    custom, details = restore_experimental(
        loaded, delta_gain=args.delta_gain,
        periodic_strength=args.periodic_strength,
        resampler=args.resampler)
    custom_seconds = round(time.perf_counter()-started, 3)

    outputs = {"original": loaded.rgb, "balanced": balanced, "experimental": custom}
    # Preserve original PNG bytes (and any embedded C2PA credentials) for the
    # public PR showcase. A re-encoded PNG silently drops its caBX/JUMBF block.
    # When orientation metadata requires display rotation, re-encode to match
    # the processed pixels and report that source credentials were not retained.
    with Image.open(path) as source_metadata:
        orientation = source_metadata.getexif().get(274, 1)
    preserve_original = path.suffix.lower() == ".png" and orientation == 1
    for version, pixels in outputs.items():
        destination = target / f"{version}.png"
        if version == "original" and preserve_original:
            if path != destination.resolve():
                copyfile(path, destination)
        else:
            save_image(destination, pixels, loaded.alpha,
                       image_format="png", bit_depth=8, icc_profile=loaded.icc_profile)
    # Report is local only. Do not expose the source absolute path to the web manifest.
    report = {
        "source_name": path.name, "width": width, "height": height,
        "provenance": {
            "provided_by": args.credit if args.publish else "local user",
            "generator": args.source_model or "unspecified",
            "license": args.license if args.publish else "local-only",
            "publication_permission_confirmed": bool(args.publish),
        },
        "bit_depth": loaded.bit_depth, "view_xywh": view,
        "original_png_bytes_preserved": preserve_original,
        "balanced_elapsed_seconds": baseline_seconds,
        "experimental_elapsed_seconds": custom_seconds,
        "experimental": details,
        "vs_original": {
            "balanced": _change(balanced, loaded.rgb),
            "experimental": _change(custom, loaded.rgb),
        },
        "limits": "Pixel difference is not image quality; 2x doesn't create ground truth.",
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{path.name}] outputs: {target}", flush=True)
    # The existing viewer reads this exact object structure.
    media_dir = "contributed_media" if args.publish else "test_media"
    return {
        "id": slug,
        "title": args.name or path.stem,
        "width": width,
        "height": height,
        "images": {key: f"{media_dir}/{slug}/{key}.png" for key in outputs},
        "preview": f"{media_dir}/{slug}/original.png",
        "view": view,
        "provenance": report["provenance"],
    }


def add_strong_to_existing(path: Path, args, item: dict) -> dict:
    """Add a Strong control WITHOUT rerunning Balanced or the costly 2x pass."""
    slug = _slug(path)
    target = (PUBLIC_MEDIA if args.publish else PRIVATE_MEDIA) / slug
    if not (target / "balanced.png").is_file() or not (target / "experimental.png").is_file():
        raise FileNotFoundError(
            f"Run the normal A/B test first; expected existing outputs in {target}")
    loaded = load_image(path)
    if loaded.size != (item["width"], item["height"]):
        raise ValueError(f"Source dimensions changed for {path.name}; rerun the normal comparison")
    print(f"[{path.name}] Adding Strong preset only...", flush=True)
    started = time.perf_counter()
    strong, stage_reports, _ = restore_array(loaded, preset="strong", bit_depth=16)
    elapsed = round(time.perf_counter() - started, 3)
    save_image(target / "strong.png", strong, loaded.alpha,
               image_format="png", bit_depth=8, icc_profile=loaded.icc_profile)
    item = dict(item)
    item["images"] = dict(item["images"])
    media_dir = "contributed_media" if args.publish else "test_media"
    item["images"]["strong"] = f"{media_dir}/{slug}/strong.png"
    report_path = target / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["strong_elapsed_seconds"] = elapsed
    report.setdefault("vs_original", {})["strong"] = _change(strong, loaded.rgb)
    report["strong_stages"] = {
        name: {"enabled": values.get("enabled", True)}
        for name, values in stage_reports.items()
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[{path.name}] Strong added without rerunning existing comparisons.")
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Image files or folders; folders use only their direct image children")
    parser.add_argument("--reset", action="store_true",
                        help="Start a fresh showcase instead of retaining earlier entries")
    parser.add_argument("--add-strong", action="store_true",
                        help="Add Strong to existing A/B results without rerunning the 2x method")
    parser.add_argument("--publish", action="store_true",
                        help="Opt in to Git-trackable public demo media and contributed showcase")
    parser.add_argument("--source-model",
                        help="Image generator/model used (required when --publish)")
    parser.add_argument("--credit", help="Public source credit/attribution")
    parser.add_argument("--license", choices=("demo-only", "CC-BY-4.0", "CC0-1.0"),
                        help="Rights for NEW images (required when --publish)")
    parser.add_argument("--name", help="Display name, for one input image only")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"),
                        help="Native-pixel initial zoom region, for one input image only")
    parser.add_argument("--resampler", choices=("zscale", "opencv"), default="zscale",
                        help="zscale=FFmpeg Spline36/Lanczos; opencv is NOT equivalent")
    parser.add_argument("--delta-gain", type=float, default=0.5)
    parser.add_argument("--periodic-strength", type=float, default=1.0)
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if args.add_strong and args.reset:
        parser.error("--add-strong cannot be combined with --reset")
    if args.publish and not args.add_strong and (not args.source_model or not args.license):
        parser.error("--publish requires --source-model and --license; only publish images you are authorized to share")
    if args.publish and not args.add_strong and not args.credit:
        parser.error("--publish requires --credit for source attribution")
    manifest_path = PUBLIC_MANIFEST if args.publish else PRIVATE_MANIFEST
    stems = [_slug(path) for path in images]
    if len(stems) != len(set(stems)):
        parser.error("Two images resolve to the same sample ID; rename their files uniquely")
    if (args.name or args.roi is not None) and len(images) != 1:
        parser.error("--name and --roi require one input image")
    if not 0 <= args.delta_gain <= 2:
        parser.error("--delta-gain must be within [0,2]")
    if not 0 <= args.periodic_strength <= 2:
        parser.error("--periodic-strength must be within [0,2]")

    existing = []
    if not args.reset and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("items", [])
    items_by_id = {item["id"]: item for item in existing}

    for path in images:
        slug = _slug(path)
        if args.add_strong:
            if slug not in items_by_id:
                raise ValueError(f"{path.name} missing from existing manifest; run normal A/B first")
            item = add_strong_to_existing(path, args, items_by_id[slug])
        else:
            item = process_one(path, args)
        items_by_id[item["id"]] = item
        # Persist successfully completed work even if a later image fails.
        payload = {"items": list(items_by_id.values())}
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
        gc.collect()

    print(f"\n{len(images)} image(s) processed. Manifest: {manifest_path}")
    print("Run: python3 -m http.server 8000 --directory docs")
    if args.publish:
        print("Open: http://localhost:8000/?contributed=1")
        print("Public media are Git-trackable. Review images, rights, metadata and file sizes before commit.")
    else:
        print("Open: http://localhost:8000/?tests=1")
        print("Local test media and manifest are gitignored.")


if __name__ == "__main__":
    main()
