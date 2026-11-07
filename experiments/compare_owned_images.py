"""Create a private A/B web showcase from one or more user-owned images.

No images from the upstream docs/media directory are used. All generated
pictures and the test manifest stay ignored by Git.

Example:
  python experiments/compare_owned_images.py ~/Pictures/gpt-tests
  python experiments/compare_owned_images.py a.png b.webp c.jpg --reset
  python3 -m http.server 8000 --directory docs
  # http://localhost:8000/?tests=1

The right side uses the experimental 2x Delta 0.50 + X/Y 2px correction.
This is an alternative to Balanced, not a second pass on Balanced.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
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
MANIFEST = ROOT / "docs" / "showcase.tests.json"
FORBIDDEN_MEDIA = (ROOT / "docs" / "media").resolve()


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
        if path.is_relative_to(FORBIDDEN_MEDIA):
            raise ValueError(f"Refusing to use upstream docs/media sample: {path}")
    if not unique:
        raise ValueError(f"No input images found (supported: {sorted(EXTENSIONS)})")
    return unique


def _slug(path: Path) -> str:
    name = re.sub(r"[^a-z0-9_-]+", "-", path.stem.lower()).strip("-")[:48] or "image"
    # Two different source folders can contain the same basename.
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    return f"test-{name}-{digest}"


def _change(a: np.ndarray, b: np.ndarray) -> dict:
    difference = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    return {"mean_abs_dn": round(float(difference.mean()), 5),
            "max_abs_dn": round(float(difference.max()), 3)}


def process_one(path: Path, args) -> dict:
    slug = _slug(path)
    target = PRIVATE_MEDIA / slug
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
    for version, pixels in outputs.items():
        save_image(target / f"{version}.png", pixels, loaded.alpha,
                   image_format="png", bit_depth=8, icc_profile=loaded.icc_profile)
    # Report is local only. Do not expose the source absolute path to the web manifest.
    report = {
        "source_name": path.name, "width": width, "height": height,
        "bit_depth": loaded.bit_depth, "view_xywh": view,
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
    return {
        "id": slug,
        "title": args.name or path.stem,
        "width": width,
        "height": height,
        "images": {key: f"test_media/{slug}/{key}.png" for key in outputs},
        "preview": f"test_media/{slug}/original.png",
        "view": view,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Image files or folders; folders use only their direct image children")
    parser.add_argument("--reset", action="store_true",
                        help="Start a fresh test showcase instead of retaining earlier entries")
    parser.add_argument("--name", help="Display name, for one input image only")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"),
                        help="Native-pixel initial zoom region, for one input image only")
    parser.add_argument("--resampler", choices=("zscale", "opencv"), default="zscale",
                        help="zscale=FFmpeg Spline36/Lanczos; opencv is NOT equivalent")
    parser.add_argument("--delta-gain", type=float, default=0.5)
    parser.add_argument("--periodic-strength", type=float, default=1.0)
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if (args.name or args.roi is not None) and len(images) != 1:
        parser.error("--name and --roi require one input image")
    if not 0 <= args.delta_gain <= 2:
        parser.error("--delta-gain must be within [0,2]")
    if not 0 <= args.periodic_strength <= 2:
        parser.error("--periodic-strength must be within [0,2]")

    existing = []
    if not args.reset and MANIFEST.is_file():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8")).get("items", [])
    items_by_id = {item["id"]: item for item in existing}

    for path in images:
        item = process_one(path, args)
        items_by_id[item["id"]] = item
        # Persist successfully completed work even if a later image fails.
        payload = {"items": list(items_by_id.values())}
        temp = MANIFEST.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(MANIFEST)
        gc.collect()

    print(f"\n{len(images)} image(s) processed. Test manifest: {MANIFEST}")
    print("Run: python3 -m http.server 8000 --directory docs")
    print("Open: http://localhost:8000/?tests=1")
    print("Generated images and manifest are gitignored and must not be committed.")


if __name__ == "__main__":
    main()
