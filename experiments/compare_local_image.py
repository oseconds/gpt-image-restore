"""Local-only multiscale A/B image generator; never publishes input or outputs."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multiscale_restore import MultiscaleOptions, multiscale_fusion, repo_scale_denoiser
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import restore_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="A user-owned or otherwise permitted input image")
    parser.add_argument("--name", help="Display name in the local comparison viewer")
    parser.add_argument("--resampler", choices=("opencv", "zscale"), default="opencv",
                        help="opencv works without FFmpeg; zscale uses true Spline36")
    parser.add_argument("--kernel", choices=("spline36", "lanczos"), default="spline36")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.5, 0.25])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.6, 0.4])
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    slug = re.sub(r"[^a-z0-9_-]+", "-", image_path.stem.lower()).strip("-")[:60] or "sample"
    docs = ROOT / "docs"
    target = docs / "local_media" / slug
    target.mkdir(parents=True, exist_ok=True)
    loaded = load_image(image_path)

    print("Restoring balanced baseline...", flush=True)
    balanced, _, _ = restore_array(loaded, preset="balanced", bit_depth=16)
    print("Restoring strong reference...", flush=True)
    strong, _, _ = restore_array(loaded, preset="strong", bit_depth=16)

    options = MultiscaleOptions(scales=tuple(args.scales),
                                scale_strengths=tuple(args.strengths),
                                resampler=args.resampler,
                                kernel=args.kernel)
    print("Restoring multiscale...", flush=True)
    multiscale, report, _ = multiscale_fusion(balanced, repo_scale_denoiser, options)

    variants = {
        "original": loaded.rgb,
        "balanced": balanced,
        "strong": strong,
        "multiscale": multiscale,
    }
    for key, pixels in variants.items():
        save_image(target / f"{key}.png", pixels, loaded.alpha,
                   image_format="png", bit_depth=8,
                   icc_profile=loaded.icc_profile)

    item = {
        "id": slug, "title": args.name or image_path.stem,
        "width": loaded.size[0], "height": loaded.size[1],
        "images": {key: f"local_media/{slug}/{key}.png" for key in variants},
        "preview": f"local_media/{slug}/original.png",
        "view": [0, 0, loaded.size[0], loaded.size[1]],
    }
    manifest_path = docs / "showcase.local.json"
    manifest = {"items": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"] = [x for x in manifest.get("items", []) if x.get("id") != slug] + [item]
    manifest["files"] = [path for x in manifest["items"] for path in x["images"].values()]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Local A/B images: {target}")
    print("Run: python3 -m http.server 8000 --directory docs")
    print("Then open: http://localhost:8000/?local=1")
    print("Local-only outputs are excluded by .gitignore.")


if __name__ == "__main__":
    main()
