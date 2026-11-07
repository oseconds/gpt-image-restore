# Multiscale experiments — local only

This branch starts directly from `main` and contains code-only changes.
It intentionally does not contain any derived or regenerated `docs/media` assets.

The upstream `docs/media` examples are **not licensed for reuse**. Use images
you own or have permission to process when generating new comparisons.

## On macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/compare_local_image.py "/path/to/your-image.png" --resampler opencv
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000/?local=1 to compare your image with A=Balanced and
B=Multiscale. The normal http://localhost:8000/ remains the original sample
showcase, now with independently selectable A/B versions.

Switch to `--resampler zscale` after repairing FFmpeg for actual Spline36.
OpenCV fallback is not Spline36.

All local generated images, local manifests, and reports are .gitignored. Run
`git status --short` before pushing. No automatic GitHub Actions workflows
are included in this experimental branch.

## Detect mesh candidates directly from the repository samples

This command crops the native `docs/media/portrait/original.webp` pixels using
the ROI defined in `docs/showcase.json`, and crops `balanced.webp` and
`strong.webp` at the exact same source coordinates:

```sh
python experiments/mesh_probe_docs.py --sample portrait --resampler zscale
```

Override the native-pixel ROI when necessary:

```sh
python experiments/mesh_probe_docs.py --sample portrait --roi 472 125 510 340 --resampler zscale
```

Inspect `docs/local_media/mesh-portrait/comparison.png` and `stats.json`.
This is diagnostic analysis only, not a new restoration algorithm. Confidence
maps can falsely identify legitimate periodic textures. Outputs remain local
and excluded from Git commits by `.gitignore`; do not redistribute the
upstream sample images or their derivatives.
