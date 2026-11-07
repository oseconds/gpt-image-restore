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
The comparison includes thresholded mesh candidates and **raw peak score** maps;
`peak_score_1x.png` stays informative even if the thresholded candidate mask is black.
Use `--threshold 2.35` (default) or test a stricter value; a brighter score
is *not* proof that the pattern is an artifact.
This is diagnostic analysis only, not a new restoration algorithm. Confidence
maps can falsely identify legitimate periodic textures. Outputs remain local
and excluded from Git commits by `.gitignore`; do not redistribute the
upstream sample images or their derivatives.

## Frequency-specific mesh diagnostics (no restoration)

The older thresholded mesh candidate maps can mistake directional hair strands
for a grid. To inspect *which original-pixel wavelengths* show local FFT peaks,
run this on the checked-out source images:

```sh
python experiments/mesh_frequency_probe.py --sample portrait --roi 555 100 330 280
open docs/local_media/frequency-portrait/frequency_bands.png
```

The figure uses 1.0x, 0.75x, and 0.5x versions and separates candidate
2-4, 4-8, 8-16, 16-32 **original-pixel** periodicity. Unresolvable
downsampled bands are marked rather than misinterpreted as no artifacts.
Outputs and `stats.json` remain gitignored under `docs/local_media/`.
This is a diagnostic spectral-peak comparison, **not** proof of an artifact;
hair and legitimate textures can also produce peaks. The resampling here
uses OpenCV INTER_AREA; it is not a Spline36 comparison.
