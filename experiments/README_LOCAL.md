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

## Fixed-frequency peak comparison (Original vs Balanced vs Strong)

The general FFT-band maps can respond strongly to hair and fail to isolate
skin mesh. On the portrait sample, inspect specific 64px FFT bins
(`16,-16`, `16,16` control, and `31,0` / `0,31` near 2px):

```sh
python experiments/mesh_peak_probe.py --sample portrait --roi 555 100 330 280
open docs/local_media/mesh-peaks-portrait/comparison.png
cat docs/local_media/mesh-peaks-portrait/stats.json
```

All three source images are cropped at **identical native pixel coordinates**,
with the same color scale for each spectrum map. The JSON reports both relative
peak prominence and absolute Fourier amplitude; higher prominence can occur
even if the absolute signal did not increase (because other detail was removed).
This is diagnostic, not a restoration algorithm or proof of a specific artifact
origin. Derived assets are local-only and gitignored.

## Compare Spline36 multiscale ordering (Pre vs Post)

This tests whether the **same** multiscale grime/mottle correction differs
when added before or after existing Balanced, without modifying the preset.

From the repository root on macOS, with FFmpeg `zscale` installed:

```sh
git pull --ff-only
source .venv/bin/activate
python experiments/compare_multiscale_order.py
open docs/local_media/order-portrait/comparison.png
open docs/local_media/order-teapot/comparison.png
```

Defaults: `--scales 0.75 0.5 --strengths 0.6 0.4 --resampler zscale`
(Spline36). To inspect a different native-pixel crop:

```sh
python experiments/compare_multiscale_order.py --sample portrait --roi 555 100 330 280
```

Each sample produces full-resolution `original.png`, `balanced.png`,
`pre.png`, `post.png`, a two-row `comparison.png` with native crops and
12x absolute differences, and `report.json` with pixel-change metrics.
**Pre**: Original > Multiscale > Balanced.
**Post**: Original > Balanced > Multiscale.
Both multiscale passes reuse the same `repo_scale_denoiser`, which runs on
reduced frames; they upsample *correction residuals*, not denoised frames.
Inspect visual quality: pixel difference is not a quality metric.

All assets stay in gitignored `docs/local_media/`. Upstream sample media
are not licensed for redistribution: do not publish their derivatives.

## Lanczos / Spline36 0.5x and 2x roundtrip experiment

Compare changes from **resampling itself**, separate from scaled restoration.
Test the Lanczos half-resolution / Spline36 back-to-native suggestion, as
well as Spline36 2x / Lanczos back-to-native. Both can be prepended to
the original Balanced preset without modifying the restoration pipeline.

```sh
python experiments/compare_resampler_paths.py --sample portrait
open docs/local_media/resampler-portrait/comparison.png
python experiments/compare_resampler_paths.py --sample teapot
open docs/local_media/resampler-teapot/comparison.png
```

The half-size roundtrip defaults to a 0.25 blend with the original BEFORE
Balanced. Try `--half-blend 1` to see the full, much stronger low-pass result.
The 2x roundtrip is compared as a control. To additionally run the **full
Balanced algorithm at 2x resolution** (significantly slower, more memory):

```sh
python experiments/compare_resampler_paths.py --sample portrait --balanced-at-2x
```

Outputs: full-size PNG variants, a native-coordinate `comparison.png`, and
`report.json`. The 2x test reports both its direct downscaled restoration
and the **control-subtracted 2x filter delta**, isolating changes from
restoration instead of from resampling itself. This is a hypothesis test;
loss of fine hair/skin texture is a likely side effect of 0.5x roundtrip.
All generated demo derivatives stay local in the ignored directory.
