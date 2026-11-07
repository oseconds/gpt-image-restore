# User-owned A/B showcase for the experimental restoration

This workflow is for **new images the contributor is authorized to share**.
It does not use the original project's restrictive examples in docs/media.

## Purpose

Show whether the experimental method protects legitimate fine structure while
suppressing the periodic noise measured in GPT-generated images. Inspect
multiple subjects and negative cases before proposing any upstream change.

A = existing upstream Balanced preset.
B = experimental Original + 0.5 * 2x Balanced filter delta + native 2px X/Y
periodic correction. This is an alternative restoration, **not** Balanced
followed by an extra post-process. It uses FFmpeg zscale Spline36 for upscaling
and Lanczos for downscaling. The resampling-only control is subtracted.

No modification to the upstream preset or pipeline is required.

## Generate and view local-only tests

Prepare a folder with your own PNG / WebP / JPEG files. Then:

```sh
source .venv/bin/activate
python experiments/compare_owned_images.py ~/Pictures/gpt-restore-tests --reset
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000/?tests=1

The existing slider, synchronized pan/zoom, and original-derived Contrast
Boost work unchanged. Defaults: A=Balanced, B=Experimental. Original is also
available from the A/B dropdowns. Inputs and results are excluded from Git.
The MacPorts FFmpeg build must include the zscale filter. The 2x pass is much
heavier than Balanced alone. Avoid processing very large inputs at first.

## Optional Strong baseline on already processed images

The original project's README shows a poster example using Strong, not
Balanced. To make a like-for-like comparison without rerunning the expensive
2x restoration, add Strong to one existing local example:

```sh
python experiments/compare_owned_images.py test_inputs/design-poster.png --add-strong
```

Or append Strong to every sample already in the test manifest:

```sh
python experiments/compare_owned_images.py test_inputs --add-strong
```

This only runs restore_array(..., preset="strong") at the native resolution,
writes strong.png, and updates the manifest/report. It does not change
Original, Balanced, or Experimental. The web comparison dropdown will then
offer all four versions. Do not add --reset in this mode.

## Pre-clean -> Strong strength sweep (local experiment)

To test whether pre-cleaning the image BEFORE the Strong preset reduces
text stroke damage without losing noise suppression, use the **same** input
previously processed by compare_owned_images.py:

```sh
git pull --ff-only
source .venv/bin/activate
python experiments/sweep_prestrong.py test_inputs/design-poster.png
```

Open http://localhost:8000/?tests=1 and choose Strong on A, any pre-clean
variant on B. The existing original/Strong/Balanced/Experimental outputs
remain unchanged. The tool appends images to the current local test sample
and adds dropdown options without requiring a new HTML page.

Default sweep: half-roundtrip Lanczos down 0.5x -> Spline36 up 1x with
blend gains 0.05, 0.10, 0.25; plus 2x Spline36 -> Balanced @2x -> Lanczos
down to 1x, subtracting the pure roundtrip and applying delta gains 0.25,
0.50, 1.00. Each delta variant then gets native X/Y 2px periodic correction
(strength 1.0) before Strong. Strong uses the same upstream preset settings
as the baseline and is applied to all pre-cleaned images independently.

For a shorter initial run (half 0.10, delta 0.50 and 1.00 only):

```sh
python experiments/sweep_prestrong.py test_inputs/design-poster.png \
  --half 0.10 --delta 0.50 1.00
```

For a delta-only test, omit the half gains with `--half`. To isolate delta
without the native periodic stage, add `--xy-strength 0`.

The expensive 2x Balanced operation is shared across all gains in one
invocation (and the half-roundtrip is also shared), but each variant has
a separate full Strong pass. Timings and axis detection information are
written to the test sample's existing report.json under prestrong_sweep.
The program uses floating-point intermediates, saving 8-bit PNG for the
same web viewer. It does NOT automatically reduce Strong's destructive
stage strengths. This is a hypothesis test, not a guaranteed improvement.
These new local outputs are gitignored; only approved examples should
later be selected for the public PR.

## Post-Strong / post-Balanced: recover original glyph strokes (no rerun)

Instead of trying to change what Strong detects before processing, this
experiment works **after** the Strong or Balanced result already exists.
It computes a soft protection mask from the Original using the repository's
existing text-row detector plus high-contrast source strokes, then blends
back only masked differences from one donor. The Strong background remains
otherwise untouched. This tests whether aggressive cleanup and fine text
preservation can be separated without editing the Strong stage code.

```sh
git pull --ff-only
source .venv/bin/activate
python experiments/sweep_postrestore.py test_inputs/design-poster.png
```

The default donor sources are Original, Balanced and Experimental;
each is tested at strengths 0.50, 0.80 and 1.00, producing nine variants
alongside the existing Strong baseline in the same web comparison:

  http://localhost:8000/?tests=1

To test Balanced as the base instead of Strong, or limit donors/strengths:

```sh
python experiments/sweep_postrestore.py test_inputs/design-poster.png \
  --target balanced --donors original --strengths 0.5 0.8 1
```

If tiny timetable text is missing from automated text detection, supply
an extra high-contrast source-stroke support region in native pixels:

```sh
python experiments/sweep_postrestore.py test_inputs/design-poster.png \
  --target strong --donors original balanced --roi 455 580 560 640
```

The optional --roi adds protection support rather than cropping processing;
the per-pixel source-detail mask still gates it. Inspect the local
docs/test_media/design-poster/post_mask.png for false positives, especially
the poster's dotted procedural mesh. The algorithm is **not** automatic
content reconstruction: it literally reintroduces original/baseline pixels,
including any original noise at falsely protected positions. Never infer
improvement from a single poster or from lower pixel difference alone.

No Strong, Balanced or Experimental reconstruction is rerun or overwritten,
and all new outputs stay gitignored. The small comparisons are exploratory
and separate from the eventual clean PR implementation.
## Publish a selected set for the PR (explicit opt-in)

Once rights and quality are reviewed, process only the 3-5 new inputs suitable
for public display, with the author's credit and generator metadata:

```sh
python experiments/compare_owned_images.py ~/Pictures/gpt-restore-pr \
  --publish --reset \
  --source-model "GPT Image" --credit "0seconds" --license demo-only
```

Open http://localhost:8000/?contributed=1

The public mode creates:

- docs/showcase.contributed.json
- docs/contributed_media/<sample>/original.png
- docs/contributed_media/<sample>/balanced.png
- docs/contributed_media/<sample>/experimental.png
- docs/contributed_media/<sample>/report.json

The source credit/model/license travel with the item in the manifest.
For PNG inputs without EXIF orientation changes, the script copies the
original bytes unchanged into original.png, preserving embedded C2PA caBX
metadata. Balanced and Experimental are newly encoded derivatives and do
not inherit a valid original C2PA signature. The generic softwareAgent
version "gpt-image" is NOT proof of a specific GPT Image 2.6 revision;
record the model selected at generation time separately, if known.
The code license and example image rights are separate. demo-only authorizes
hosting/PR display but not unrestricted reuse. CC BY 4.0 or CC0 are alternatives
only if the contributor can grant those rights.

Before committing, inspect git status, check the new files and total size,
and ensure no personal details, private images, or upstream sample derivatives
are present. PNG is used because lossy WebP or JPEG can hide or introduce the
very artifacts being evaluated. --reset resets the manifest but does not
delete previously published output directories.

## PR review checklist

- Source and share permissions are recorded for every input.
- Each sample has Original, Balanced, Experimental at identical dimensions.
- Review with Contrast Boost both on and off and at native 100% zoom.
- Include textured objects, hairline/fine edges, flat backgrounds, and a case
  where the experimental method is not better.
- Include per-image processing times, periodic detection status and
  observed limitations; do not equate high-frequency energy with true detail.
- Keep sample size modest and do not include the original docs/media set.
- Extract a **clean PR branch from upstream main** containing only the
  experiment's required implementation, viewer support, documentation, tests,
  and authorized new sample images. Do not PR the entire explorative branch.

This is a validation tool; it is not yet proof the experimental approach
improves restoration across arbitrary sources.
