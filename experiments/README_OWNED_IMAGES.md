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
