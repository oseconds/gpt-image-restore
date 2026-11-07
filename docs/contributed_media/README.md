# User-contributed before/after examples

This directory is reserved for **new images contributed with permission to
publish**, not images copied from the upstream project's docs/media or
docs/previews. The upstream example images have separate rights restrictions.

Generate evaluation images from inputs you created or have authorization to
redistribute, using:

    python experiments/compare_owned_images.py "/path/to/new-images" \
      --publish --source-model "GPT Image" --credit "0seconds" \
      --license demo-only

Only this explicit opt-in creates files under docs/contributed_media and writes
docs/showcase.contributed.json. The GitHub Pages viewer can display those files
at ?contributed=1, with aligned Original/Balanced/Experimental A/B controls and
a common original-derived Contrast Boost. Default processing instead writes
ignored local-only test files.

The source model, credit, and rights status for each sample are stored in its
manifest entry and report.json.

Media rights are **per contributed sample**, independent of the source code's
AGPL-3.0-or-later license:

- demo-only: contributor permits the repository and associated PR / showcase
  to publicly display the image and processed comparison, but does not grant
  general reuse rights
- CC-BY-4.0: contributor opts into CC BY 4.0 for contributed image material
- CC0-1.0: contributor dedicates contributed image material to CC0 to the
  extent they are authorized to do so

The contributor must verify the rights of the input and generated outputs,
including any third-party materials, trademarks, people or likenesses, and
provider terms. Do not publish private photos or images for which publication
permission is unclear. Do not include original prompts, local absolute file
paths, or metadata containing credentials.

The PR should contain a small representative set (roughly 3-5 inputs, each
with the exact original, Balanced, Experimental results), rather than an
undifferentiated large dataset. PNGs are lossless to avoid obscuring small
periodic and edge differences. Review repository size before commit.
