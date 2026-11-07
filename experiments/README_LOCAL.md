# Multiscale experiments — local only

This branch starts directly from \`main\` and contains code-only changes.
It intentionally does not contain any derived or regenerated \`docs/media\` assets.

The upstream \`docs/media\` examples are **not licensed for reuse**. Use images
you own or have permission to process when generating new comparisons.

## On macOS

\`\`\`sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/compare_local_image.py "/path/to/your-image.png" --resampler opencv
python3 -m http.server 8000 --directory docs
\`\`\`

Open http://localhost:8000/?local=1 to compare your image with A=Balanced and
B=Multiscale. The normal http://localhost:8000/ remains the original sample
showcase, now with independently selectable A/B versions.

Switch to \`--resampler zscale\` after repairing FFmpeg for actual Spline36.
OpenCV fallback is not Spline36.

All local generated images, local manifests, and reports are .gitignored. Run
\`git status --short\` before pushing. No automatic GitHub Actions workflows
are included in this experimental branch.
