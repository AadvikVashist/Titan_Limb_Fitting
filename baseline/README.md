# Preserved baseline

- Source commit: `b38d0e9`
- Source branch: `master`
- Data files: 4,533
- Data bytes: 6,650,915,801
- Manifest schema: 1
- Manifest validation: valid

`data-manifest.json` records path, size, modification time, and SHA-256 for every
file under the original `data/` directory.

The source checkout had material uncommitted changes before modernization:

- A rebuilt `LaTex/limb_paper.pdf`.
- Sixteen deleted tracked SRTC++ diagnostic PNG files.
- An untracked LaTeX quad-figure directory.

The modernization branch starts from the committed source state and does not
include or alter those source-checkout changes.

