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

`selected-fits-summary.json` records the row counts and schema from converting
all 29 saved selected-fit pickles. The generated Parquet file stays under the
ignored `artifacts/` directory.

`profiles-summary.json` records the matching conversion of all saved sorted and
filtered radial profiles. Each Parquet row keeps one profile's pixel rows,
columns, distances, emission angles, and brightness values as typed lists.

`fit-quality-summary.json` records the default structural audit and shows the
effect of several possible R² and coefficient rules without choosing one.

`observations-summary.json` records the selected-cube count, UTC date span, and
typed Parquet fields from the preserved Nantes metadata table.

`transitions-summary.json` records the quality-gated crossing counts, the two
ambiguous north series, and differences from the saved global result.

`asymmetry-summary.json` records the number of paired eligible rows and basic
descriptive values. It does not claim statistical significance.
