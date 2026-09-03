# OpenCV radial-line drift

Status: accepted 3 September 2026

## Finding

The saved first-band, north-facing radial line for cube `C1477456872_1` has 42
pixels. The same legacy method under OpenCV 4.14 has those 42 pixels plus one:

- Current OpenCV only: row 38, column 30.
- Shared: all 42 saved pixels.

The cube center, line angle, scale, thickness, and rounding rule match the old
code. This points to a change in OpenCV's line raster output across package
versions.

## Decision

The full current-stack run produced 20,416 selected profiles. Of these, 7,613
match the saved pixel and value arrays exactly and 12,312 have the same point
count. OpenCV 4.9 reproduces 20,064 saved profiles when paired with the saved
centers and angles. This confirms that most drift comes from the line raster,
not from the Titan geometry.

We accept the current OpenCV line sets as the new raw result. The saved products
remain the comparison baseline, not the source of record for new runs. Every
raw release must keep its OpenCV version in the lock file and save the complete
row-level drift report. This decision follows the project owner's direction on
3 September 2026.
