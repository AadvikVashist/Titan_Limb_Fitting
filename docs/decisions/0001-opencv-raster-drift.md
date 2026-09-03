# OpenCV radial-line drift

Status: open scientific review

## Finding

The saved first-band, north-facing radial line for cube `C1477456872_1` has 42
pixels. The same legacy method under OpenCV 4.14 has those 42 pixels plus one:

- Current OpenCV only: row 38, column 30.
- Shared: all 42 saved pixels.

The cube center, line angle, scale, thickness, and rounding rule match the old
code. This points to a change in OpenCV's line raster output across package
versions.

## Rule

Keep the saved profiles as the source of record. Do not replace them with new
profiles until we test this drift across the reference cubes and approve one
fixed pixel-selection rule. The real-data test keeps the known difference
visible.
