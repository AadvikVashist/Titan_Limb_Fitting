# Architecture

The project is moving from stage-sized pickle files and script entry points to a
typed Python package with explicit commands and stable data formats.

## Boundaries

- `io`: reads and writes external formats.
- `processing`: converts VIMS cube data into radial brightness profiles.
- `fitting`: applies limb laws and assigns fit-quality results.
- `analysis`: derives transitions, asymmetry, trust, and seasonal results.
- `simulations`: processes SRTC++ cases and trains comparison models.
- `plotting`: renders accepted result tables without changing them.

The first finished input boundary is `io.legacy`. It converts each saved
selected-fit pickle into one row per cube, band, and hemisphere. Failed legacy
fits remain rows with a clear status and reason. The writer stores the rows as
compressed Parquet for fast Polars scans.

`io.vims` is the only new PyVIMS boundary. It loads cubes on demand, so package
imports do not parse cube data or import PyVIMS. `processing.geometry` contains
the first pure profile functions. See `docs/decisions/0001-opencv-raster-drift.md`
before using them to replace saved profiles.

`processing.destripe` holds the visible-channel column correction. It has no
plot or file access. All 96 bands from the first reference cube match the saved
arrays exactly under the locked environment.

Numerical functions receive values and return values. They do not read settings,
write files, display plots, or prompt the user.

## Migration

The `fitting_code` package is the legacy implementation. It remains available for
comparison until reference-output tests cover each replacement path.
