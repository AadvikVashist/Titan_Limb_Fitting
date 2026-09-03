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

`io.observations` reads the preserved Nantes CSV and the selected-cube map. It
normalizes the 29 observation times to UTC, calculates decimal years, and joins
stable time, label, and flyby fields to later result tables.

`io.vims` is the only new PyVIMS boundary. It loads cubes on demand, so package
imports do not parse cube data or import PyVIMS. `processing.geometry` contains
the first pure profile functions. See `docs/decisions/0001-opencv-raster-drift.md`
before using them to replace saved profiles.

`processing.destripe` holds the visible-channel column correction. It has no
plot or file access. All 96 bands from the first reference cube match the saved
arrays exactly under the locked environment.

`fitting.optimizer` holds interpolation, smoothing, quadratic fitting,
covariance, and R² selection. It returns all three candidates and the selected
fit. The accepted package-upgrade tolerances are recorded in decision 0002.

`fitting.quality` applies hard validity checks and separate review flags. Global
analysis may use only `eligible` rows. No R² or coefficient cutoff is active by
default; those values need a stated scientific choice.

`analysis.transitions` joins fits to quality results, requires paired eligible
north and south rows, applies the central one-based band policy, and emits one
row per crossing. It never averages multiple crossings. Its saved table also
contains the observation time, decimal year, selection label, and flyby.

`analysis.asymmetry` pairs eligible north and south fits by cube and band. It
stores direct north-minus-south differences without treating them as
significant. Its saved table uses the same observation fields.

Numerical functions receive values and return values. They do not read settings,
write files, display plots, or prompt the user.

## Migration

The `fitting_code` package is the legacy implementation. It remains available for
comparison until reference-output tests cover each replacement path.
