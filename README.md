# Titan Limb Fitting

Modern Python tools for the Cassini/VIMS Titan limb-profile study.

The new package is being built beside the legacy `fitting_code/` implementation.
Raw data and legacy results remain unchanged until reference-output checks cover
their replacement.

## Setup

```bash
uv sync
uv run titan-limb --help
just check
```

## Configuration

Copy `configs/default.toml` to `configs/local.toml` and set the local data path,
or use environment variables:

```bash
export TITAN_DATA_DIR=/absolute/path/to/Titan_Limb_Fitting/data
export TITAN_ARTIFACT_DIR=/absolute/path/to/output/artifacts
```

The environment variables take priority over the TOML file.

## Baseline data manifest

```bash
just manifest /absolute/path/to/Titan_Limb_Fitting/data
just validate-manifest /absolute/path/to/Titan_Limb_Fitting/data
```

The manifest records each file's relative path, byte size, modification time,
and SHA-256 digest. It does not alter source data.

## Convert saved fits

```bash
just migrate-selected-fits /absolute/path/to/Titan_Limb_Fitting/data/selected_fits
```

This reads the old pickles without changing them and writes one compressed,
typed Parquet table to `artifacts/processed/legacy-selected-fits.parquet`.

Convert the saved radial profiles in the same way:

```bash
just migrate-profiles /absolute/path/to/Titan_Limb_Fitting/data/sorted_and_filtered
```

Audit the converted fits before analysis:

```bash
just audit-fits
```

The default audit rejects failed or malformed fits and sends negative-R² fits
to review. Optional R² and coefficient limits remain off until scientific
review sets them.

Build the typed observation timeline from the preserved Nantes table and cube
selection:

```bash
just observations
```

This writes the 29 selected observation times and source metadata to
`artifacts/processed/observations.parquet`.

Build explicit wavelength crossings from eligible fits:

```bash
just analyze-transitions
```

The result keeps multiple crossings as separate rows.

Build the paired north-minus-south coefficient table:

```bash
just analyze-asymmetry
```

Render the first two analysis figures:

```bash
just figures
```

This writes a dated transition plot and a bandwise asymmetry spectrum under
`artifacts/figures/`. The asymmetry line shows the median across observations;
its shaded range shows the middle half. It does not show a confidence interval.

## Current migration rule

Do not edit or delete legacy numerical code until the matching new module has
reference-output tests against existing Cassini or SRTC++ results.
