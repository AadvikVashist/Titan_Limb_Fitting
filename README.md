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

## Current migration rule

Do not edit or delete legacy numerical code until the matching new module has
reference-output tests against existing Cassini or SRTC++ results.
