# Titan Limb Fitting

Modern Python tools for the Cassini/VIMS Titan limb-profile study.

The typed package now covers raw cube profiles, limb fitting, fit checks,
north--south and seasonal analysis, plots, sensitivity runs, and the SRTC++
comparison. The legacy `fitting_code/` tree remains read-only for audit work.

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
export TITAN_CONFIG_DIR=/absolute/path/to/policy/configs
```

The environment variables take priority over the TOML file. Explicit command
flags take priority over both. The CLI ships its default TOML files, so installed
commands do not need a source checkout for the standard band and season policy.
Use one settings file for a run with:

```bash
uv run titan-limb --config configs/local.toml data manifest
```

Artifact-producing raw, seasonal, and SRTC++ runs save a typed JSON receipt.
Each receipt records input and output hashes, resolved paths, parameters, the Git
revision when available, the `uv.lock` hash, timestamps, schema versions, run
status, and the rejection-ledger location. Raw resume accepts an old cube result
only when its receipt and output hashes still match the current inputs and
settings.

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
to review. The main analysis uses only eligible fits and applies no hard
coefficient bound.

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

Build and plot the seasonal phase summary:

```bash
just analyze-seasons
just seasonal-figure
```

The analysis reduces each cube to one spectral median per channel before it
compares seasons. It writes bootstrap intervals only for phases with at least
five observations.

## Current migration rule

Build the full raw result with:

```bash
just build-raw /absolute/path/to/Titan_Limb_Fitting/data/original_cubes
just validate-raw /absolute/path/to/Titan_Limb_Fitting/data/selected_fits
just sensitivity
```

The accepted line raster comes from the locked current stack. The validation
report records every difference from the old OpenCV output. Do not edit or
delete the legacy numerical code; it remains the audit source.
