# Reproduction

## Fast check

```bash
uv sync
just check
```

## Validate local data

```bash
just manifest /absolute/path/to/Titan_Limb_Fitting/data
just validate-manifest /absolute/path/to/Titan_Limb_Fitting/data
```

## Convert and check saved selected fits

```bash
just reference-test /absolute/path/to/Titan_Limb_Fitting/data/selected_fits
just migrate-selected-fits /absolute/path/to/Titan_Limb_Fitting/data/selected_fits
```

The reference test checks known values from cube `C1477456872_1`. The conversion
produces 20,416 rows, including all 117 saved failures. See
`baseline/selected-fits-summary.json` for the frozen counts and table schema.

```bash
just migrate-profiles /absolute/path/to/Titan_Limb_Fitting/data/sorted_and_filtered
```

This creates 163,328 profile rows with 5,562,272 selected points. The source
pickles stay unchanged.

The real-data check also rebuilds all 96 visible bands for the first reference
cube and requires exact agreement with the saved destriped arrays.

## Audit fit quality

```bash
just audit-fits
```

The default audit yields 19,900 eligible rows, 399 review rows with negative
R², and 117 ineligible failed rows. It keeps every row and writes every reason.
Candidate threshold effects are frozen in `baseline/fit-quality-summary.json`.

## Build the observation timeline

```bash
just observations
```

This reads `fitting_code/ingestion/data/combined_nantes.csv` and
`settings/s3xy_cubes.json`. It writes 29 typed rows from 26 October 2004 through
8 June 2017. See `baseline/observations-summary.json` for the frozen result.

## Build transition crossings

```bash
just analyze-transitions
```

This produces 60 crossing rows for 58 cube/hemisphere series. Two north series
have two crossings and remain explicit review cases. Each row includes its
observation time and decimal year.

## Build north-south differences

```bash
just analyze-asymmetry
```

This writes 4,067 paired eligible rows across all 29 cubes and 141 allowed
bands. Each row includes its observation time and decimal year. The output is
descriptive until uncertainty rules are added.

## Render global figures

```bash
just figures
```

This writes `transition-timeline.png` and `asymmetry-spectrum.png` under
`artifacts/figures/`. The transition figure marks the two series with more than
one crossing. The asymmetry figure shows each band's median and middle half
across the selected observations; the shaded range is not a confidence
interval.

## Build the seasonal phase summary

```bash
just analyze-seasons
just seasonal-figure
```

The first command writes 58 cube-channel rows and six phase summaries. There
are 12 northern-winter, 16 northern-spring, and one northern-summer observation
per channel. Only winter and spring receive fixed-seed 95 percent bootstrap
intervals. The second command writes `seasonal-asymmetry.png`.

The method treats cubes, not bands, as the sampling units. It does not include
fit covariance, time correlation, or instrument drift and does not support a
significance claim. See decision 0005 and `baseline/seasonal-summary.json`.

Full scientific reproduction commands will follow as each old numerical stage
gets a tested replacement.
