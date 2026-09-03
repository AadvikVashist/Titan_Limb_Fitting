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

## Build transition crossings

```bash
just analyze-transitions
```

This produces 60 crossing rows for 58 cube/hemisphere series. Two north series
have two crossings and remain explicit review cases.

## Build north-south differences

```bash
just analyze-asymmetry
```

This writes 4,067 paired eligible rows across all 29 cubes and 141 allowed
bands. The output is descriptive until uncertainty rules are added.

Full scientific reproduction commands will follow as each old numerical stage
gets a tested replacement.
