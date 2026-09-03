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

Full scientific reproduction commands will follow as each old numerical stage
gets a tested replacement.
