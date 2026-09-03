# Titan Limb Fitting

## Required practice

- Preserve raw VIMS and SRTC++ inputs.
- Add a reference-output test before changing scientific calculations.
- Keep code-only changes separate from method changes.
- Use typed models at file and processing boundaries.
- Use one-based VIMS band numbers in scientific records.
- Convert to zero-based array positions only through named helpers.
- Do not perform file access, plotting, or long work during module import.
- Do not use broad exception handlers.
- Do not hide failed fits or missing values.
- Record a reason for every rejected cube, band, profile, and fit.
- Put constants before function definitions.
- Use `pathlib.Path` for paths.
- Use Polars for main tables, NumPy for arrays, SciPy for fitting, Seaborn for
  statistical plots, and Matplotlib for image and layout control.
- Run `just check` before committing.

## Commands

- Install: `uv sync`
- Format: `just format`
- Check: `just check`
- Test: `just test`
- CLI help: `uv run titan-limb --help`

## Legacy code

`fitting_code/` remains read-only until a replacement path has reference-output
coverage. Delete a legacy module only after all active behavior has moved and its
replacement passes the matching tests.

