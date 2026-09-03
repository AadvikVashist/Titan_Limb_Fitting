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

Numerical functions receive values and return values. They do not read settings,
write files, display plots, or prompt the user.

## Migration

The `fitting_code` package is the legacy implementation. It remains available for
comparison until reference-output tests cover each replacement path.

