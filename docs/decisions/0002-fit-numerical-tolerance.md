# Fit tolerance after the package upgrade

Status: accepted for compatibility checks

The new fitter keeps the saved method: cosine conversion, PCHIP interpolation
to 200 points, Gaussian smoothing with sigma 20, the original moving-average
rule with window 20, unconstrained `u1` and `u2`, and selection by R² on the
source profile.

The reference set covers the first, middle, and last selected cubes and visible,
overlap, infrared, and final bands. It checks both hemispheres: 24 fits in all.
The locked SciPy and scikit-learn versions choose the same smoothing method for
all 24. The checks allow these small package-version differences:

- Fit parameters: `2e-7` absolute.
- R²: `5e-8` absolute.
- Covariance values: `1e-7` absolute.

These limits preserve compatibility. They do not define whether a fit is fit for
scientific use. A separate quality policy must reject or flag weak and extreme
fits before global analysis.
