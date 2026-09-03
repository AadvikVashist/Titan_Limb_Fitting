# Initial fit-quality policy

Status: adopted 3 September 2026

The default gate marks a fit ineligible when the old fit failed, the profile has
fewer than six points, a required value is not finite, or the covariance is not
a finite 3 by 3 matrix. A successful fit with negative R² requires review. All
other structurally valid fits remain eligible.

The main analysis requires non-negative R². This is already enforced by treating
negative-R² fits as review rows and allowing only eligible rows into global
analysis. We do not set a coefficient-size cutoff: large coefficients often
occur in excluded or noisy bands, and a hard coefficient bound lacks a physical
basis. Sensitivity tables must show the effect of stricter R² and coefficient
limits.

Global analysis must use only eligible rows unless a command names and records
an override policy.
