# Initial fit-quality policy

Status: structural checks active; scientific thresholds open

The default gate marks a fit ineligible when the old fit failed, the profile has
fewer than six points, a required value is not finite, or the covariance is not
a finite 3 by 3 matrix. A successful fit with negative R² requires review. All
other structurally valid fits remain eligible.

No minimum R² or coefficient range is active by default. Those choices affect
many rows and need a scientific basis. The baseline summary records the effect
of several candidate values so the choice can be made with its cost visible.

Global analysis must use only eligible rows unless a command names and records
an override policy.
