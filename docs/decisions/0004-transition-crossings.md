# Transition crossing output

Status: adopted 3 September 2026

The compatibility detector keeps the old inputs and method: the one-based band
exclusion list, Gaussian smoothing with sigma 4, PCHIP interpolation, 3,000
sample points, and a 0.6 micrometre lower bound. It now uses only paired north
and south rows that pass the fit-quality gate.

The output keeps every crossing. It does not average unrelated crossings. Two
north series have two crossings: `C1861904325_1` and `C1875658704_1`. The saved
global result reduced each pair to one mean.

Fifty-three of 58 series match the saved crossing exactly. Three single-crossing
series shift because quality review removes one or more paired bands. The two
multi-crossing series also change slightly and remain unresolved. The baseline
report records the values.

A series with one crossing can enter scalar transition summaries. A series with
more than one crossing stays in the row-level result and figures but does not
enter a scalar summary. We do not choose or average crossings without a physical
reason. The paper must name each excluded multi-crossing series.
