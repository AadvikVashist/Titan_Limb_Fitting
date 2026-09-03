# Seasonal grouping and uncertainty

Status: descriptive phase comparison adopted; fit errors added

The northern season boundaries are dated config, not rough decimal-year
cutoffs. Northern spring begins on 11 August 2009, the Saturn and Titan vernal
equinox date given by NASA. Northern summer begins on 24 May 2017, the solstice
date given by NASA.

Sources:

- https://science.nasa.gov/solar-system/planets/saturn/saturn-moons/on-titan-the-sky-is-falling/
- https://science.nasa.gov/mission/cassini/grand-finale/grand-finale-orbit-guide/

The analysis first calculates one median north-minus-south `u1 + u2` value for
each cube and VIMS channel. This keeps the 141 allowed bands from acting as 141
independent observations. Each cube summary also retains its band count and
within-cube band quartiles.

For each season and channel, the output reports the median across cube values.
A fixed-seed percentile bootstrap resamples whole cube values 5,000 times and
reports the central 95 percent interval. Groups with fewer than five cubes get
no interval. The current data have 12 winter, 16 spring, and one summer cube per
channel, so the summer result stays descriptive.

These intervals describe sampling variation among the selected observations.
The band-level asymmetry table now propagates each fit covariance to
`u1 + u2`, then combines the north and south standard errors in quadrature.
Season intervals still describe variation among whole observations. They do
not yet fold the band-level fit errors into a single formal test, and they do
not include band-selection uncertainty, time correlation, or instrument drift.
The output makes no significance claim.
