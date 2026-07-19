# D3 Threshold Contract

Forward plausibility uses inclusive comparisons. The exact limits are:

- global mean squared whitened residual: `1.2543178712712195`;
- g/r/z mean squared whitened residual:
  `1.2000065947013574`, `1.2258474450543715`, and
  `1.256406290721562`;
- absolute relative-flux residual: `0.12280256285502243`.

Truth coverage uses primary normalized distance at most `1.0`. Its component
limits are image symmetric relative L2 `0.25`, relative flux `0.20` in each
g/r/z band, g-r and r-z color error `0.20` magnitude, and centroid error `0.50`
mean PSF FWHM. Ordinary expert concentration has primary diameter at most
`1.0`. Color and centroid components retain their frozen applicability masks.

The forward numeric values have one frozen authoritative machine-readable
source. Production/reference code independently confirms the keys, inclusive
operators, band semantics, finite-value rule, and formulas. Truth-coverage
values and rules agree across the production distance, differentiable
scientific constants, and independent reference implementation.

