# Failure-specific target schema

The prospective schema separates query ownership from reconstruction quality.
It never creates one composite recoverability label.

`QUERY_STATE` applies to every row and has exactly three values:

- `UNIQUE_VALID`: one uniquely associated source and requested-source truth;
- `NULL`: no source within the inclusive four-pixel matching radius and no
  requested-source truth;
- `AMBIGUOUS`: a nearest/second-nearest distance gap at most one pixel,
  including ties, and no single requested-source truth.

Only `UNIQUE_VALID` rows receive continuous image NRMSE, per-band and maximum
relative flux risk, centroid risk in pixels and PSF-FWHM units, binary source
confusion, and binary catastrophic-valid failure. Catastrophe is confusion, a
nonfinite required risk, or a two-fold violation of a moderate image, flux, or
centroid limit.

Only `NULL` rows receive continuous output-energy and absolute-flux exposure
ratios plus binary `NULL_HALLUCINATION`. Only `AMBIGUOUS` rows receive the
descriptive `AMBIGUOUS_FORCED_OUTPUT`; that outcome was not trained and assigns
no truth source.

Each target has an explicit applicability mask. Authoritative v4 sample tables
retain missing nonapplicable values; the audit found zero undefined-to-false/
zero coercions and zero values defined outside applicability. Thresholds have
strict, moderate-primary, and permissive scientific interpretations while all
continuous values are preserved.
