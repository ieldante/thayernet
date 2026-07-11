# CatSim five-way source-split protocol

Version: `catsim-btk-v1`; split seed: `2026071199`.

## Identity and grouping

The source catalog is the official BTK v1.0.9 `input_catalog.fits`, SHA-256
`cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46`.
Its 86,273 `galtileid` values are unique, but published documentation does not
promise that field is a globally persistent identifier. The binding identity
is therefore `(catalog SHA-256, zero-based stable catalog row, galtileid)`.
An in-memory DataFrame index is never an identity.

Exact original `(ra, dec)` pairs define conservative duplicate groups. This
groups 13 repeated position pairs even though all complete rows and all
`galtileid` values are unique. Every group is indivisible.

## Engineering exclusion and allocation

Every source used by the CatSim pilot, replay tests, any COSMOS comparison, or
the optional neural smoke run is `engineering_excluded`. A group containing
one such row is wholly excluded. The current 60 pilot sources are excluded from
all five final partitions. They may be reused by the engineering smoke run but
may never later become final training, validation, calibration, development,
or lockbox sources.

Remaining groups are deterministically stratified by catalog-supported fields:
dominant bulge/disk/AGN component, r-band magnitude quintile, and redshift
quartile. Groups are shuffled within strata with the fixed split seed and
allocated approximately:

- training 70%;
- validation 10%;
- calibration 8%;
- development test 7%;
- sealed lockbox 5%.

Size, ellipticity proxy, g−z color, redshift, magnitude, and morphology balance
are diagnostics, not reasons to split a duplicate group. The catalog has no
missing values in the audited fields. No unavailable subtype is invented.

The reproducible row/group assignment is
`outputs/runs/thayer_select_btk_foundation_20260711_152613/manifests/btk_engineering_source_groups.csv`.
Counts, fractions, zero overlaps, zero engineering leakage, and per-variable KS
distances are in `tables/btk_split_summary.csv` in the same run. The largest
reported KS distance versus the full eligible pool is below 0.016.

## Scene-role and lockbox rules

Targets and contaminants in a scene must come from the same partition. A
source/group may not be a target in one partition and a contaminant in another,
and may not change partitions through a different scene role. Normalization is
fit from training only; threshold calibration uses calibration only.

The sealed-lockbox assignment may be stored, hashed, and checked for overlap,
but its sources must not be sampled, rendered, plotted, inspected, normalized,
debugged against, or used for selection. Opening requires a separately
authorized final evaluation with frozen model, normalization, metric,
calibration, and reporting protocol. Any duplicate/group crossing or
engineering leakage is a fail-closed error.
