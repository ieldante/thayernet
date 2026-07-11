# Final-Test Protocol

## Current status

The current normal and hard-stress evaluations are development benchmarks. They
have already informed architecture, loss, and model-selection decisions, so
they should not be treated as an untouched final test for a paper claim.

The earlier manifest-preparation run is:

`outputs/runs/final_test_manifest_prep_20260710_061737`

Its role was explicitly `provisional_locked_manifest_prep`. It is now
**superseded and not eligible for final evaluation**. Although no model was
loaded and no inference, metric comparison, rendering, or qualitative
inspection was performed when it was created, its source pool is not isolated
from the later grouped development protocol. Mapping its 1,000 sources onto the
verified grouped split assigns 683 to train, 173 to validation, and only 144 to
test. The actual grouped train and validation blend manifests use 499 and 91 of
those sources respectively: 590/1,000 in total. The files remain preserved as
historical manifest-infrastructure work and must not be relabeled as final.

The earlier leakage audit is
`outputs/runs/source_leakage_audit_20260710_062950`. Its append-only independent
review found no sustained exact, coordinate, or perceptual duplicate link for
any selected source relative to the historical partitions it checked. That
result does not override the later grouped-split mapping above and does not make
the provisional pool final-eligible.

## Historical provisional source-pool construction

The historical source split is a seed-42 random index split. A source-leakage
audit found 59 duplicate-coordinate groups overall and 27 exact-coordinate
pairs crossing those historical splits. Random-index disjointness therefore
does not guarantee object-level disjointness.

The provisional manifest generator attempted to correct the then-known
coordinate leakage without altering the dataset or historical splits:

1. Reconstruct the seed-42 70/15/15 global-index split.
2. Treat the first 1,000 test-split sources as the development-test prefix.
3. Scan the remaining 1,661 test sources in fixed split order.
4. Hash each exact finite float64 `(RA, Dec)` pair into a coordinate-group ID.
5. Exclude a candidate if its coordinate group appears in train, validation, or
   the development-test prefix.
6. Keep only the first representative of each coordinate group.
7. Freeze the first 1,000 eligible representatives.

This excluded 11 test-tail sources and left 1,650 eligible group-isolated
representatives. The selected 1,000-source pool has zero coordinate-group
overlap with train, validation, or the development prefix. The selection method,
source indices, group IDs, exclusion counts, and assignment hashes are recorded
in `logs/split_and_source_pool_audit.json` within the run.

The later audit streamed exact-image hashes for all 17,736 cutouts and compared
the 1,000 selected manifest sources against train, validation, and the
development-test prefix. It found zero exact-image or same-coordinate blockers.
Three selected sources produced four medium perceptual-hash candidates, but no
high-confidence candidate. Independent review classified all four as distinct
galaxies and fields: angular separations were 20,235–267,420 arcsec, dHash
distances were 17–29, and surrounding structures differed. Those candidates did
not by themselves require regeneration. The pool is nevertheless superseded
because the grouped split and grouped development manifests later established
direct development-protocol reuse.

The review supplement is
`outputs/runs/source_leakage_audit_20260710_062950/diagnostics/provisional_manifest_independent_review.md`
with SHA-256
`d3ea7baa7e5488a78540d4a0e92581b75618dd3361772206aacca8132e285c9e`.
Unknown access from external or untracked notebooks and the absence of an
external object-ID/catalog crossmatch remain limitations.

## Historical provisional suites

The run historically defined these suites with seeds that were not reused from
the then-current development evaluation families:

| Historical suite label | Samples | Seed | Purpose |
| --- | ---: | ---: | --- |
| Normal final test | 1,000 | 910300101 | Same broad parameter family as the normal development generator |
| Hard stress final test | 1,000 | 910300211 | Bright, close, similar-or-larger contaminants |
| Compact bright final test | 1,000 | 910300307 | Compact, bright contaminant failures |
| High core obstruction final test | 1,000 | 910300401 | Strong target-core obstruction |
| Halo/artifact stress final test | 1,000 | 910300503 | Halo-oriented blur/broad-artifact proxy |

The halo/artifact suite is not a validated source-artifact subset. Galaxy10 has
no validated source-quality labels for that purpose, so the suite is explicitly
labeled a halo-oriented proxy and does not claim to contain known artifacts.

## Manifest contents and integrity

Each of the five suites has a CSV and JSON manifest under `manifests/`. Every
row records:

- global target and contaminant source indices and class labels;
- test split, source-pool selection method, and hashed coordinate-group IDs;
- shift, brightness, size ratio, blur, noise, and rotation parameters;
- foreground-mask and evaluation-mask parameters;
- affected fraction, core obstruction, halo fraction, and blend severity;
- an independent per-sample replay seed and the suite seed;
- generator version and combined source-code hash;
- an immutable per-sample SHA-256 fingerprint;
- explicit coordinate-audit and perceptual-audit state at generation time.

No target, contaminant, blended image, pixel, tensor, or mask arrays are stored.
The manifest pairs contain 5,000 rows total and 78 fields per row. Validation
confirmed exact CSV/JSON counts and schemas, unique IDs, fingerprints, seeds,
and generation signatures, finite size ratios, nonempty affected masks, source
membership, and coordinate-group isolation. All manifest/schema files are mode
`0444`; `manifests/SHA256SUMS` verifies all 12 locked files.

## Requirements for a future final evaluation

The provisional manifests must not be used for a final claim. A fresh source
pool and fresh suite manifests must be generated only after the model,
checkpoint list, blend protocol, masks, metrics, clipping policy, and reporting
rules are frozen. The new pool must be group-disjoint from every source group
used for grouped training, validation, or development testing. In particular:

1. Preserve the superseded run unchanged as a historical artifact.
2. Do not render, evaluate, or tune from the future final pool before the
   protocol freeze.
3. Audit exact pixels and exact coordinates before first inference, and review
   high-confidence perceptual candidates. Exact-pixel and exact-coordinate
   group disjointness is not proof that all near-duplicate identities have been
   found.
4. Create a new timestamped run if an audit establishes a blocker; never modify
   a frozen run in place.
5. Freeze candidate checkpoints and SHA-256 values before first final
   inference.
6. Evaluate all predeclared models once on the same manifests and report every
   suite, including unfavorable results.

This separation matters because repeated inspection of development tests can
turn them into an informal validation set. A future genuinely locked,
leakage-audited final manifest would give the final comparison a defensible
boundary and make sample generation reproducible without putting raw images in
tracked files.

## Current grouped development infrastructure

The verified grouped source split is disjoint by exact-pixel and exact stored
coordinate groups. Its grouped blend manifests contain 8,000 training rows,
1,000 validation rows, and four 1,000-row test suites. All 71/71 integrity
checks passed and all 13,000/13,000 rows replay exactly. These are strong
development-protocol checks, not an untouched final evaluation: their sources,
distributions, and results have already informed this audit and retrain.

The grouped v0.2 Moderate retrain and grouped development evaluation are
complete. That evaluation reports 28.8x and 15.8x lower affected-region MSE
than identity on the grouped normal and hard-stress suites, respectively. These
numbers do not authorize reuse of the superseded provisional pool; a new
untouched group-disjoint final pool remains required.

## Generator

The historical append-only generator is
`scripts/prepare_final_test_manifests.py`. Each
invocation creates a new timestamped directory and refuses to overwrite an
existing run. The earlier interrupted draft at
`outputs/runs/final_test_manifest_prep_20260710_060845` is preserved and marked
`BLOCKED_DRAFT_DO_NOT_USE` because it was based only on random-index
disjointness. A later conservative-exclusion setup was stopped before manifest
generation after the independent review cleared its three proposed exclusions;
that decision is preserved at
`outputs/runs/final_test_manifest_prep_conservative_exclusion_setup_20260710_063746`.
