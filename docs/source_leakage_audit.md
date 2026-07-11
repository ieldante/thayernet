# Source Leakage Audit

The completed audit is under
`outputs/runs/source_leakage_audit_20260710_062950/`. It is a read-only audit of
`data/Galaxy10_DECals.h5`; it does not load a model, modify the dataset, or
rewrite historical metrics.

## Main result

The historical seed-42 random-index partitions are disjoint by HDF5 row index but
are not disjoint by source object or exact image content.

- Total rows: 17,736
- Train / validation / test: 12,415 / 2,660 / 2,661
- Pairwise row-index intersections: 0
- Exact pixel-hash duplicate groups: 60
- Exact pixel-identical pairs: 62
- Cross-split exact duplicate groups: 28
- Cross-split exact pixel-identical pairs: 29
- Exact `(RA, Dec)` duplicate groups: 59
- Exact-coordinate pairs crossing train/validation/test: 27

The 29 cross-split pixel-identical pairs comprise 27 same-coordinate duplicated
objects plus two cross-split pairs from one three-row constant-value image group.
The latter images have every RGB byte equal to 32 but carry different catalog
coordinates and labels. The exact-coordinate duplicates are repeated source
objects and often also carry different class labels in the HDF5 rows.

Cross-split exact-pixel pair counts are:

| Partition pair | Exact pairs |
| --- | ---: |
| Train / validation | 13 |
| Train / test | 12 |
| Validation / test | 4 |

This is a major blocker for treating the original random-index normal/stress
benchmarks as a locked final scientific test. Existing results remain useful as
development-benchmark evidence. The blocker was resolved for the audit
campaign's retrain by building an exact-pixel and exact-coordinate
group-disjoint split; it remains a blocker for any claim based only on the
original random-index protocol.

## Grouped-protocol resolution

The verified grouped split contains 12,417 train, 2,660 validation, and 2,659
test sources. Exact image hashes and exact finite stored coordinates are unioned
before assignment, leaving zero such groups across split boundaries. This is a
strong correction for the observed leakage, but it is not proof that every
possible near-duplicate identity has been exhaustively discovered.

The grouped blend manifests contain 8,000 train rows, 1,000 validation rows,
and four 1,000-row development-test suites. Both source roles obey the grouped
split, all 71/71 integrity checks pass, and all 13,000/13,000 rows replay
exactly. The grouped v0.2 Moderate retrain and its grouped development
evaluation are complete. These artifacts support development validation, not a
final-paper claim.

## What did pass

The partition-before-blending implementation behaves as intended at the row
index level. The audit examined 56 historical per-sample/manifest-like CSVs. Of
those, 23 retained auditable source indices, covering 21,060 indexed blend rows.
No target/contaminant role-containment failure, out-of-range index, or same-source
target/contaminant pair was found.

The completed stress-test and ResUNet architecture-ablation targeted-suite
tables use subset-local indices that map entirely into the held-out test
partition. The five provisional
final-manifest CSVs use global HDF5 indices and likewise keep both roles in their
declared test source pool. Of the other 33 artifacts, 32 do not contain source
index columns and one ResUNet normal table contains those columns but leaves
them blank. Their individual samples cannot be independently re-identified from
saved metrics alone.

## HDF5 metadata and local dataset variants

The local file contains `images`, `ans`, `ra`, `dec`, `redshift`, and `pxscale`.
It has no separate object-ID/catalog-ID dataset or root metadata attributes.
Exact coordinates therefore provide the strongest available local object-group
key, with perceptual fingerprints used as a conservative secondary screen.

A read-only full-home filename search and a direct project-data check found no
local `Galaxy10_DECals_NoDuplicated.h5`-style file. If such a file is obtained
later, it should be audited rather than trusted from its filename alone.

## Perceptual duplicate screen

Every image was streamed through:

- SHA-256 of contiguous raw `256 x 256 x 3 uint8` pixel bytes;
- a 64-bit DCT pHash from a `32 x 32` block-mean grayscale view;
- a 64-bit gradient dHash;
- downsampled grayscale normalized cross-correlation and RMSE;
- an RA/Dec separation check when metadata was available.

The multi-index search completely covers pHash Hamming distance <= 7 and dHash
distance <= 3 across train, validation, the first 1,000 test positions used by
development evaluations, and the remaining provisional final-test tail. It
retained 25 medium candidates, with no non-exact high-confidence or critical
candidate. These are ranked in
`tables/near_duplicate_candidates.csv`; perceptual candidates are review leads,
not proof of duplicate identity.

## Superseded provisional final-manifest cross-check

The five CSVs under
`outputs/runs/final_test_manifest_prep_20260710_061737/` contain 5,000 blend rows
using 1,000 unique global sources from the post-development test tail.

- Sources outside that tail: 0
- Sources linked to train/validation/development-prefix by exact pixels: 0
- Sources linked by exact or <=1-arcsec coordinates: 0
- Automated medium perceptual candidates: 3 sources / 4 pairs
- Sources clear before manual candidate review: 997

All four medium pairs were then reviewed from source cutouts. Their coordinates
differ by 20,235--267,420 arcsec, their dHash distances are 17--29, and their
surrounding stars and galaxy details differ. They are morphology-driven hash
collisions, not plausible duplicates. The append-only review is recorded in
`diagnostics/provisional_manifest_independent_review.md`, with its contact sheet
under `figures/` in the audit run. The manifest files were not changed, and all
entries in their existing `SHA256SUMS` file still verify.

Those perceptual candidates did not by themselves invalidate the pool.
Subsequent grouped-split accounting did: the 1,000 sources map to 683 grouped
train, 173 grouped validation, and 144 grouped test sources. The actual grouped
train and validation blend manifests use 499 and 91 of them, or 590/1,000 in
total. The pool is therefore **superseded and not final-eligible**, regardless
of the earlier within-protocol cross-check. It remains preserved for historical
reproducibility.

## Completed correction and remaining final-test requirement

The audit campaign completed these corrective development steps:

1. Group exact pixels and exact coordinates before train/validation/test
   assignment.
2. Preserve global source indices and group IDs for both blend roles.
3. Verify role containment, cross-split group isolation, and exact replay.
4. Retrain v0.2 Moderate and evaluate it on the grouped development suites.

Before a final paper claim, freeze the model and complete protocol, then create
a fresh untouched source pool that is group-disjoint from all training,
validation, and development-test groups. Audit exact hashes, coordinates, and
high-confidence perceptual candidates before one predeclared evaluation. Keep
artifact flags as reviewable strata rather than automatic exclusions unless a
separate rule is frozen in advance.

## Reproduction

```bash
.venv/bin/python scripts/source_leakage_audit.py \
  --batch-size 64 \
  --full-home-preflight-no-match
```

The flag records the separate completed full-home read-only filename preflight;
the script itself checks the project data directory and refuses to overwrite an
existing run directory.

The primary artifacts are:

- `tables/source_partition_audit.csv`
- `tables/exact_duplicate_audit.csv`
- `tables/near_duplicate_candidates.csv`
- `tables/provisional_manifest_source_crosscheck.csv`
- `diagnostics/source_leakage_audit_report.md`
- `diagnostics/source_leakage_audit_summary.json`

Two earlier timestamped audit directories are preserved with interruption notes.
Their computations were stopped only because macOS blocked while opening synced
user directories during an optional filename search; neither is a completed
result and neither was deleted or reused.
