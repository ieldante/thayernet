# Duplicate-Safe Grouped Split Protocol

## Why the split changed

The original deterministic split shuffled HDF5 row indices. Those row sets were
disjoint, but Galaxy10 contains repeated source cutouts. The audit found 29
pixel-identical cross-split pairs and 27 exact-coordinate cross-split pairs, so
row-level separation did not imply object/content-level separation.

Historical metrics remain recorded as original development-split results. They
are not duplicate-safe final estimates and cannot be repaired by relabeling the
old rows after training.

## Group construction

`scripts/build_grouped_source_split.py` builds groups before assigning splits:

1. SHA-256 is computed from every stored uint8 RGB image.
2. Rows with identical image hashes are unioned.
3. Rows with exactly equal finite stored `(RA, Dec)` coordinates are unioned.
4. The unions are transitive, so any connected evidence belongs to one group.
5. Perceptual candidates are not grouped automatically. A future version may
   add only pairs explicitly reviewed as the same source.

Accordingly, this split is exact-pixel and exact-coordinate group-disjoint. It
is not proof that all possible near-duplicate source identities have been
exhaustively found.

The current manifest uses grouping version
`grouped_source_split_v1_exact_pixels_exact_coordinates` and assignment seed
`20260710`.

## Assignment and verification

Whole groups are assigned to train, validation, or development test with a
deterministic, label-aware objective targeting 70/15/15 source fractions. The
builder fails unless:

- every dataset source appears exactly once;
- every group appears in exactly one split;
- exact-image and exact-coordinate groups have zero cross-split leakage;
- only the three declared split labels occur; and
- source-count deviations remain within the maximum group-size tolerance.

The first verified manifest is under
`data/manifests/grouped_source_split_20260710_100907/`:

| Split | Sources | Groups |
| --- | ---: | ---: |
| train | 12,417 | 12,374 |
| validation | 2,660 | 2,651 |
| test | 2,659 | 2,650 |

It contains 17,675 groups for 17,736 rows. All 60 exact-image evidence groups
and all 59 exact-coordinate evidence groups stay within one split.

## Verified grouped blend manifests

Grouped blend manifests retain global source indices and group IDs for
both target and contaminant. Training blends use only train groups, validation
blends only validation groups, and evaluation blends only test groups. Target
and contaminant must be distinct groups even within one sample. Each row must
also retain an independent replay seed, explicit blend parameters, code/data/
split hashes, and generated blend/mask hashes.

The verified manifest set under
`data/manifests/grouped_blends_20260710_103233/` contains:

| Role / suite | Rows |
| --- | ---: |
| train | 8,000 |
| validation | 1,000 |
| normal development test | 1,000 |
| hard-stress development test | 1,000 |
| compact-bright development test | 1,000 |
| high-core-obstruction development test | 1,000 |

All 71/71 integrity checks passed, including split-role and group containment,
and all 13,000/13,000 rows replay exactly from their recorded sources and
parameters.

## Superseded provisional final pool

The earlier 1,000-source provisional pool at
`outputs/runs/final_test_manifest_prep_20260710_061737` is not final-eligible.
When mapped to the grouped split, its sources fall into 683 train, 173
validation, and 144 test sources. The actual grouped train and validation blend
manifests use 499 and 91 of those sources, respectively, or 590/1,000 total.
The run remains preserved as historical infrastructure, but it cannot be used
as an untouched final pool.

## Grouped retrain status

The authorized v0.2 Moderate grouped retrain and grouped development evaluation
are complete. The grouped checkpoint reports 28.8x lower normal and 15.8x lower
hard-stress affected-region MSE than identity. These are development results;
they neither rehabilitate the superseded provisional pool nor establish a
final-paper estimate.

## Claim status

The grouped manifests in this campaign are duplicate-safe development
infrastructure, not a pristine locked final-paper test: their construction and
distributions are part of the current audit and can still influence protocol
decisions. After the infrastructure and model settings are frozen, a fresh
untouched source pool and grouped final manifest must be generated. Its source
groups must be disjoint from every grouped train, validation, and development
test group, and it should be evaluated once under the frozen protocol.
Confidence intervals should account for repeated target/contaminant groups
rather than treating every blend row as an independent source draw.
