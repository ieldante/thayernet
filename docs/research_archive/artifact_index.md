# Artifact index

## Scope

This index classifies the initial research workspace without making the ignored
56 GB run tree part of Git. Exact run-directory coverage is in
[`experiment_ledger.csv`](experiment_ledger.csv); exact hashes and dispositions
for every file above 5,000,000 bytes are in
[`large_file_audit.csv`](large_file_audit.csv); selected compact evidence is in
[`../experiment_archive/`](../experiment_archive/).

The initial metadata scan excluded `.git`, local virtual environments, Python
and tool caches, and notebook checkpoints. It found 35,060 research files,
including 31,492 below 124 top-level run directories.

## Artifact classes

| Class | Initial count or scale | Canonical locations | Commit disposition | Evidence use |
| --- | ---: | --- | --- | --- |
| Source modules | 62 untracked Thayer Python modules plus tracked foundation modules | `src/` | Commit after syntax/tests/privacy audit | Model, solver, audit, physical-contract, and reproducibility implementation |
| Experiment/audit scripts | 130+ untracked Thayer scripts plus tracked foundation scripts | `scripts/` | Commit relevant reviewed scripts | Campaign drivers, finalizers, replay, audits, archive builders |
| Tests | 111 total in broad scan | `tests/` | Commit | Unit, contract, artifact, replay, corruption, and integration validation |
| Configs | 20 YAML/TOML/config-like files in broad scan | `configs/`, `pyproject.toml`, run protocols | Commit portable configs; archive frozen protocols | Environment and frozen parameter contracts |
| Notebooks | 2 | `notebooks/`, `data_exploration/notebooks/` | Main notebook already tracked; exclude the executed exploratory notebook unless outputs/paths are cleared in a separate task | Exploratory workflow only |
| Documentation/reports | 392 initial documentation/report-like files | `docs/`, `reports/`, ignored run reports | Commit canonical docs and selected compact authorities | Claim authority, methodology, status, historical outcomes |
| Compact tables/manifests | 5,108 JSON/CSV/TSV files in broad scan | Mostly ignored runs; selected archive | Commit selected small, nonprotected copies | Exact values, membership, decisions, hashes, replay contracts |
| Figures | 851 | `reports/figures/`, ignored run trees, curated archive | Commit only selected compact central figures | Poster, paper, supplement, visual QA |
| HDF5 | 88 | `data/`, ignored runs | Exclude | Raw inputs, rendered scenes, features, predictions, audit episodes |
| NPY/NPZ | 1,005 | ignored runs | Exclude | Dense arrays, embeddings, caches, observations |
| Checkpoint-like PT/PTH | 769 in broad extension/path scan | `outputs/checkpoints/`, ignored runs | Exclude | Historical model/state replay; hashes only in Git |
| Logs/text outputs | 1,113 broad scan | ignored run trees | Exclude except compact validation summaries | Engineering diagnosis and access audit |

Counts overlap where extensions and paths satisfy multiple categories.

## Curated archive

The compact archive contains 34 campaign directories and was built from an
explicit allowlist. Its validator found:

- 174 files totaling 2,331,401 bytes;
- 42 CSV, 15 JSON, 14 PNG, 100 Markdown, and 3 Python files;
- zero file above 5,000,000 bytes;
- zero HDF5, FITS, NPY/NPZ, PT/PTH, binary checkpoint, log, or JSONL files;
- all JSON and CSV parse successfully and all PNGs identify as PNG;
- all 139 provenance rows match original/archive sizes and SHA-256 values;
- zero machine-user paths, email addresses, or high-confidence secret patterns;
- zero exact duplicate files inside the curated archive.

Seven initially selected figures were removed during protection review: five
rendered isolated/hidden truth panels and one rendered development blends and
model outputs. A seventh training-microset output grid was noncentral generated
imagery. Their local originals remain untouched; metric-only figures and
text/tables carry the committed claims.

Text copies with a machine-specific home/repository path have only that path
tokenized. Each `SOURCE_PROVENANCE.md` records original and archive SHA-256,
sizes, transformation, authority, supersession, and the local-only status of
bulky originals.

The BTK foundation manifest exposes 60 engineering source identifiers, catalog
row numbers, positions, HDF5 key names, and hashes, but no tensor values. It is
retained because exact engineering membership is required for replay and is
not a protected outcome partition. Scene-stratification tables contain compact
derived morphology/truth features; they are retained with explicit
nonoperational labeling.

## Run-directory authority classes

| Class | Examples | Treatment |
| --- | --- | --- |
| Authoritative scientific | Flux-free, P2, corrected stratification, null-space report | Archive final report/protocol/manifest/table/figure |
| Conditional/oracle authority | Oracle-flux 7/8 prior ladder | Preserve as an upper-bound contract; do not promote to ordinary inference |
| Valid negative | Prompted ResUNet, PU truth coverage, MH, ME, output mappings | Archive compact evidence; include in supplement/history |
| Valid partial | Recoverability head, PU stochastic behavior, direct PRE audit | Preserve exact metrics and failed gates |
| Correction | Grouped data, PU batch executor, external convergence | Preserve predecessor and correction; correction supersedes only its field |
| Preflight/engineering | BTK, Model-9, signed residual, D3 contract chain | Archive only decisive foundation/terminal records; ledger all attempts |
| Engineering-invalid | First flux-free attempt, invalid output-parameterization launches, fail-closed D3 launches | Ledger stop reason; no scientific outcome |
| Planning/status | Roadmaps and current status | Lower authority; point to canonical map |

## Large-file audit

At the exact decimal thresholds the audit records 340 files above 5 MB, 88
above 50 MB, and 75 above 100 MB. All 340 were already Git-ignored; none is an
unignored or tracked staging candidate. Every row records size, SHA-256,
category, protection status, ignore status, 50/100 MB flags, duplicate group,
and exclusion reason.

The largest file is the 6,291,456,128-byte grouped-retrain blended replay cache.
Other multi-GB artifacts include target-set HDF5, sampled PU/MH hypotheses,
scene HDF5, features, and two identical Galaxy10 HDF5 copies. All remain
local-only.

## Duplicate and superseded artifacts

- Two byte-identical source/package pairs are intentional archive exceptions:
  `reports/thayer_recoverability_v0.md` is a live hashed authority used by the
  flux-free driver and is also packaged as
  `experiment_archive/recoverability_nullspace/final_report.md`; the grouped
  correctness package repeats the already-tracked
  `reports/research_correctness_audit_final_report.md`. Removing either source
  or package copy would break clean-clone code authority or the required
  self-contained campaign package. No other exact duplicate is selected for
  the commit.
- The two Galaxy10 HDF5 paths have the same SHA-256 and size; this audit does
  not delete either.
- Repeated source-split manifests and PSF provenance tables form exact duplicate
  groups. One compact authority is archived where scientifically needed.
- D3 prospective cache tensors repeat across readiness/entrypoint campaigns.
  They remain ignored; hashes establish equality.
- Contact sheets duplicate figure-directory copies in the source-artifact
  audit. None is committed by this curation.
- Superseded reports and correction chains are not treated as byte duplicates;
  both are preserved when needed to understand the history.

## Protected and non-committable material

Never stage:

- raw datasets, downloaded survey catalogs, HDF5 or FITS;
- development, Atlas, lockbox, or isolated-source tensors;
- generated observations and dense source outputs;
- model checkpoints, optimizer states, semantic-state binaries;
- large NPY/NPZ arrays, embeddings, caches, and endpoint histories;
- raw logs, access traces, and temporary diagnostics;
- an archive or staged file above 100 MB;
- any file above 50 MB without a narrow written scientific justification.

No such justification is needed for this commit because no selected file is
above 5 MB.

## Figure index

| Figure class | Selected examples | Disposition |
| --- | --- | --- |
| Poster | flux-free frontier, corrected information-source comparison, metric-only intervention graphics | Curated archive; prompt-swap truth-panel grid remains local pending publication review |
| Main paper | grouped correction comparison, PSF frontier, information-source comparison | Curated archive |
| Supplement | loss-gradient heatmap, E1P prompt diagnosis, audit risk curve, training curves | Curated archive where compact |
| Archive-only local | full Atlas pages, endpoint grids, contact sheets, all training diagnostics | Ignore in Git |
| Exclude | duplicates, raw generated scenes, figures over 5 MB without unique narrative value | Local-only |

## Complete authority lookup

Use these artifacts together:

1. [`claim_authority_matrix.md`](claim_authority_matrix.md) for which evidence
   controls a claim;
2. [`supersession_ledger.md`](supersession_ledger.md) for conflicts;
3. [`experiment_ledger.csv`](experiment_ledger.csv) for all 124 run roots;
4. [`../experiment_archive/README.md`](../experiment_archive/README.md) for the
   34 curated campaign packages;
5. [`reproducibility_index.md`](reproducibility_index.md) for replay inputs and
   tests;
6. [`evidence_and_data_use_map.md`](evidence_and_data_use_map.md) for retention
   and publication use.
