# Reproducibility index

## Reproduction levels

| Level | Meaning | Current coverage |
| --- | --- | --- |
| R0 — evidence verification | Parse compact authorities, verify provenance hashes, recalculate headline values from committed tables | Complete for the curated archive |
| R1 — deterministic analysis replay | Re-run analysis/decision logic from frozen compact or local inputs without training | Supported for null-space, Model-9, identifiability, acquisition, audit, and many integrity campaigns; local tensors required where noted |
| R2 — model inference replay | Recreate frozen outputs from checkpoints and local scene manifests | Local-only because checkpoints and dense scenes are intentionally excluded from Git |
| R3 — training reproduction | Retrain a model from raw/catalog data and frozen splits | Possible for many families with local data/environment, but expensive and not required to verify this archive |
| R4 — independent scientific validation | Reimplement/run on unseen independent scenes | Not executed; planned in [`phase_ii_validation_plan.md`](phase_ii_validation_plan.md) |

This curation did not rerun scientific campaigns. Validation of the commit is
limited to syntax, tests, schema/link/hash/privacy/protection checks, and
read-only recomputation from compact tables.

## Repository and environment anchors

| Anchor | Location / value | Role |
| --- | --- | --- |
| Pre-archive HEAD | `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174` | Historical code base for all uncommitted campaigns |
| Backup branch | `backup/pre-thayer-canonical-archive-20260719T215720Z` | Protects pre-audit Git history only |
| Python dependencies | `requirements.txt`, `pyproject.toml` | Broad environment hints only: 15 root requirements are unpinned, `pyproject.toml` contains tool configuration rather than a project environment, and no lockfile exists |
| Portable config | `configs/default.yaml` | Early benchmark defaults |
| Historical scientific runtime | Several decisive MPS campaigns report Python 3.9.6, PyTorch 2.8.0, Apple MPS | Exact campaign-specific environment is recorded in run reports/manifests; there is no single universal environment lock |
| Raw Galaxy10 input | Local `data/Galaxy10_DECals.h5`, hash in large-file audit | Early RGB benchmark input; not committed |
| CatSim/BTK inputs | Local catalog/scene containers and compact manifests | Structured simulated scenes; not committed as tensors |
| Archive provenance | One `SOURCE_PROVENANCE.md` per curated campaign | Original/archive SHA-256, size, path-token transformation, authority and supersession |

## Implementation and validation map

| Scientific component | Primary implementation | Drivers / audits | Focused tests | Required local artifacts |
| --- | --- | --- | --- | --- |
| Early direct/residual/balanced models | `src/models.py`, `src/train.py`, `src/blend.py`, `src/data.py` | `scripts/train_residual_unet.py`, `scripts/train_balanced_residual_unet.py`, `scripts/train_grouped_v02.py`, evaluation/audit scripts | Existing metric, leakage, FITS, and Thayer-Select tests | Galaxy10 HDF5 and local checkpoints for inference replay |
| BTK scene foundation | `src/btk_scene.py`, `src/prompt_semantics.py` | `scripts/run_btk_engineering_pilot.py`, `scripts/run_thayer_select_btk_smoke.py` | `tests/test_thayer_select.py`, provenance/split tests | BTK environment and CatSim catalog |
| Coordinate prompting / Condition C | `src/coordinate_prompt.py`, `src/models_thayer_select.py`, `src/prompt_semantics.py` | prompt-ablation prepare/train/finalize scripts | `tests/test_thayer_select.py` | Frozen scene manifests and checkpoint for exact inference |
| Recoverability and hierarchy | `src/recoverability.py`, `src/hierarchical_safety.py`, `src/hierarchical_feasibility.py`, calibration modules | recoverability, hierarchy, calibration, scale, shape, observability and PSF scripts | recoverability/hierarchy/calibration/scale/shape tests | Ignored features, labels and checkpoints for exact historical replay |
| Ambiguity Atlas / competing hypotheses | `src/competing_hypotheses.py` | `scripts/build_ambiguity_atlas.py`, `scripts/audit_ambiguity_atlas_v0.py`, evaluation/finalization scripts | `tests/test_ambiguity_atlas.py`, `tests/test_competing_hypotheses.py` | Protected Atlas arrays only for full replay; compact tables verify reported outcomes |
| Prompted ResUNet | `src/models_prompted_resunet.py` | prepare/train/evaluate/finalize scripts | `tests/test_prompted_resunet.py` | Local training scenes and checkpoint |
| Probabilistic U-Net / PU correction | `src/models_probabilistic_unet.py`, `src/canonical_tensor_hash.py` | PU prepare/train/evaluate/finalize, eligibility and batch-repair scripts | `tests/test_probabilistic_unet.py`, `tests/test_thayer_pu_eligibility_v1.py`, hash tests | Local scenes, latents, aligned outputs and checkpoints |
| Multi-hypothesis / two expert | `src/models_multiple_hypotheses.py`, `src/models_two_expert_decoder.py` | preparation/training/evaluation/audit/finalize scripts | `tests/test_multiple_hypotheses.py`, `tests/test_two_expert_decoder.py` | Local target sets and checkpoints |
| Loss geometry and scientific alignment | `src/loss_geometry.py`, `src/scientific_alignment.py` | loss-geometry and scientific-alignment bootstrap/audit/finalize scripts | `tests/test_loss_geometry.py`, `tests/test_scientific_alignment.py` | Persisted local outputs for exact trajectory replay; compact tables for R0 |
| Output conditioning / feasibility / mapping | `src/output_conditioning.py`, `src/feasibility_projection.py`, `src/output_parameterization.py` | corresponding bootstrap/run/close/finalize scripts | output-conditioning, feasibility, and parameterization tests | Local microset targets and outputs |
| D0–D3 / executable policy chain | `src/d3_*`, `src/canonical_tensor_hash.py` | D3 bootstrap, capsule, readiness, execution, policy, replay and validator scripts | D3 contract/policy/protocol/serialization/integration suites | Local semantic tensors/checkpoints for full R1/R2; compact R3 report/JSON for result audit |
| Direct PRE/POST audit | `src/direct_catalog_safety_auditor.py` | `scripts/bootstrap_thayer_audit_v0.py`, `scripts/run_thayer_audit_v0.py` | direct-auditor and artifact tests | Local audit episodes and model outputs; compact metrics establish label collapse |
| Family-E/E1/E1P | `src/family_e.py`, `src/family_e1.py`, `src/family_e1p.py`, `src/family_e_signed_residual.py` | Family-E/E1/E1P bootstrap/run/finalize and signed-residual audit | Family-E/E1/E1P and physical-contract tests | Local micro scenes/checkpoints for exact replay |
| Unrestricted null-space analysis | Analytic report plus `scripts/analyze_thayer_recoverability_v0.py` | same analysis script | related Model-9/family tests | Eight frozen local scene tensors; report supplies equations and hashes |
| Structured Model-9 solver | `src/model9_structured.py`, `src/model9_optimizer.py`, `src/model9_galsim_adapter.py`, `src/model9_joint.py`, `src/model9_synthetic.py` | `scripts/analyze_thayer_identifiability_v1.py`, `scripts/run_thayer_flux_free_identifiability_v0.py`, preparation/validation scripts | `tests/test_model9_foundation.py`, `tests/test_model9_joint.py` | Local BTK observations for scientific replay; synthetic fixtures for implementation validation |
| PSF-diverse acquisition | `src/psf_diverse_acquisition.py` | `scripts/run_thayer_psf_diverse_flux_identifiability_v0.py` | `tests/test_psf_diverse_acquisition.py` | Paired local observations; compact metrics/manifest in archive |
| External photometry | Model-9 modules plus measurement likelihood in campaign driver | preflight/correction scripts; archived scene-stratification and correction drivers | Model-9 and acquisition tests plus schema/replay validation | Compact measurement table committed; local observations/fit records for full replay |
| Archive construction | `scripts/build_research_archive_inventory.py`, `scripts/build_curated_experiment_archive.py` | metadata-only builders | Validated by schema/hash/privacy/protection audit | Workspace metadata and ignored source authorities |

## Canonical replay sequence

1. Verify repository/branch identity and the pre-archive record.
2. Parse all curated JSON and CSV files.
3. Recalculate hashes from every campaign's provenance table.
4. Validate that archive text contains no machine-specific paths or secrets.
5. Confirm no banned tensor/checkpoint/log type or file above 5 MB entered the
   archive.
6. Recalculate central counts from committed tables: 7/8 oracle-contract
   unique, 0/8 flux-free strict unique, 0/8 P2 strict unique, and the corrected
   4/8 helpful split.
7. Reproduce local/global counterexamples from rank/class/condition tables.
8. Run Python syntax checks and the focused/full test suite appropriate to the
   available environment.
9. Only with local scientific inputs, use each frozen protocol and driver for
   exact analysis replay. Do not substitute archive tables as inference inputs.

## Hash and archive verification

The curated builder rejects HDF5, FITS-adjacent dense artifacts, NPY/NPZ,
PT/PTH/checkpoints, logs, and files above 5 MB. Text path tokenization is
byte-preserving except for the exact replaced path strings, and both original
and archive hashes are recorded. The verification rule is:

- `byte_exact`: original SHA-256 must equal archive SHA-256;
- `machine_paths_tokenized`: original SHA-256 proves source identity and
  archive SHA-256 proves the reviewed sanitized copy;
- every non-provenance archive file must appear exactly once in its campaign
  provenance table.

## Known reproducibility gaps

- Early Direct/Residual/BR runs have no run-level final reports/manifests; the
  experiment log is their main surviving narrative.
- Directory suffixes mix local and UTC conventions; use frozen/provenance UTC.
- The grouped RGB result has one clean training seed and no untouched final
  source pool.
- Full neural inference/training replay requires excluded checkpoints and dense
  scene data.
- The first Atlas had one meaningful model-family cluster; cross-family audit
  transfer is unproven.
- Many D3 engineering attempts are reproducibility history, not distinct
  scientific results; parameter counts describe different optimized objects.
- The valid R3 D3 result has no compact original root-level scientific table or
  figure; the final report and run-local JSON carry the outcome.
- The 7/8 oracle and 0/8 flux-free contracts differ beyond photometry.
- External photometry used four starts while S1/P2 used 16; Phase II must match
  budgets.
- The low-|ΔB/T| rule is truth-derived/post-fit and has no validated observable
  proxy.
- No independent-scene or real-survey validation exists.
- No exact repository-wide environment lock exists; campaign reports/freezes
  are the available records of historical local runtimes.
- Nineteen tests reference the ignored `outputs/runs` tree and 21 require
  environment-selected or other local artifacts, so a clean clone cannot run
  the entire historical-artifact suite without restoring those inputs.
- Seven D3 modules have no direct textual test reference
  (`d3_checkpoint_adapter_v41r1.py`, `d3_contract_tokens_v41.py`,
  `d3_contract_tokens_v41r1.py`, `d3_execution_mode_contract_r1.py`,
  `d3_hash_callsite_r1.py`, `d3_policy_registry.py`, and
  `d3_tensor_hash_contract_r1.py`); they may be exercised transitively, so this
  is a direct-reference gap rather than a claim that they are untested.

## Historical checkpoint policy

Checkpoints remain local and excluded. Campaign reports variously inventoried
600, 743, or 767 historical checkpoints because their scopes and timing
differ; these numbers must not be silently merged into one universal count.
For any replay, use the checkpoint inventory attached to that campaign, verify
before/after hashes, and distinguish protected historical checkpoints from
campaign-created micro-only states. The broad initial extension/path scan found
769 checkpoint-like files, which is an archive inventory count rather than a
scientific authority count.
