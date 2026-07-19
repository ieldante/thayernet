# Don’t Even Try: Galaxy Deblending Recoverability

This repository studies a narrower question than “can a model produce a
plausible deblend?”: **when is a coordinate-requested galaxy scientifically
determined by the available image, priors, and auxiliary measurements?** It
contains the Thayer neural baselines, prompt and ambiguity studies, structured
Model-9 solver, global identifiability audits, acquisition interventions, and a
compact evidence-linked archive of the complete Phase-I research program.

The work uses controlled synthetic Galaxy10 and CatSim/BTK scenes. It is a
research testbed, not a survey-grade deblender or deployed safety system.

## Phase I status

Phase I exploratory recoverability map is complete for the frozen eight-scene
mechanistic set.

Headline frozen-contract results:

- unrestricted two-source allocation has an exact 10,800-dimensional null
  space in each `3×60×60` scene;
- Condition C achieved 98.0% prompt-swap success, establishing promptability
  but not source recovery;
- hard morphology plus oracle source photometry/truth-derived noise information
  was strict `UNIQUE` in 7/8 scenes;
- the valid flux-free morphology contract was strict `UNIQUE` in 0/8 scenes;
- the P2-versus-S1 composite information/geometry criterion improved in 15/16
  family fits but remained 0/8 strict unique; after the same-PSF S2 control,
  only 3/16 gains were PSF-specific;
- external total photometry was helpful relative to P2 in Scenes 0, 5, 51, and
  73 and not helpful in 3, 6, 18, and 81: 4/8, exact 95% CI 15.7%–84.3%.

These contracts are not interchangeable. In particular, the 7/8 oracle and
0/8 flux-free results also differ in renderer, noise handling, and scientific
gates, so they are not a clean remove-photometry-only causal ablation. The
eight-scene low-`|ΔB/T|` rule is exploratory and truth-derived.

## Canonical documentation

- [Complete research program map](docs/thayer_research_program_map.md)
- [Evidence and data-use map](docs/research_archive/evidence_and_data_use_map.md)
- [Claim authority matrix](docs/research_archive/claim_authority_matrix.md)
- [Proof and validation matrix](docs/research_archive/proof_and_validation_matrix.md)
- [Improvement opportunity map](docs/research_archive/improvement_opportunity_map.md)
- [Reproducibility index](docs/research_archive/reproducibility_index.md)
- [Curated experiment archive](docs/experiment_archive/README.md)
- [Phase-II validation plan](docs/research_archive/phase_ii_validation_plan.md)

## Audit-layer status

| Layer | Current status |
| --- | --- |
| PRE query-validity/ambiguity audit | Useful research component; formal gate not fully passed |
| Global identifiability audit | Strongest functioning audit within the frozen simulation contracts |
| POST safety classifier | Not operational; eligible neural outputs supplied no meaningful safe-positive class |

The intended future decision is to reconstruct only after a valid PRE query and
a global identifiability pass; otherwise acquire prespecified information or
abstain. No current result authorizes real-survey deployment.

## Phase II

Independent-scene validation remains unexecuted. The frozen plan excludes all
eight discovery scenes, matches starts and compute budgets, prespecifies
thresholds and statistical tests, and prohibits post-hoc tuning. See the
[Phase-II plan](docs/research_archive/phase_ii_validation_plan.md).

## Repository layout

```text
configs/                  portable early experiment defaults
docs/                     canonical map, historical docs, curated archive
notebooks/                original notebook workflow
reports/                  compact repository-level reports and paper skeleton
scripts/                  campaign, audit, replay, and archive drivers
src/                      neural, structured-solver, and audit implementations
tests/                    unit, contract, replay, and artifact tests
```

Raw datasets, generated observations, protected tensors, checkpoints, dense
endpoint arrays, caches, and logs are intentionally excluded from Git. Their
roles, hashes, and retention rules are documented in the
[data-use map](docs/research_archive/evidence_and_data_use_map.md) and
[large-file audit](docs/research_archive/large_file_audit.csv).

## Environment and reproduction

The root requirements are broad and unpinned; no exact repository-wide lockfile
exists. Historical campaign reports/manifests carry the available runtime
records, and exact neural/structured replay requires the ignored local
artifacts whose hashes were audited. A basic environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Do not interpret a clean-clone test result as a full historical replay: some
tests intentionally require local `outputs/runs` artifacts or environment-
selected scientific inputs.
