# Current project status

## Scientific status

Phase I exploratory recoverability map is complete for the frozen eight-scene
mechanistic set. Independent-scene validation remains unexecuted.

The governing results are:

- unrestricted additive source allocation: analytic nullity 10,800 per frozen
  `3×60×60` scene;
- coordinate prompting: 98.0% prompt-swap success, which establishes
  promptability rather than recovery;
- oracle structural contract: 7/8 strict `UNIQUE`, conditional on exact
  per-source g/r/z flux and truth-derived signed-noise information;
- valid flux-free contract: 0/8 strict `UNIQUE`, with best-family outcomes six
  `NEAR_UNIQUE` and two `PARTIALLY_IDENTIFIABLE`;
- P2 PSF-diverse contract: 0/8 strict unique, despite P2-versus-S1 composite
  information/geometry improvement in 15/16 family fits; only 3/16 gains were
  PSF-specific after the same-PSF S2 control;
- external total photometry versus P2: helpful for Scenes 0, 5, 51, and 73;
  not helpful for 3, 6, 18, and 81; descriptive rate 4/8, exact 95% CI
  15.7%–84.3%.

The oracle and flux-free contracts differ beyond source photometry, so the
7/8-to-0/8 transition is not a clean one-variable causal ablation. The
low-`|ΔB/T|` separator is exploratory, truth-derived/post-fit, and not an
operational routing rule.

## Audit status

| Layer | Status |
| --- | --- |
| PRE | Useful research component for query validity and ambiguity; formal gate not fully passed |
| Identifiability | Strongest functioning audit under frozen simulation contracts |
| POST | Blocked and nonoperational; repaired PU execution yielded 0 safe / 7,591 unsafe outputs |

## Evidence authority

Use the [complete program map](thayer_research_program_map.md),
[claim matrix](research_archive/claim_authority_matrix.md), and
[supersession ledger](research_archive/supersession_ledger.md) for current
interpretation. The [curated archive](experiment_archive/README.md) preserves
compact reports, protocols, manifests, tables, figures, and provenance.

The long append-only status text that existed immediately before canonical
curation is preserved unchanged in
[`current_status_pre_canonical_archive.md`](research_archive/current_status_pre_canonical_archive.md).

## Next authorized scientific work

Execute the frozen [Phase-II independent validation plan](research_archive/phase_ii_validation_plan.md)
only on fresh scenes that exclude the eight Phase-I cases. Match starts and
budgets across S1, total-photometry, and P2 conditions; freeze thresholds and
tests; do not tune on validation outcomes. Real-survey claims remain
unauthorized until robustness and closure studies succeed.
