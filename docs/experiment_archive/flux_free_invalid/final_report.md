# Thayer-Flux-Free-Identifiability-v0 final report

## Campaign decision

**Outcome: `FLUX_FREE_INVALID`.**

The campaign stopped at the authorization gate before scientific execution.
The repository did not contain a `Thayer-Model-9-Preparation-v0` final report,
the required exact readiness decision `MODEL_9_FOUNDATION_READY`, the draft
flux-free protocol attributed to that preparation campaign, or preparation
tests that could be rerun. The user contract says to proceed only if that
preparation decision exists and to stop fail-closed when any gate requirement
fails. No structured solver was instantiated, no observation tensor was read,
and no optimization or identifiability classification was attempted.

This is a contract-invalid result, not evidence that flux-free uniqueness
collapses. The Level-4, Level-5, and union unique counts are **not estimable**;
they must not be reported as zero. Whether the previous oracle-flux 7/8 result
survives is therefore **not assessed**.

## Exact findings

1. The authoritative `Thayer-Identifiability-v1` report and prior freeze are
   present and hash to their recorded values. They establish the previous
   conditional result: Level 4 was unique for scenes 0, 3, 5, and 73; Level 5
   added scenes 6, 18, and 81; scene 51 was outside support. They also state
   explicitly that Levels 4 and 5 fixed isolated-truth per-source g/r/z fluxes.
2. `Thayer-Recoverability-v0` and the authoritative signed-residual contract
   are present. Their hashes were recorded without reading protected arrays.
3. No file or report in the repository contains the exact token
   `MODEL_9_FOUNDATION_READY`. No path or report named for
   `Thayer-Model-9-Preparation-v0` or a flux-free preparation protocol was
   found. The mandatory `Thayer-Project-Synthesis-v1` artifact was likewise
   not present under that title.
4. Because the preparation foundation is absent, the required preparation
   tests, renderer round trips, finite-difference checks, replay checks,
   nonnegativity checks, PSF-normalization checks, inference-path oracle audit,
   and numerical freeze cannot be independently rerun as a coherent Model-9
   gate. Existing older reports are not a substitute for the missing gate.
5. The Git index was empty, README had no diff, and 376 pre-existing worktree
   entries were present before this run. They are user-owned and were not
   modified. Historical files hashed in the manifests remained unchanged at
   closure.
6. Development access was zero. Protected Atlas access was zero. Lockbox
   access was zero. Frozen observation-array access was zero. Neural training,
   neural imports, and optimizer steps were zero.

## Solver information and oracle removal

The solver received **no inputs**, because no solver was allowed to run. The
planned controlled inputs—blended g/r/z observation, coordinate prompts,
known PSF, geometry, noise convention, and observation-derived likelihood—were
not supplied to an inference process. Isolated source images, isolated masks,
true source parameters, true morphology labels, and true per-source fluxes
were not supplied either. Historical isolated-source hashes were copied from
the prior campaign solely to identify the frozen scenes; they were never used
as inference values.

Accordingly, this run did not test removal of the oracle assumption. It only
established that the prerequisite needed to perform that test reproducibly is
missing from the repository state presented to the campaign.

## Scene and family classifications

Every one of the eight frozen scenes at both permitted families is classified
`INVALID_CONTRACT`, exactly once per scene/family, because the common
authorization prerequisite is absent. These are not non-unique scientific
solutions and are not assigned information-theoretic, support, numerical, or
optimization failure causes.

| Scene | Level 4 | Level 5 | Minimum family | Current reason |
| ---: | --- | --- | --- | --- |
| 0 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |
| 3 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |
| 5 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |
| 6 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |
| 18 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |
| 51 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness; prior support failure not retested |
| 73 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |
| 81 | INVALID_CONTRACT | INVALID_CONTRACT | not estimable | Missing preparation readiness |

Primary endpoint status:

- Level-4 unique count: not estimable
- Level-5 unique count: not estimable
- Union unique count: not estimable
- Previous conditional oracle-flux union count: 7/8
- Observation-only survival of 7/8: not assessed

## Interpretation

The previous 7/8 result justifies a rigorously prepared flux-free structural
test, but this invalid run supplies no new evidence that morphology plus the
observation identifies source photometry. A morphology-aware direct model
therefore remains a scientifically motivated hypothesis, not an authorized
next-stage reconstruction method. PriorNet training is **not authorized**.
The POST audit layer cannot meaningfully return until a valid flux-free target
and a successful direct structural solver exist.

No claim is made about real-survey generalization.

## Exactly one recommended next experiment

Run **Thayer-Model-9-Preparation-v0** to completion in an append-only campaign
that produces the missing final report, exact `MODEL_9_FOUNDATION_READY`
decision if warranted, executable preparation tests, and a single frozen
flux-free protocol. Do not begin scientific flux-free fitting, Direct
Structural Solver work, PriorNet, or POST work until that gate passes.

## Artifact note

The required tables and figure paths are present as fail-closed audit
artifacts. Numerical cells are `NA`, endpoint/flux tables contain no data
rows, and the frontier image states that execution was not authorized. No
empty admissible set or absent run is counted as unique.
