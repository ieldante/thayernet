# D3 Scientific Artifact Contract

## Required pre-scientific inputs

An authoritative D3 run must freeze every value needed to evaluate its stop
rules before the scientific interpreter loads tensors. In addition to cached
features, P0 targets, initial decoder states, the D1 reference, mapping, loss,
assignment, and evaluator code, the forward and truth-coverage gates require:

- the exact scientific sky-electron vector;
- the global forward plausibility threshold;
- every per-band forward plausibility threshold; and
- the absolute relative-flux plausibility threshold.

Each item needs a semantic name, value, source artifact, source hash, canonical
serialization, and provenance showing that the isolated copy predates D3.
Synthetic evaluator fixtures do not satisfy this contract.

## Fail-closed behavior

Thayer-D3A found that the permitted D3B runtime freeze and D1R endpoint
evidence contain evaluator code hashes and pass metrics but not the scientific
forward-gate values. The historical source cannot be reopened under D3A's
no-Atlas rule. Dropping plausibility from truth coverage, inferring thresholds
from prior pass/fail results, or substituting synthetic values would change the
scientific contract.

The required response is a preregistration stop before third-party imports,
tensor loads, model construction, optimizer construction, or decoder forward.
The stop record must explicitly distinguish missing evaluator inputs from a D3
scientific failure and must leave both scaling routes unauthorized.

## Complete D3 persistence after resolution

Once the forward-gate inputs exist in an isolated, hashed artifact, a future
campaign must still persist its initial and one-step states, all scheduled and
semantic checkpoints, penultimate tensors, raw and square-mapped outputs,
scientific metrics, gradients and updates, optimizer state, replay manifest,
access proof, and postprocessing inputs. Artifact absence after a
pre-scientific stop is represented by explicit `NOT_RUN` records and must not
be mistaken for an incomplete executed fit.

## Thayer-D3C capsule resolution

Thayer-D3C packaged the complete small-value scientific contract in
`outputs/runs/thayer_d3_scientific_capsule_20260713_155637/`. The capsule
contains 97 resolved dependencies, including the exact sky vector, forward and
truth-coverage thresholds, numerical tolerances, units, semantics, and code
hashes. The four large scientific containers remain immutable hash references.

The schema, hash chain, corruption suite, 12-case evaluator comparison,
zero-I/O audit, and cwd/environment independence checks passed. No scientific
tensor, model, optimizer, or D3 step was executed. This resolves the contract
blocker and permits one new separately preregistered capsule-only D3 campaign;
it does not supply a D3 result or authorize broader data or a capacity ladder.

## Thayer-D3E member-level completion

The executable-contract audit found that capsule v1 did not express all
member-level requirements enforced by the downstream consumer. Capsule v2 now
binds the P0 target set, D1 endpoint, cached encoder features, and initial model
state by exact file and member identities, including names, shapes, dtypes,
endianness, semantic roles, and hashes. Header-only validation passed without
loading a scientific array value. The actual consumer rejected every tested
missing or corrupted member before model execution.

This resolves the artifact contract for execution but does not evaluate a
scientific D3 output.
