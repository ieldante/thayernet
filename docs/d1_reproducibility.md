# D1 Reproducibility

The authoritative square-D1 free-feature result is exactly reproducible for
the frozen ambiguous scene. Metadata reproduced before optimization, the
initial functional-head state was byte-identical, and every one of the 54
scheduled trajectory hashes matched.

The two detached paired-prompt tensors contained 230,400 optimized scalars.
Both final heads, all targets, the mapping, and the evaluator remained frozen.
The replay again reached 100% own, 100% alternate, and 100% both-mode coverage,
with prompt swap and forward consistency at 100% and unchanged identity/swap
assignments.

Fresh-process regeneration and 13 batch/serialization checks passed with zero
numerical difference. All four endpoint tensors, raw logits, mapped outputs,
physical decompositions, optimizer provenance, semantic metadata, and
canonical hashes are now persisted.

The previous D3 attempt remains a valid fail-closed no-result campaign: it
stopped because these feature tensors were absent. Thayer-D1R resolves that
artifact gap but does not retroactively create a D3 trajectory or establish
decoder sufficiency. The next justified experiment is exactly one separately
preregistered square-only D3 run on the same scene and frozen contract.

D3 was not run in Thayer-D1R.
