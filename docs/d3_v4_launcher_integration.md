# D3 v4 launcher integration

Thayer-D3I reproduced the historical launcher regression before correction:
bundle-v3 policy preflight passed, while the scientific branch still resolved
bundle v2 directly and named two absent subprocess files. Six deliberately
pre-fix tests failed as expected.

The append-only v4 path consists of `scripts/run_thayer_scientific_d3_v4.py`,
`scripts/run_thayer_scientific_d3_process_v4.py`,
`scripts/run_thayer_d3_postprocess_v4.py`, and `src/d3_execution_bridge_v4.py`.
Bundle v3 governs policy and continuation; bundle v2 is resolved only through v3
and governs architecture/numerics; capsule v1 governs scientific values; runtime
readiness governs isolation.

The actual v4 synthetic CLI passed 180 requirements, 16 policies,
model/optimizer construction, forward/backward/step, semantic persistence,
fresh-process checkpoint replay, isolated postprocessing, exact flow closure,
and 25/25 corruption rejections. The frozen bridge SHA-256 is
`3ab6e4a525297f48cc7fd9428651c604aa1236ed0a4425f9953c5b5772345dc5`.

The mandatory scientific continuation then stopped on frozen
implementation/runtime events before model construction. The frozen sources were
not changed or retried.
