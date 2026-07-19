# D3 v4.1 R1 independent contract compliance

Thayer-D3I41R1 froze a 30-row audit-derived ledger and collected, executed,
and passed all 19 exact test names from the authoritative CSV. Candidate
self-attestation was not accepted as eligibility evidence.

The complete dtype and production-checkpoint corrections passed focused and
cold-process checks. Candidate 001 failed at direct worker import and was
preserved. Candidate 002 was then frozen, but the orchestrator collision-refused
an already existing candidate-001 synthetic log before launching the worker.
Because log isolation is outside the authorized R1 corrections, the campaign
stopped without independent eligibility or scientific payload access.

Outcome: `IMPLEMENTATION_OR_CONTRACT_FAILURE`. Authorization: `none`.
Evidence is under `outputs/runs/thayer_d3_i41r1_20260713_221426/`.
