# Preregistration: Thayer-OC output-space conditioning

Frozen at UTC `2026-07-13T02:55:02.212718+00:00` before any per-scene HDF5 array load, detached gradient, curvature computation, or detached optimization. The authoritative 64 row IDs are listed below and identically persisted in `tables/frozen_row_ids.csv` (manifest SHA-256 `9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085`).

## Immutable scientific scope

This campaign keeps the exact Thayer-SA corrected scalar objective: requested reconstruction + companion reconstruction + weight-1 threshold-normalized scientific surrogate, with weight-1 ordinary concentration. Prompts, experts, and scenes are averaged. Ambiguous rows retain the exact hard minimum over identity and swap. Forward, source-sum, prompt-swap, and pair-equivalence remain evaluation-only. Targets, thresholds, source semantics, normalization, architecture, hard assignment, 64-row microset, coverage implementation, and 90% gates are immutable. There is no neural fitting, model-weight update, Atlas, validation, calibration, development, or lockbox access.

Frozen code hashes are `src/scientific_alignment.py` `62c0f1f7704a50a66b16c0044df7e140b3fae71563f1fa7db895f1d260655b07`, `src/output_conditioning.py` `989699a959aa03de25d45a5285de7a8abe6b1ab61dbfc255451526d41d09474a`, and `scripts/run_thayer_output_conditioning.py` `e1d9d58d12e8cec605deb98123b36e2d5d6d2dc9e84f5c01a746577561e2d207`. The Thayer-ME architecture hash is `9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415`. The threshold file hash is `a479a94bc1940b5fa146bc1a3eda3aeee6c931c90f25cc3a2108197486833e0a`.

## Exact physical coordinates and projection

For each expert, `T=S_req+S_comp`, `D=0.5(S_req-S_comp)`, `S_req=0.5T+D`, and `S_comp=0.5T-D`. COMMON perturbations add identical changes to requested and companion layers. ALLOCATION perturbations add equal and opposite changes and preserve their sum. The frozen projection decodes to physical sources, maps nonfinite values to zero, clamps each physical requested/companion pixel to `>=0`, and exactly re-encodes T/D. It uses no target beyond the ordinary supervised objective. C0 preserves the exact historical unprojected raw-space protocol; C1-C5 project after every accepted update.

## Initializations and deterministic seeds

All methods run from exactly five starts: persisted Thayer-SA `final_outputs.h5/source_sum_wrong_allocation` (file hash `a73ee3be59b54d0dacb7a82025c9d54fcb46b4d27916256096f5ecea813cb671`), persisted Thayer-ME physical expert decompositions converted by frozen scales (file hash `612d02fc72686ccde704ebf56bdb10c30af28fde3d20939afebac3fa34553446`), exact collapsed target means, exact 50/50 source-sum-preserving wrong allocations, and exact truths. No random restart is allowed. Campaign seed is `2026071304`; method seeds are C0..C5 = `2026071310`..`2026071315`.

## Optimizers and matched budgets

Every method has at most 401 corrected-objective evaluations, 400 corrected-objective gradient evaluations, 600 seconds per method/initialization, stopping gradient L2 tolerance `1e-8`, and trajectory logging at evaluation 0 and every 20 accepted/effective updates plus final. C0 is exact historical CPU float32 raw-output Adam: learning rate `1e-4`, zero weight decay, 400 updates, no projection. C1 is projected raw-space limited-memory BFGS with history 5, Armijo `c1=1e-4`, shrink 0.5, at most 8 trials, normalized-coordinate trust RMS 0.01, at most 120 accepted iterations. C2 optimizes physical T/D with projected Adam for 400 joint updates: per-band `lr_T[b]=1e-4*normalization_scale[b]`, `lr_D[b]=5e-4*normalization_scale[b]`, zero weight decay. C3 is projected joint physical-T/D L-BFGS with the C1 line search and physical trust RMS `0.01*median(normalization_scale)`. C4 has five frozen cycles, each 40 D-only Adam steps, 20 T-only steps, and 20 joint steps, using the C2 learning-rate formulas (400 total updates). C5 is projected physical-T/D Adam with the C2 rates; before each step its unchanged-objective gradients are multiplied by the median-normalized inverse absolute local scientific-surrogate Jacobian, `clip(median_positive(|J|)/(|J|+1e-8),0.1,10)`. C5 may use 400 auxiliary surrogate-Jacobian gradients but no coverage-adaptive information.

## Geometry, trajectories, and stopping

At exact truth and the persisted Thayer-SA compromise, raw/common/allocation gradients, per-band and per-expert norms, hard-assignment margins, positivity saturation, deterministic common/allocation Hessian-vector curvatures, finite-difference agreement at `h=1e-3`, modal curvature ratio, and a two-mode local condition estimate are fixed diagnostics. No dense Hessian is allowed. Every trajectory records the corrected objective and components, exact frozen coverage and distances, smooth maximum, common/allocation step and gradient norms, assignments and margins, evaluation-only forward consistency, expert diameter, and projection fraction. Nonfinite objectives, gradients, or outputs stop that run as numerical instability. No condition is retuned after results.

## Frozen success and interpretation

A method passes only if every truth-start row remains stationary/fully covered and, for every non-truth initialization, final ordinary own, ambiguous own, alternate, and both-mode coverage each reach at least 29/32 (>=90%), while objective, assignment, thresholds, targets, and protected-data boundaries remain unchanged. One global method must pass; per-scene method selection is forbidden. Partial success requires a materially nonzero coverage increase of at least 0.20 absolute over C0 for the same initialization while a 90% gate remains unmet. Failure includes no material improvement, truth instability, insufficient allocation conditioning, assignment barrier, projection barrier, or basin extremity. Exactly one primary category will be selected from the eight specified categories, with `MIXED CAUSE` only under direct multiple-mechanism evidence. Exactly one next experiment will be recommended and not run.

## Gate attainability

Every rate gate is an integer 29/32 requirement and exact persisted truths previously achieved 32/32 for all four coverage metrics. The corrected objective has exact truth at zero with zero gradient. The T/D map is bijective before projection and exact truths are feasible nonnegative points. Therefore every numerical and coverage gate is mathematically attainable. The detailed audit is `tables/preregistered_gate_attainability.csv`.

## Frozen row IDs

- `0`: `pu_training_ordinary_00000` (source index `0`, `ordinary`, pair `none`)
- `1`: `pu_training_ordinary_00001` (source index `1`, `ordinary`, pair `none`)
- `2`: `pu_training_ordinary_00002` (source index `2`, `ordinary`, pair `none`)
- `3`: `pu_training_ordinary_00003` (source index `3`, `ordinary`, pair `none`)
- `4`: `pu_training_ordinary_00004` (source index `4`, `ordinary`, pair `none`)
- `5`: `pu_training_ordinary_00005` (source index `5`, `ordinary`, pair `none`)
- `6`: `pu_training_ordinary_00006` (source index `6`, `ordinary`, pair `none`)
- `7`: `pu_training_ordinary_00007` (source index `7`, `ordinary`, pair `none`)
- `8`: `pu_training_ordinary_00008` (source index `8`, `ordinary`, pair `none`)
- `9`: `pu_training_ordinary_00009` (source index `9`, `ordinary`, pair `none`)
- `10`: `pu_training_ordinary_00010` (source index `10`, `ordinary`, pair `none`)
- `11`: `pu_training_ordinary_00011` (source index `11`, `ordinary`, pair `none`)
- `12`: `pu_training_ordinary_00012` (source index `12`, `ordinary`, pair `none`)
- `13`: `pu_training_ordinary_00013` (source index `13`, `ordinary`, pair `none`)
- `14`: `pu_training_ordinary_00014` (source index `14`, `ordinary`, pair `none`)
- `15`: `pu_training_ordinary_00015` (source index `15`, `ordinary`, pair `none`)
- `16`: `pu_training_ordinary_00016` (source index `16`, `ordinary`, pair `none`)
- `17`: `pu_training_ordinary_00017` (source index `17`, `ordinary`, pair `none`)
- `18`: `pu_training_ordinary_00018` (source index `18`, `ordinary`, pair `none`)
- `19`: `pu_training_ordinary_00019` (source index `19`, `ordinary`, pair `none`)
- `20`: `pu_training_ordinary_00020` (source index `20`, `ordinary`, pair `none`)
- `21`: `pu_training_ordinary_00021` (source index `21`, `ordinary`, pair `none`)
- `22`: `pu_training_ordinary_00022` (source index `22`, `ordinary`, pair `none`)
- `23`: `pu_training_ordinary_00023` (source index `23`, `ordinary`, pair `none`)
- `24`: `pu_training_ordinary_00024` (source index `24`, `ordinary`, pair `none`)
- `25`: `pu_training_ordinary_00025` (source index `25`, `ordinary`, pair `none`)
- `26`: `pu_training_ordinary_00026` (source index `26`, `ordinary`, pair `none`)
- `27`: `pu_training_ordinary_00027` (source index `27`, `ordinary`, pair `none`)
- `28`: `pu_training_ordinary_00028` (source index `28`, `ordinary`, pair `none`)
- `29`: `pu_training_ordinary_00029` (source index `29`, `ordinary`, pair `none`)
- `30`: `pu_training_ordinary_00030` (source index `30`, `ordinary`, pair `none`)
- `31`: `pu_training_ordinary_00031` (source index `31`, `ordinary`, pair `none`)
- `32`: `pu_training_near_00000` (source index `12000`, `near_collision`, pair `pu_training_pair_00001`)
- `33`: `pu_training_near_00001` (source index `12001`, `near_collision`, pair `pu_training_pair_00001`)
- `34`: `pu_training_near_00002` (source index `12002`, `near_collision`, pair `pu_training_pair_00002`)
- `35`: `pu_training_near_00003` (source index `12003`, `near_collision`, pair `pu_training_pair_00002`)
- `36`: `pu_training_near_00004` (source index `12004`, `near_collision`, pair `pu_training_pair_00003`)
- `37`: `pu_training_near_00005` (source index `12005`, `near_collision`, pair `pu_training_pair_00003`)
- `38`: `pu_training_near_00006` (source index `12006`, `near_collision`, pair `pu_training_pair_00004`)
- `39`: `pu_training_near_00007` (source index `12007`, `near_collision`, pair `pu_training_pair_00004`)
- `40`: `pu_training_near_00008` (source index `12008`, `near_collision`, pair `pu_training_pair_00005`)
- `41`: `pu_training_near_00009` (source index `12009`, `near_collision`, pair `pu_training_pair_00005`)
- `42`: `pu_training_near_00010` (source index `12010`, `near_collision`, pair `pu_training_pair_00006`)
- `43`: `pu_training_near_00011` (source index `12011`, `near_collision`, pair `pu_training_pair_00006`)
- `44`: `pu_training_near_00012` (source index `12012`, `near_collision`, pair `pu_training_pair_00007`)
- `45`: `pu_training_near_00013` (source index `12013`, `near_collision`, pair `pu_training_pair_00007`)
- `46`: `pu_training_near_00014` (source index `12014`, `near_collision`, pair `pu_training_pair_00008`)
- `47`: `pu_training_near_00015` (source index `12015`, `near_collision`, pair `pu_training_pair_00008`)
- `48`: `pu_training_near_00016` (source index `12016`, `near_collision`, pair `pu_training_pair_00009`)
- `49`: `pu_training_near_00017` (source index `12017`, `near_collision`, pair `pu_training_pair_00009`)
- `50`: `pu_training_near_00018` (source index `12018`, `near_collision`, pair `pu_training_pair_00010`)
- `51`: `pu_training_near_00019` (source index `12019`, `near_collision`, pair `pu_training_pair_00010`)
- `52`: `pu_training_near_00020` (source index `12020`, `near_collision`, pair `pu_training_pair_00011`)
- `53`: `pu_training_near_00021` (source index `12021`, `near_collision`, pair `pu_training_pair_00011`)
- `54`: `pu_training_near_00022` (source index `12022`, `near_collision`, pair `pu_training_pair_00012`)
- `55`: `pu_training_near_00023` (source index `12023`, `near_collision`, pair `pu_training_pair_00012`)
- `56`: `pu_training_near_00024` (source index `12024`, `near_collision`, pair `pu_training_pair_00013`)
- `57`: `pu_training_near_00025` (source index `12025`, `near_collision`, pair `pu_training_pair_00013`)
- `58`: `pu_training_near_00026` (source index `12026`, `near_collision`, pair `pu_training_pair_00014`)
- `59`: `pu_training_near_00027` (source index `12027`, `near_collision`, pair `pu_training_pair_00014`)
- `60`: `pu_training_near_00028` (source index `12028`, `near_collision`, pair `pu_training_pair_00015`)
- `61`: `pu_training_near_00029` (source index `12029`, `near_collision`, pair `pu_training_pair_00015`)
- `62`: `pu_training_near_00030` (source index `12030`, `near_collision`, pair `pu_training_pair_00016`)
- `63`: `pu_training_near_00031` (source index `12031`, `near_collision`, pair `pu_training_pair_00016`)
