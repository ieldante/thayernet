# Direct Scientific-Feasibility Projection preregistration

Working name: **Thayer-FP (Thayer Feasibility Projection)**  
Frozen at UTC: `2026-07-13T03:42:22.031886+00:00`  
Microset manifest SHA-256: `9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085`  
Frozen-row table SHA-256: `bddf95524dddc1abf7715776f36c4166bc2a6b4e13729dca2b5e39f8fd68950c`

## Scope and immutable boundary

Thayer-FP uses exactly the 64 training-only Thayer-ME microset observations, the persisted Thayer-ME expert outputs as initial candidates, and the unchanged approved targets. It tests offline projection into the unchanged scientific region, then only if the projection gate passes tests whether the unchanged 165,612-parameter Thayer-ME can memorize those training-only representatives. Atlas, validation, calibration, development, and lockbox arrays are prohibited. Truth and constraint values may construct offline targets but may never enter model inference tensors. Projections are training-only representatives, not new astronomical truth.

## Frozen outputs, targets, assignment, and constraints

Initial candidates are file `outputs/runs/thayer_two_expert_decoder_20260712_203121/diagnostics/micro_overfit_20260712_203540/expert_outputs/micro_final_decompositions.h5`, dataset `decompositions`, hash `612d02fc72686ccde704ebf56bdb10c30af28fde3d20939afebac3fa34553446`. Target file hash is `7fc92222ff2d980c4beb787b961fa7bdaf3130c055ce842dc8fd5f600c29c19a`. The full six-channel contract is requested g/r/z followed by companion g/r/z, zero background, exact 60 x 60 dimensions and band order, finite values, and nonnegative physical source layers. The hard two-permutation assignment is the unchanged Thayer-SA pairwise requested-reconstruction + companion-reconstruction + scientific-cost assignment; identity wins exact ties. Ordinary rows assign both experts to target zero. Ambiguous rows retain identity/swap assignment and require the unordered own/alternate set.

Scientific feasibility uses the exact nondifferentiable frozen requested-source components: symmetric image distance / 0.25; each g/r/z symmetric relative-flux error / 0.20; each applicable g-r and r-z magnitude difference / 0.20 mag; and centroid displacement / mean PSF FWHM / 0.50. Authoritative acceptance remains componentwise <=1.0. Forward consistency, source-sum consistency, and prompt swap are evaluation-only and never feasibility or training objectives.

## Guaranteed homotopy projection P0

For every expert-target assignment, evaluate `X(alpha)=(1-alpha) candidate + alpha exact_truth` on exactly 1,025 evenly spaced alpha values from zero through one. Record every feasible interval, component entry alpha, the limiting component, and whether component ratios and binary feasibility are monotone. Locate the earliest feasible boundary by 40 deterministic bisection steps. Independently locate the earliest fixed training interior with every scientific ratio <=0.95 and move an additional alpha `1e-8` inward. The scientific evaluation threshold remains 1.0. If float32 conversion violates 0.95, use the exact-truth alpha=1 anchor. Correction distance is full-decomposition L2 correction divided by original-candidate L2 plus 1e-12.

## Fixed refinement P1

P1 starts only from P0 and uses one global CPU float32 projected augmented-Lagrangian method for every pairing: Adam, 80 iterations, learning rate 2e-4, zero weight decay, dual updates every 10 iterations, penalties 10/30/100/300 in four equal blocks, elementwise nonnegative projection after every update, and seven component constraints at the fixed 0.95 training interior. A final fixed 40-step bisection toward P0 restores any numerical violation. No neural parameter enters P0 or P1. No per-scene method selection is allowed. A separate SciPy solver is omitted because the full 21,600-variable per-pair problem is not computationally proportionate and P0 already supplies an exact feasible anchor.

## Projection gates and global selection

Exact truth must be feasible for all 256 scene/prompt/expert pairings. An eligible method must be finite, contract-valid, hard-assignment valid, canonically stable, deterministic, and unchanged-threshold feasible. The primary gate requires >=95% feasible pairings overall, >=90% ordinary sets, ambiguous own, alternate, and both-mode sets, and >=90% ordinary and ambiguous all-expert forward consistency. Selection is global and lexicographic: feasible rate, both-mode rate, smaller median correction, greater median interior slack, forward-consistency retention, deterministic stability, then P0 on an exact tie. Projection failure stops before neural work.

## Frozen projected targets

Ordinary target slots are two feasible representatives of the same approved truth region and may be identical. Ambiguous target slots remain an unordered two-target set in canonical target order. Persist tensors, canonical per-sample hashes, source-truth provenance outside inference inputs, assignment metadata, and projection metadata. No constraint or truth feature is appended to blend or prompt tensors.

## Unchanged Thayer-ME micro learning

If authorized, instantiate the exact shared Condition-C-prompted encoder and two independent 46,470-parameter expert decoders, total 165,612 parameters, with expert seeds 2026071201/2026071202, training seed 2026071250, exact Condition-C encoder warm start, and phase-2 trainability. Use MPS only, AdamW, batch size 8, learning rate 1e-3, zero weight decay, no augmentation, and at most 400 epochs. Each batch has the frozen four ordinary plus two complete ambiguity-pair schedule. The only loss is direct requested-source MSE plus direct companion-source MSE under the unchanged hard two-permutation matching; ordinary rows supervise both experts to the projected ordinary slots. No scientific surrogate, forward, source-sum, prompt-swap, concentration, diversity, uncertainty, or Atlas term is allowed.

Evaluate at epoch 1 and every 20 epochs. Stop on nonfinite loss/output, MPS fallback, target-hash mismatch, output-contract violation, nonfinite assignment, five consecutive epochs with expert gradient <=1e-12, or set prompt identity below 0.50 at/after epoch 100. Save only fresh campaign-local checkpoints. Success requires >=29/32 ordinary set-level coverage with both experts covered, median ordinary diameter <=1.0, >=29/32 ambiguous own/alternate/both-mode coverage, >=0.90 set prompt identity, and >=0.90 ordinary/ambiguous forward consistency. Partial success requires every coverage category materially nonzero but at least one success gate unmet. Failure includes feasible targets that the unchanged model cannot memorize or prompt/forward collapse. Gates cannot change after training.

## Interpretation

Success supports microset sufficiency of current decoder capacity and authorizes only one separately preregistered full non-Atlas feasibility-learning campaign. Projection pass with neural failure directly implicates capacity, encoder conditioning, or neural parameterization and authorizes a controlled decoder-capacity ladder without changing targets or thresholds. Projection failure prohibits a capacity ladder. Truth-coverage improvement with failed prompt/forward gates is partial and recommends one constrained consistency correction without reintroducing the dominating forward loss.

## Frozen rows

- `0` `pu_training_ordinary_00000` `ordinary` `` source index `0`
- `1` `pu_training_ordinary_00001` `ordinary` `` source index `1`
- `2` `pu_training_ordinary_00002` `ordinary` `` source index `2`
- `3` `pu_training_ordinary_00003` `ordinary` `` source index `3`
- `4` `pu_training_ordinary_00004` `ordinary` `` source index `4`
- `5` `pu_training_ordinary_00005` `ordinary` `` source index `5`
- `6` `pu_training_ordinary_00006` `ordinary` `` source index `6`
- `7` `pu_training_ordinary_00007` `ordinary` `` source index `7`
- `8` `pu_training_ordinary_00008` `ordinary` `` source index `8`
- `9` `pu_training_ordinary_00009` `ordinary` `` source index `9`
- `10` `pu_training_ordinary_00010` `ordinary` `` source index `10`
- `11` `pu_training_ordinary_00011` `ordinary` `` source index `11`
- `12` `pu_training_ordinary_00012` `ordinary` `` source index `12`
- `13` `pu_training_ordinary_00013` `ordinary` `` source index `13`
- `14` `pu_training_ordinary_00014` `ordinary` `` source index `14`
- `15` `pu_training_ordinary_00015` `ordinary` `` source index `15`
- `16` `pu_training_ordinary_00016` `ordinary` `` source index `16`
- `17` `pu_training_ordinary_00017` `ordinary` `` source index `17`
- `18` `pu_training_ordinary_00018` `ordinary` `` source index `18`
- `19` `pu_training_ordinary_00019` `ordinary` `` source index `19`
- `20` `pu_training_ordinary_00020` `ordinary` `` source index `20`
- `21` `pu_training_ordinary_00021` `ordinary` `` source index `21`
- `22` `pu_training_ordinary_00022` `ordinary` `` source index `22`
- `23` `pu_training_ordinary_00023` `ordinary` `` source index `23`
- `24` `pu_training_ordinary_00024` `ordinary` `` source index `24`
- `25` `pu_training_ordinary_00025` `ordinary` `` source index `25`
- `26` `pu_training_ordinary_00026` `ordinary` `` source index `26`
- `27` `pu_training_ordinary_00027` `ordinary` `` source index `27`
- `28` `pu_training_ordinary_00028` `ordinary` `` source index `28`
- `29` `pu_training_ordinary_00029` `ordinary` `` source index `29`
- `30` `pu_training_ordinary_00030` `ordinary` `` source index `30`
- `31` `pu_training_ordinary_00031` `ordinary` `` source index `31`
- `32` `pu_training_near_00000` `near_collision` `pu_training_pair_00001` source index `12000`
- `33` `pu_training_near_00001` `near_collision` `pu_training_pair_00001` source index `12001`
- `34` `pu_training_near_00002` `near_collision` `pu_training_pair_00002` source index `12002`
- `35` `pu_training_near_00003` `near_collision` `pu_training_pair_00002` source index `12003`
- `36` `pu_training_near_00004` `near_collision` `pu_training_pair_00003` source index `12004`
- `37` `pu_training_near_00005` `near_collision` `pu_training_pair_00003` source index `12005`
- `38` `pu_training_near_00006` `near_collision` `pu_training_pair_00004` source index `12006`
- `39` `pu_training_near_00007` `near_collision` `pu_training_pair_00004` source index `12007`
- `40` `pu_training_near_00008` `near_collision` `pu_training_pair_00005` source index `12008`
- `41` `pu_training_near_00009` `near_collision` `pu_training_pair_00005` source index `12009`
- `42` `pu_training_near_00010` `near_collision` `pu_training_pair_00006` source index `12010`
- `43` `pu_training_near_00011` `near_collision` `pu_training_pair_00006` source index `12011`
- `44` `pu_training_near_00012` `near_collision` `pu_training_pair_00007` source index `12012`
- `45` `pu_training_near_00013` `near_collision` `pu_training_pair_00007` source index `12013`
- `46` `pu_training_near_00014` `near_collision` `pu_training_pair_00008` source index `12014`
- `47` `pu_training_near_00015` `near_collision` `pu_training_pair_00008` source index `12015`
- `48` `pu_training_near_00016` `near_collision` `pu_training_pair_00009` source index `12016`
- `49` `pu_training_near_00017` `near_collision` `pu_training_pair_00009` source index `12017`
- `50` `pu_training_near_00018` `near_collision` `pu_training_pair_00010` source index `12018`
- `51` `pu_training_near_00019` `near_collision` `pu_training_pair_00010` source index `12019`
- `52` `pu_training_near_00020` `near_collision` `pu_training_pair_00011` source index `12020`
- `53` `pu_training_near_00021` `near_collision` `pu_training_pair_00011` source index `12021`
- `54` `pu_training_near_00022` `near_collision` `pu_training_pair_00012` source index `12022`
- `55` `pu_training_near_00023` `near_collision` `pu_training_pair_00012` source index `12023`
- `56` `pu_training_near_00024` `near_collision` `pu_training_pair_00013` source index `12024`
- `57` `pu_training_near_00025` `near_collision` `pu_training_pair_00013` source index `12025`
- `58` `pu_training_near_00026` `near_collision` `pu_training_pair_00014` source index `12026`
- `59` `pu_training_near_00027` `near_collision` `pu_training_pair_00014` source index `12027`
- `60` `pu_training_near_00028` `near_collision` `pu_training_pair_00015` source index `12028`
- `61` `pu_training_near_00029` `near_collision` `pu_training_pair_00015` source index `12029`
- `62` `pu_training_near_00030` `near_collision` `pu_training_pair_00016` source index `12030`
- `63` `pu_training_near_00031` `near_collision` `pu_training_pair_00016` source index `12031`
