# Model card: Thayer-Select

Thayer-Select is a compact U-Net family for selecting and reconstructing a
requested isolated galaxy from a controlled two-source BTK blend using three
normalized `g,r,z` channels and one Gaussian coordinate prompt. It is a
research model, not a survey production deblender.

Phase-I Condition C has 119,091 parameters and a reconstruction head only.
Phase-II R0 keeps that reconstruction-only architecture but trains on valid,
perturbed-valid, null, and ambiguous queries. Phase-II R1 shares the same
backbone and adds a finite bounded pixel log-variance map, a global empirical
contract-success probability, and a no-source probability. Morphology, source
identity, split labels, simulator difficulty, clean targets, evaluation errors,
and masks never enter the forward pass.

The completed R0 has 119,091 parameters. R1 has 123,368 parameters (+4,277),
bounded log variance `[-8, 2]`, a global actionable-success head, and a
no-source head. R1 completed 20 epochs on MPS after fail-closed loss-design
corrections. Its selected calibrator is isotonic and its primary development
contract is PERMISSIVE because every predeclared actionable contract was highly
imbalanced on training/validation labels.

Intended use is controlled investigation of promptability, recoverability, and
selective abstention under frozen BTK manifests. Known limitations include
heavy-tailed reconstruction errors, simulator dependence, one primary training
seed unless replication completes, uncertain transfer to real data, and the
fact that a calibrated global score does not automatically calibrate each pixel
uncertainty. Null and ambiguous queries require abstention-aware evaluation.
The current R1 is not a successful safety model: ambiguous queries rank above
clear valid queries on average, catastrophic failure does not fall at 80%
coverage, and null hallucination is not better than R0 or frozen Phase-I C on
identical new null-coordinate scenes. Full uncertainty maps were not persisted
for the one-time development pass. The lockbox remains unavailable for model
selection, debugging, calibration, threshold tuning, figures, or qualitative
inspection.
