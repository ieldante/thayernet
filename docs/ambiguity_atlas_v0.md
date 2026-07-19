# The Ambiguity Atlas v0

Atlas v0 is a fixed-observation-model stress benchmark for Thayer-Select. It
asks whether two materially different requested galaxies can participate in
source decompositions whose g/r/z observations are essentially
indistinguishable relative to the frozen BTK noise model.

The authoritative run is
`outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/`. Its preregistration
hash is `2b54bf035f5a51721b4d012faa84060bb926a81610463fcb393c16d5f3f39185`.
It uses the historical 0.86/0.81/0.77 arcsec g/r/z PSFs, 0.2 arcsec pixels,
unclipped source addition, and the exact BTK source-plus-sky Poisson contract.

Route 1 generated 30,000 approved training/search scenes and found 100
numerically valid near-collisions. The first 25 passed exact replay, artifact
checks, and both noiseless and noise-normalized observed-image review. Route 2
preserved 600 bounded catalog-parameter trials and produced 25/25 valid
optimization-feasibility pairs without optimizing source pixels.

The direct Atlas result passes: every frozen pair admits two constructed truth
decompositions inside the calibrated measurement tolerance, and every tested
checkpoint produced at least one unsafe requested reconstruction per pair.
However, the initial Atlas is strongly noise-dominated. It demonstrates finite
non-identifiability under this controlled observation model; it does not show
that high-information blends are generally ambiguous or estimate population
frequency.

The operational detector result fails. Same-cluster candidate diameter has
AUROC 0.4712 and zero recall at the frozen 4% control false-positive rate,
while the narrow R1 self-confidence comparison reaches AUROC 0.9176. No
Thayer-Audit model, catalog policy, development result, lockbox result, or
cross-deblender claim is authorized.

## Prompted-ResUNet follow-up

The prospective architecture-diversity run stopped before Atlas inference.
Its 199,219-parameter residual model achieved only 39.47% prompt-swap success
on the fresh Atlas-excluded validation manifest, below the frozen 80% gate.
Atlas artifacts and the authoritative 19/50, 0.4712, and zero-recall operational
results are unchanged. The failed pre-Atlas gate does not add a candidate family
and does not authorize Thayer-Audit.

## Thayer-PU stochastic follow-up

The separately preregistered Thayer-PU campaign passed every non-Atlas gate and
was evaluated once under a frozen K=32 prior protocol. Model-generated witnesses
rose from 19/50 to 24/50. Candidate-diameter AUROC rose from 0.4712 to 0.856 with
a pair-cluster bootstrap 95% interval of 0.751–0.942, and recall at the frozen
4% control false-positive rate rose from zero to 0.32.

This is partial success, not an Atlas witness pass. The 30/50 witness target
failed, and retained samples covered neither own nor paired alternate truth on
any Atlas observation. The direct constructed 50/50 ambiguity result remains
unchanged. No auditor or catalog policy is authorized.

## Thayer-PF pre-Atlas stop

The conditional flow-prior follow-up was not evaluated on Atlas. Its mandatory
non-Atlas posterior/decoder gate found 0% ordinary own-truth, near-collision
own-truth, and cross-decoded alternate-truth coverage. No flow was implemented
or fitted, and no one-time Atlas protocol was frozen. The authoritative Atlas
results remain the deterministic 19/50 baseline and Thayer-PU 24/50 result.

## Thayer-MH pre-Atlas stop

The K=2 ambiguity-set decoder passed promptability but achieved zero ordinary
own-truth, near-own, near-alternate, and both-mode coverage on protected
validation data. Its one-time Atlas gate remained closed. Atlas inference count
is zero, and the authoritative deterministic and Thayer-PU results are unchanged.

## Thayer-ME pre-training stop

The independent two-expert continuation failed its isolated training-only
micro-overfit capacity gate. Truth coverage remained zero despite prompt-faithful
and forward-consistent outputs. Full training and one-time Atlas evaluation were
prohibited. The authoritative Atlas results and frozen artifacts are unchanged.
