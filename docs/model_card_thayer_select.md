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

## Hierarchical safety-policy addendum

The reconstruction model remains Phase-I Condition C at SHA-256
`e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`.
The hierarchical campaign did not add parameters to, fine-tune, or alter that
model. It adds external lightweight CPU heads over frozen model-accessible
features:

- a five-seed F_COMBINED small-MLP ensemble for UNIQUE_VALID/NULL/AMBIGUOUS;
- separate five-seed image-, flux-, and centroid-risk median/q=0.90 heads;
- a separate five-seed confusion-risk ensemble;
- vector scaling, temperature scaling, and split-conformal upper residuals fit
  on natural calibration only.

These heads are a research diagnostic, not a deployable release. The query gate
worked, but the complete moderate-limit policy accepted only 0.05% of valid
development scenes. The full-policy zero false acceptance for invalid queries
therefore reflects abstention collapse, not useful safety. Raw risk-head outputs
also showed log-to-linear tail instability, although log-space rankings and
conformal marginal coverage were strong. Do not expose a reconstruction for a
NULL or AMBIGUOUS query; do not treat marginal conformal coverage as
class-conditional or deployment coverage; do not use generator variables as
head inputs.

The fresh development manifest was evaluated once after freeze. No lockbox
evaluation exists. Current classification is **FAILURE** for the complete
hierarchical policy.

## Protocol-status correction

The 2026-07-11 hierarchical result remains valid historical evidence but is not
certified as a fully preregistered sequence: the required preregistration and
full original-composite postmortem were absent before head fitting and the
one-time development evaluation. A later append-only audit reproduced every
persisted Phase-II moderate label exactly and made no model or policy change.
Do not interpret that retrospective audit as preregistration, and do not
authorize lockbox evaluation from the historical result.

## Prospective feasibility addendum

The 2026-07-12 feasibility campaign did not change Thayer-Select or produce a
new deployable model. It reused byte-identical Phase-I Condition C for all
training, validation, and calibration outcomes and fitted external CPU-only
query/risk heads over frozen model-accessible features.

The query gate and all risk rankers showed strong prospective signal, but the
campaign is only PARTIAL SUCCESS: its catastrophic AUPRC gate was unattainable
at the observed prevalence, and image/flux calibration was only marginally—not
conditionally—reliable. No operational threshold or combined policy exists.
There was zero development and lockbox access. Intended use remains controlled
promptable selective galaxy deblending research; survey deployment, final
selective-risk claims, and lockbox evaluation remain prohibited.
