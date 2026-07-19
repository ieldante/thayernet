# Scientific-alignment objective

Thayer-SA tested a corrected training objective for the unchanged Thayer-ME
architecture on its frozen 64-scene microset. The objective contains only
requested-source reconstruction, companion-source reconstruction, a
threshold-normalized differentiable scientific distance, and ordinary expert
concentration. Ambiguous scenes retain the hard minimum over the two expert-to-
target permutations. Prompts, experts, and rows are averaged so none creates
an implicit multiplicity factor.

Forward reconstruction, source-sum consistency, prompt swap, and paired-scene
consistency are evaluation metrics only. They do not contribute training loss.
The source-recovery thresholds, target sets, architecture, and data boundary
were unchanged.

The surrogate and exact-truth stationary tests passed, but the preregistered
free-output optimizer did not reliably enter the frozen coverage region from
trained, collapsed, wrong-allocation, or random starts. The corrected objective
therefore remains an experimental loss geometry and is not authorized for
neural fitting or full-data training.

## Thayer-OC conditioning result

Thayer-OC kept this scalar objective, every weight, target, threshold, and the
hard assignment unchanged. Conditioning materially improved coverage for some
fixed starts, but no global method passed all 90% gates. Raw L-BFGS reached
0.844/0.875/0.750 ambiguous own/alternate/both-mode coverage from Thayer-ME
outputs while ordinary coverage remained 0.125. Adam-based T/D methods failed
exact-truth stationarity. The corrected objective remains unauthorized for
neural fitting.

## Feasibility-projection result

Thayer-FP did not reuse the scalar surrogate for training. Offline P0 targets
inside the unchanged region achieved complete projection coverage, but direct
requested/companion reconstruction with the unchanged architecture still left
all scientific coverage rates at zero. This supports a neural
capacity/conditioning/parameterization barrier after feasibility projection;
it does not validate the former scalar objective.
