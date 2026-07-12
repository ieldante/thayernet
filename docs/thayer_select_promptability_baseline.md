# Thayer-Select Phase I promptability baseline

Status: complete and frozen controlled-BTK development evidence. The completed
run is `outputs/runs/thayer_select_prompt_ablation_20260711_164329`; its outputs
and checkpoints must not be edited or used as a tuning surface.

Condition A used centered requested sources and no prompt. Condition B used
the same compact three-channel reconstruction architecture with randomized,
exchangeable source positions and no prompt. Condition C used the same
randomized scenes as B and appended one Gaussian coordinate-prompt channel.
A/B each had 118,947 parameters and C had 119,091: prompting added exactly 144
first-layer parameters. Each condition used 8,000 training scenes, 1,000
validation scenes, 20 epochs, batch size 8, one fixed training seed, Adam, a
cosine scheduler, and MPS. The source split was persistent-source and
duplicate-group disjoint. Calibration and development sources were separate;
the sealed lockbox was assigned but no lockbox scene was generated or opened.

On the one-time 1,000-scene development evaluation, macro mean requested-source
MSE was `1.827208e6` for A, `2.029314e6` for B, and `1.020042e6` for C. The
paired C-minus-B mean was `-1.009271e6` with bootstrap 95% interval
`[-1.976624e6, -2.929767e5]`. C won 720/1,000 whole-image comparisons but
499/1,000 source-region comparisons; these are distinct statements. Macro and
micro source-region MSE also differ (`1.020042e6` macro versus `1.623287e6`
micro for C) because masks and heavy-tailed scene errors have heterogeneous
sizes. The result does not mean every scene improved.

Prompt swap required both source-A and source-B queries on one identical blend
to reconstruct closer to their requested isolated truth than to the alternate
truth. Condition C passed on 98.0% of scenes. Output collapse meant the
prompt-swapped output difference was below 10% of the isolated-truth
difference; it occurred on 0.2%. Coordinate prompting substantially reduced
mean requested-source error and produced 98% prompt-swap success, although
source-region errors remained heterogeneous and heavy-tailed.

The no-harm tests exposed the Phase II motivation. Exact empty prompts met the
declared hallucination criterion in 100% of cases. Prompts on the alternate
real source selected that source and therefore appeared as 99% confusion only
relative to the formerly chosen target; Phase II corrects the semantics by
treating that coordinate as a valid request for the alternate galaxy. Phase I
shows promptability, not no-source understanding, calibrated uncertainty,
final-paper performance, or real-sky validity.
