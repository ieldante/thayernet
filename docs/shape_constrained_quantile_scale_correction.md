# Shape-constrained quantile scale correction

## Scope and outcome

The prospective run
`outputs/runs/thayer_select_shape_constrained_quantile_20260712_033406/`
completed with **FAILURE**. It used only the four frozen deployable proxies,
persisted source-group-held-out training predictions, fixed risk heads, and the
existing train/validation/natural-calibration partitions. Condition C was not
executed or modified. Development and lockbox remained sealed.

The training-only audit rejected global monotonicity. Image q=0.90 absolute
residuals changed from `9.483` to `1.476` across the lowest and highest z0
deciles and from `7.206` to `1.455` across z1. Flux showed corresponding
endpoint reversals `16.811` to `11.403` and `15.070` to `10.193`.

## Frozen selection and result

Q1 used four convex centered hinge effects. Q2 added exactly one nonnegative
upper-half product,
`softplus(gamma) * relu(z0 - 0.50) * relu(z1 - 0.50)`. Every fitted main
effect was convex and nondecreasing above its anchor, and every interaction
coefficient was nonnegative.

Q2 did not improve worst supported validation-cell coverage over Q1 for either
risk and failed the preregistered `0.03` improvement requirement. Validation
selected Q1 for both risks without calibration access. Q1 worst supported
validation-cell coverage was `0.828` image and `0.673` flux.

After normalized conformal correction, selected image/flux marginal coverage
was `0.9221`/`0.9221`, and worst supported subgroup coverage was only
`0.5440`/`0.5907`. Low-SNR/high-obstruction coverage was identical to those
worst values. The connected-source bootstrap 95% lower bounds were
`0.4730`/`0.5222`. Both components fail; centroid remains PASS.

An earlier restart at `..._032938` was superseded before handoff because its
main-effect centering constant was cached before optimization. The corrected
run centers the learned hinge basis dynamically from frozen training means;
post-fit centering errors are below `2.31e-7` and are now integrity-tested.

No full policy campaign is authorized. The only recommended follow-up is one
separately preregistered train/validation/calibration-only convex
tensor-product quantile experiment over z0 and z1, retaining the same four
proxies, OOF targets, gates, and sealed development/lockbox partitions.
