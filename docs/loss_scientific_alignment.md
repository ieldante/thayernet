# Loss and scientific alignment

Thayer-LG separates pixel-objective improvement from scientific source
recovery. Across the 416 canonical scene/configuration evaluations, total
objective and primary scientific distance were negatively rank-correlated
(Spearman -0.157). Requested reconstruction, companion reconstruction, and
target source-sum terms each had Spearman correlations above 0.80 with primary
scientific distance. The optimized forward term had Spearman -0.350: lower
forward loss generally accompanied worse source truth.

At exact truth, the source-target gradients are zero but the normalized
forward-to-observed gradient is nonzero and dominant. Set-matching and forward
gradients conflict on 63.3% of ordinary evaluations and 51.6% of ambiguous
evaluations; severe cosine conflict occurs on 31.25% and 25.0%, respectively.
The differentiable scientific surrogate is nearly orthogonal to the full
objective on average.

Moving only 5% from exact truth toward the trained output lowers the mean
objective from 0.029377 to 0.029047 while the combined frozen coverage rate
falls from 1.0 to 0.094. The path objective reaches its minimum near alpha 0.5,
where coverage is zero. Forward consistency remains high, confirming that it
is not source-identification correctness.

Among detached diagnostic objectives, source reconstruction plus ordinary
concentration produced the lowest final mean scientific distance after the
frozen 40-step protocol. This is an ablation result, not authorization to
change the historical objective or its weights.
