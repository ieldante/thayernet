# Posterior/decoder sufficiency preregistration

Frozen before any Part-D neural inference. The test uses only the authoritative
Thayer-PU best checkpoint and its non-Atlas validation arrays. It draws K=32
posterior samples with seed 2026078401. The first 256 ordinary validation scenes
and every one of the 250 frozen validation near-collision pairs (500 scenes) are
selected by manifest order. This fixed sample is large enough to expose a route
failure while avoiding any Atlas or unauthorized development access.

For each scene and both coordinate prompts, q(z|x,y_A,y_B) is sampled in canonical
source order. Own decode uses the scene's own posterior latent under its own
condition. For each frozen near-collision pair, left posterior latents are also
decoded under the right observed condition and right posterior latents under the
left observed condition. The destination scene's corresponding prompt is used;
the teacher latent is the only transferred quantity. Full decompositions are
inverse-normalized before all scientific and forward metrics.

Coverage means at least one forward-consistent requested-source sample within
the unchanged primary scientific distance <=1.0. Source identity means requested
MSE is strictly smaller to the named truth than to the other source in the same
destination scene. Forward consistency uses the already-frozen Thayer-PU
calibration thresholds without recalibration.

Prospective hard gates:

1. ordinary posterior own-truth coverage >=0.70;
2. near-collision own-posterior own-truth coverage >=0.70;
3. near-collision cross-decode alternate-truth coverage >=0.30;
4. ordinary, near-own, and near-cross forward-consistent sample fractions each >=0.50;
5. ordinary and near-own requested-source identity each >=0.70;
6. cross-decode alternate identity >=0.70;
7. every metric is finite, all 250 pairs are evaluated, and all four source
   groups in each pair remain disjoint.

These rates lie in [0,1], have nonempty attainable pass regions, and do not
depend on realized Part-D outcomes. Gate 3 matches the frozen Atlas alternate
coverage floor; Gates 1-2 match the frozen Atlas own-coverage floor; the
forward and identity floors reuse existing campaign semantics. If any gate
fails, the campaign stops before flow implementation or fitting and recommends
exactly one ambiguity-set decoder-training experiment. Posterior samples remain
diagnostics only and never become deployable hypotheses.
