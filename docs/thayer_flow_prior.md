# Thayer-PF conditional flow-prior campaign

Thayer-PF was the planned Thayer Prior Flow correction to the frozen Thayer-PU
conditional Gaussian prior. The campaign stopped at its mandatory
posterior/decoder sufficiency gate. No flow was implemented, fitted, selected,
or evaluated.

The frozen truth-coverage metric passed an independent synthetic-array audit,
and every persisted Thayer-PU baseline reproduced without new Atlas inference.
The K=32 posterior diagnostic then evaluated 256 ordinary validation scenes and
all 250 non-Atlas validation near-collision pairs. Posterior own-truth coverage
was 0% for both ordinary and near-collision scenes. Cross-decoded alternate-
truth coverage was also 0%. Median best scientific distances were 6.37, 7.83,
and 7.83 frozen thresholds for ordinary, near-own, and near-cross evaluation.

The decompositions were usually forward-consistent—0.930, 1.000, and 1.000
sample fractions—and own posterior source identity remained high. This does not
rescue truth coverage. It shows that the frozen decoder can emit observationally
consistent, prompt-associated decompositions while remaining far from the
known source truth under the unchanged scientific metric. Cross-decoded latents
also selected the alternate truth only 1.76% of the time.

The data do not support the claim that the diagonal Gaussian prior is the
primary bottleneck. Normalizing-flow galaxy priors are established techniques,
but no flow-prior result exists here. Posterior samples are diagnostics only,
no witness proves uniqueness, and absence of a witness remains non-probative.
The frozen Atlas was not inferred on again, and the final lockbox remains
untouched.

The single next experiment is an ambiguity-set decoder-training campaign: for
each approved non-Atlas near-collision pair, train one decoder to represent both
decompositions under each observationally equivalent condition while preserving
prompt identity and forward consistency.

## Ambiguity-set continuation result

The prescribed Thayer-MH follow-up constructed all prospective target sets and
trained a compact shared K=2 decoder. Promptability passed, but own and alternate
truth coverage remained zero on protected non-Atlas data. Explicit set-valued
supervision alone did not make the approved modes operationally reachable. No
flow or Atlas continuation is authorized.
