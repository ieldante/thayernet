# Scientific-gradient preconditioning

Thayer-OC tested a frozen local preconditioner without changing the scalar
objective. At each T/D Adam step, the unchanged-objective gradient was
multiplied by

`clip(median_positive(|J|) / (|J| + 1e-8), 0.1, 10)`

where `J` is the local Jacobian of the threshold-normalized scientific
surrogate. Coverage outcomes never entered the preconditioner, and all 400
auxiliary Jacobian-gradient evaluations per initialization were recorded.

The preconditioned condition improved several compromise starts. From the
persisted Thayer-SA compromise it ended at 0.406 ordinary, 0.719 own, 0.719
alternate, and 0.562 both-mode coverage. From collapsed means it reached 0.281,
0.562, 0.562, and 0.375. It remained at zero minimum coverage from the
persisted Thayer-ME outputs.

The method failed its mandatory exact-truth control: final ordinary, own,
alternate, and both-mode coverage was 0.969, 0.906, 0.906, and 0.812. It is
therefore ineligible and does not authorize a gradient-preconditioned neural
micro-overfit.
