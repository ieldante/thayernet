# Atlas candidate diversity

Ambiguity Atlas v0 remains the authoritative direct ambiguity result: 25 frozen
pairs, constructed witnesses on 50/50 observations, model-candidate witnesses
on 19/50, candidate-diameter AUROC 0.4712, and zero recall at the frozen 4%
control false-positive rate.

The prompted-ResUNet campaign was designed to add one architecture family but
stopped before Atlas inference because its non-Atlas promptability gate failed.
Consequently, no Atlas candidate diameter, witness count, AUROC, operating-point
recall, cross-family distance, or source-group bootstrap was recomputed. The
authoritative Atlas values are unchanged rather than treated as negative
ResUNet results.

Absence of a new candidate remains non-probative. One failed deterministic
residual architecture does not establish model-agnostic behavior or justify a
black-box auditor.

