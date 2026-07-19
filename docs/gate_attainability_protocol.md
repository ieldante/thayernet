# Prospective gate-attainability protocol

Every scientific gate must be audited before preregistration is hashed and
before any fitting or calibration-outcome access. The audit records the metric
range, baseline, requested threshold, derivation, attainability, and final
prospective threshold. Impossible rules must be corrected prospectively; they
must never be repaired after results are known.

Required range checks include:

- AUROC and coverage lie in `[0, 1]`.
- AUPRC is at most `1`, with random-ranking baseline equal to prevalence.
- Spearman correlation lies in `[-1, 1]`.
- interval width lies in `[0, infinity)`.
- a calibration sample of size `n` has order-statistic resolution `1/(n+1)`;
  the 90% upper-conformal rank is `min(n, ceil((n+1) * 0.90))`.

A prevalence-relative AUPRC gate must spend a fraction of the remaining
achievable gap:

```text
threshold = prevalence + alpha * (1 - prevalence),  0 <= alpha <= 1
```

It must not use `multiplier * prevalence`, because that can exceed one. In the
conditional-calibration campaign, prevalence `0.8165` and `alpha=0.75` gave an
attainable threshold of `0.954125`; the unchanged catastrophic head passed at
validation AUPRC `0.9971`.

The machine-readable audit belongs in the timestamped run. Tests must cover
the prevalence formula, metric ranges, conformal rank, order-statistic
resolution, and boundary cases. A failed attainability test is a stop
condition, not a reason to weaken a gate after fitting.

