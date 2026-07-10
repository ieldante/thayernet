# Advisor Update: Correctness Audit and Grouped v0.2 Retrain

## Bottom line

Thayer-BR v0.2 Moderate remains the strongest model family. After rebuilding
the benchmark with exact-pixel and exact-coordinate group-disjoint sources, a
fresh grouped retrain still substantially outperforms identity. The corrected
normal development result is **28.81x lower affected-region MSE than identity**
(about 5.37x lower RMSE), not the historical 32.3x headline.

This is a grouped development result, not a final-paper estimate. No additional
training is planned until a genuinely untouched final source partition exists.

## What the audit found

- The historical row-index split contained 29 pixel-identical and 27
  exact-coordinate cross-split pairs, implicating 57/17,736 sources (`0.321%`).
- Removing implicated historical evaluation rows changed the reported ratios by
  at most about `0.31%`. The measured aggregate effect was minor, but the split
  protocol was still invalid.
- The new grouped source split has zero cross-split source, group, exact-pixel,
  or exact-coordinate overlap. All 13,000 grouped blend rows replay exactly.
- The old checkpoint remains strong on grouped manifests, but that comparison
  is diagnostic only: `54.575%` of rows expose an old training or validation
  source group.

## Grouped retrain result

The v0.2 Moderate grouped retrain used MPS, one training seed, 8,000 training
blends, 1,000 validation blends, and the same affected/core-weighted residual
formulation.

| Grouped development suite | Affected MSE | Lower MSE vs identity |
| --- | ---: | ---: |
| Normal | 0.00231890 | 28.8127x |
| Hard stress | 0.00458983 | 15.8025x |
| Compact bright | 0.00872771 | 9.18304x |
| High core obstruction | 0.00491680 | 15.8378x |

The grouped result is lower than the historical `32.3x` normal and `19.6x`
hard result. That difference cannot be attributed solely to leakage because the
historical run used 12,000 training blends while the grouped retrain used 8,000,
and only one grouped training seed has been run.

## Model and claim status

- **Original development split:** preserve `32.3x` normal and `19.6x` hard as
  historical development evidence only.
- **Historical checkpoint on grouped manifests:** diagnostic, not
  source-independent, because of historical source exposure.
- **Grouped v0.2 retrain:** current source-group-disjoint development reference.
- **Thayer-BR v0.3 Delta:** preservation/color tradeoff ablation, not current
  best.
- **Thayer-ResUNet v0.4:** architecture ablation, not current best.
- **Future final-paper result:** not available yet.

The earlier provisional final pool is also superseded: 590/1,000 of its sources
were later used by grouped training or validation. A defensible final result now
requires either independent data or a new four-way grouped
train/validation/development/final split established before retraining. The
final partition must remain untouched until the model and analysis protocol are
frozen.

