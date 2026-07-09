# Professor Briefing

## 30-Second Explanation

Thayer-Net is a controlled synthetic galaxy deblending project and model
family. I use Galaxy10 DECaLS cutouts to create foreground-only blends with
known clean targets, then test whether compact U-Nets can recover the target
better than identity or a simple threshold baseline. The best current model is
Thayer-BR v0.1: it learns contaminant light to subtract and is trained on a mix
of normal, core-overlap, and brightness/size stress blends.

## Naming Cheat Sheet

- Thayer-Net = project and model family.
- Thayer-Direct = direct reconstruction model.
- Thayer-Residual = residual prediction model.
- Thayer-BR v0.1 = current best balanced residual model.

## 2-Minute Explanation

The project started as a synthetic deblending benchmark, but the main work was
making the synthetic task defensible. Naively adding whole cutouts can create
rectangular artifacts and double backgrounds, so the pipeline now extracts
foreground contaminant light with halo-aware masks. The evaluation also focuses
on affected-region metrics because whole-image metrics hide failures when most
pixels are unchanged.

Thayer-Direct maps `blended -> target` and proves that learned reconstruction
beats identity and threshold baselines. A hard stress test then shows that
Thayer-Direct degrades under smaller shifts, brighter contaminants, similar-size
sources, and core overlap. Thayer-Residual improves robustness by learning
`blended - target` and subtracting that predicted contaminant layer. Thayer-BR
v0.1 trains the same residual formulation on a targeted mix of normal and hard
cases, improving both normal and stress aggregate affected-region MSE in the
current comparable evaluation.

The main lesson so far is that residual prediction helps, but targeted
hard-case training helps even more.

## Key Numbers

| Model | Normal affected MSE | Normal improvement | Stress affected MSE | Stress improvement | Worse-than-identity stress cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.068122 | 1.00x | 0.075541 | 1.00x | 0/1000 |
| Threshold | 0.073101 | 0.93x | 0.082746 | 0.91x | 990/1000 |
| Thayer-Direct | 0.004236 | 16.08x | 0.009390 | 8.04x | 13/1000 |
| Thayer-Residual | 0.004431 | 15.37x | 0.007069 | 10.69x | 0/1000 |
| Thayer-BR v0.1 | 0.002451 | 27.79x | 0.004587 | 16.47x | 0/1000 |

Earlier direct normal evaluation on an 800-blend held-out set gave `14.13x`
improvement. The table above is the current 1,000-blend same-run comparison.

## Architecture Explanation

- Thayer-Direct: input is the blended image; target is the clean target galaxy.
- Thayer-Residual: input is the blended image; target is
  `blended - target`; final reconstruction is `blended - predicted_residual`.
- Thayer-BR v0.1: same residual objective, but trained with 50% normal, 30%
  high-overlap/core-obstruction, and 20% brightness/size stress blends.

## Why the Result Makes Sense

Thayer-Residual helps in some cases because the model learns what light to
remove instead of redrawing the whole target. Thayer-BR v0.1 helps because hard
overlap and bright/similar-size contaminants become common during optimization
rather than rare accidents in random sampling.

## What I Am Not Claiming

- This is not full real-sky deblending.
- This is not proof of survey-scale robustness.
- This is not a claim that one model wins on every individual blend.
- The results are for a controlled synthetic benchmark and may change with more
  realistic sky backgrounds, PSFs, noise, or blend-generation settings.

## Questions for Professor

- Should I stop modeling and focus on the report?
- Should the next step improve blend realism and preprocessing diagnostics?
- Is the Thayer-BR v0.1 audit enough before writing it up?
- Is targeted hard-case training an acceptable final model comparison?
- Should the paper emphasize methodology, model comparison, or failure analysis?
