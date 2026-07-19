# Preregistration: prompted ResUNet candidate diversity

Frozen before the corrected campaign model implementation, scene rendering, or
model fitting. An untrained rejected scaffold from the stopped predecessor run
is present in the source tree; it has incompatible dimensions, produced no
checkpoint or scientific output, and is not this campaign's model.

## Scientific hypothesis

A compact residual-block encoder-decoder, initialized and trained independently under the exact Condition-C source-layer contract, will produce candidate differences that materially exceed same-family seed differences while retaining promptability and forward consistency. Useful diversity must add valid model-candidate ambiguity witnesses and improve frozen candidate-diameter detection; raw disagreement caused by error or output artifacts is not success.

## Frozen data and exclusions

- Catalog and source split hashes are frozen in `logs/input_provenance.json`.
- All 59 source groups appearing in the 25 frozen Atlas pairs or their targeted feasibility pairs are excluded from both ResUNet training and validation.
- Fresh manifests contain 10,000 training and 1,500 validation two-source scenes with group-disjoint approved training/validation partitions.
- Requested source is seeded uniform A/B. Positions are symmetric about a seeded midpoint, separation is uniform on [0.8, 3.2] arcsec, and one explicit BTK `add_noise=all` realization is used.
- Historical development and final lockbox scenes remain inaccessible. Access counts must stay 0/0.

## Exact architecture

Input is a four-channel 60x60 tensor: normalized g/r/z blend plus a unit-peak Gaussian coordinate prompt with sigma 2 pixels. Output is three unconstrained linear channels representing the requested noiseless g/r/z source on zero background.

Every encoder/decoder transform is a predeclared residual block: 3x3 convolution, GroupNorm, SiLU, 3x3 convolution, GroupNorm, residual/1x1 projection, then SiLU. Convolutions followed by normalization have no bias. Downsampling uses stride-2 residual blocks; upsampling is bilinear followed by skip concatenation and a residual fusion block.

```text
4x60x60 -> RB(4,16) -> RBs2(16,32) -> RBs2(32,64)
         -> RB(64,64)
         -> up+skip32 -> RB(96,32)
         -> up+skip16 -> RB(48,16) -> 1x1 linear head -> 3x60x60
```

Exact expected trainable parameter count is 199,219, below the frozen 350,000 preferred ceiling and 500,000 absolute ceiling. Condition C has 119,091; the expected ratio is 1.672830. No Condition-C blocks, encoder/decoder weights, or historical checkpoints may be loaded. Initialization is fresh Kaiming-normal for convolutions, zero convolution bias, GroupNorm scale one/bias zero.

## Training

- Loss: whole-image normalized MSE, identical in meaning to the promptability baseline.
- Optimizer: Adam, learning rate 0.001; no weight decay.
- Scheduler: cosine annealing over exactly 20 epochs.
- Batch size: 8. A smaller frozen value is permitted only after a documented MPS out-of-memory event before any optimizer step; otherwise no change.
- Seed: 2026077301 for initialization, minibatch order, and Torch/NumPy/Python.
- Checkpoint selection: minimum validation MSE only; first epoch wins exact ties. Best and final are stored separately.
- Stop on non-finite values/gradients, MPS fallback, manifest/replay mismatch, checkpoint collision, Atlas exposure, or instability defined as validation loss above 10 times epoch-1 loss for two consecutive epochs.

## Pre-Atlas validation gate

Condition C and ResUNet are evaluated on the same fresh non-Atlas validation scenes. ResUNet must satisfy all:

1. finite predictions and finite stable reconstruction metrics;
2. prompt-swap success at least 0.80 (both A/B queries closer to their requested truth than the alternate truth);
3. output-collapse rate at most 0.10 (swapped-output distance below 10% of truth distance);
4. mean whole-image MSE at most 3.0 times Condition C on identical scenes;
5. at least 0.75 of individual queries closer to requested than alternate truth;
6. no source-identity inversion signal: prompt-swap failure at most 0.20 and median signed requested-versus-alternate MSE advantage below zero.

Report whole/source-region MSE, MAE, PSNR, SSIM, per-band flux error, centroid error, prompt perturbation sensitivity, collapse, and confusion. Failure stops before any ResUNet Atlas inference.

## Candidate contract and leakage gate

Both families use identical dimensions, g/r/z order, frozen inverse normalization, electron-per-pixel units, no clipping, zero residual background, prompt alignment, and two-query full decomposition. Candidate hashes and recomposition are recorded. Trivial family leakage is tested with dynamic range, border mean/variance, clipping frequency, zero fraction, total-flux scale, and edge/interior ratios. Any deterministic contract defect is corrected before Atlas evaluation; scientific outputs are never rescaled to increase agreement. A family-ID classifier is not trained.

## One-pass Atlas evaluation

After implementation, training, checkpoint selection, promptability, contract alignment, and threshold/hash revalidation, the selected ResUNet is evaluated exactly once on the 50 frozen noisy Atlas observations and the 25 frozen matched controls. Atlas labels/results may not affect model selection or fitting. One candidate per requested source and a two-query decomposition are saved with hashes, runtime, finite audit, and frozen forward-consistency scores.

## Frozen analyses and gates

Scientific distance uses the existing frozen image (0.25), per-band flux (0.20), color (0.20 mag), and centroid (0.5 mean-PSF-FWHM) limits; valid size/shape distances are descriptive. Same-family reference is the scene-aligned Condition-C/R0/R1 pairwise output distance already in the authoritative Atlas. Bootstrap intervals use 2,000 deterministic resamples clustered by Atlas pair/source groups; 95% percentile intervals are reported.

Architecture-diversity PASS requires compatibility/promptability, median ResUNet-versus-Condition-C primary distance at least 1.25 times the median same-family distance, the 95% cluster-bootstrap lower bound on that median ratio above 1.0, and no trivial contract leakage or catastrophic reconstruction degradation.

Witness-improvement PASS requires at least 25/50 model-candidate witnesses versus 19/50, at least six paired net additions, one-sided exact paired sign probability at most 0.05, a 95% paired cluster-bootstrap improvement interval with lower endpoint above zero, forward consistency for every added witness, and bounded controls.

Diameter PASS requires candidate-diameter AUROC at least 0.60 with its 95% cluster-bootstrap interval lower endpoint above 0.5, recall at the frozen 4% control-FPR threshold at least 0.10 and nonzero, observed control FPR at that frozen threshold at most 0.08, and no family artifact explanation. The historical reference is AUROC 0.4712 and recall 0.

Overall SUCCESS requires promptability, architecture diversity, witness improvement, diameter, zero leakage, and 0/0 development/lockbox access. PARTIAL SUCCESS means genuine diversity or witness gain without diameter PASS. FAILURE means same-cluster behavior, error/artifact-driven diversity, no witness improvement, or non-informative diameter. No threshold changes follow Atlas results.

## Authorized interpretation

SUCCESS authorizes one third genuinely distinct family, not an auditor. PARTIAL SUCCESS preserves ResUNet and recommends one third classical constrained or fundamentally different family. FAILURE recommends explicit multi-hypothesis generation or posterior sampling, not another deterministic U-Net variant. No result establishes model-agnostic transfer.
