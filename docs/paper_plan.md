# Paper Plan

This is a placeholder planning document. The final paper may use a different structure after the experiments, figures, evaluation tables, and failure analysis are complete. Do not treat the section notes below as completed results.

## Abstract

To be completed after experiments.

## Introduction

Motivate astronomical deblending, controlled synthetic blends, and the research question.

## Prior Work

Summarize deblending context, simple segmentation baselines, and learned image-to-image reconstruction approaches. To be completed with proper references.

## Dataset

Describe Galaxy10 DECaLS, local data handling, preprocessing, and train/validation/test splitting.

## Synthetic Blending Method

Explain foreground extraction, halo-aware masking, shifts, brightness scaling, blur/noise, optional rotation, and difficulty metadata.

## Baselines

Define identity and threshold/connected-component baselines.

## Neural Model

Describe the compact U-Net architecture, objective function, and training setup.

## Evaluation Metrics

Describe MSE, MAE, PSNR, SSIM, and any optional IoU-style mask analysis.

## Whole-image vs Affected-region Analysis

Explain why whole-image metrics can favor identity-style outputs and how affected-region metrics isolate pixels changed by blending. To be completed after experiments.

## Results

To be completed after experiments.

## Failure Analysis

To be completed after qualitative inspection and difficulty-bin analysis.

## Limitations

Discuss limitations of synthetic blends, dataset scope, model capacity, and observational realism.

## Conclusion

To be completed after results are available.

## References

To be completed with dataset and methodology references.
