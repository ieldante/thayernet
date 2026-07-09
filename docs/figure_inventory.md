# Figure Inventory

This inventory separates reviewed public figures from raw generated output
figures. Raw `outputs/` directories remain ignored. Public figures copied into
`reports/figures/` should be reviewed before use in the README or paper.

## Main Paper Candidates

| Figure | Source path | Public destination | Intended use | Recommended caption |
| --- | --- | --- | --- | --- |
| Affected-region MSE bar chart | `outputs/runs/balanced_residual_20260708_184632/paper_figures/affected_region_mse_bar.png` | `reports/figures/balanced_affected_region_mse_bar.png` | Paper main, README optional | Affected-region MSE highlights reconstruction quality only where contaminant light changed the target image. |
| Normal vs stress improvement ratio | `outputs/runs/balanced_residual_20260708_184632/paper_figures/normal_vs_stress_improvement_ratio.png` | `reports/figures/balanced_normal_vs_stress_improvement_ratio.png` | README, paper main | Thayer-BR v0.1 improves affected-region MSE over identity on both normal held-out and hard stress tests. |
| Thayer-Direct vs Thayer-BR v0.1 scatter | `outputs/runs/balanced_residual_20260708_184632/paper_figures/direct_vs_balanced_scatter.png` | `reports/figures/balanced_direct_vs_balanced_scatter.png` | Paper main | Per-sample affected-region MSE comparison between Thayer-Direct and Thayer-BR v0.1. |
| Thayer-Residual vs Thayer-BR v0.1 scatter | `outputs/runs/balanced_residual_20260708_184632/paper_figures/old_residual_vs_balanced_scatter.png` | `reports/figures/balanced_old_residual_vs_balanced_scatter.png` | Paper main | Thayer-BR v0.1 improves many samples relative to Thayer-Residual, while some individual regressions remain. |
| Stress performance by core overlap | `outputs/runs/balanced_residual_20260708_184632/paper_figures/stress_performance_by_core_overlap_bin.png` | `reports/figures/balanced_stress_core_overlap.png` | Paper main or analysis section | Stress-test performance grouped by target-core overlap bin. |
| Thayer-BR v0.1 improvement example | `outputs/runs/balanced_residual_20260708_184632/paper_figures/balanced_residual_improves_over_old_residual.png` | `reports/figures/balanced_residual_improvement_example.png` | Paper main qualitative figure, README optional | Qualitative example where Thayer-BR v0.1 reduces contaminant error relative to Thayer-Residual. |

## Appendix and Limitations Candidates

| Figure | Source path | Public destination | Intended use | Recommended caption |
| --- | --- | --- | --- | --- |
| Thayer-BR v0.1 failure case | `outputs/runs/balanced_residual_20260708_184632/paper_figures/balanced_residual_failure_example.png` | `reports/figures/balanced_residual_failure_example.png` | Appendix, limitations | Example where Thayer-BR v0.1 still leaves error in an ambiguous overlap region. |
| Thayer-Residual or Thayer-Direct beats Thayer-BR v0.1 | `outputs/runs/balanced_residual_20260708_184632/paper_figures/old_or_direct_beats_balanced_residual.png` | `reports/figures/balanced_residual_counterexample.png` | Appendix, limitations | Example showing that Thayer-BR v0.1 is not uniformly best on every individual sample. |
| Thayer-BR v0.1 to Thayer-Residual error ratio histogram | `outputs/runs/balanced_residual_20260708_184632/paper_figures/hist_balanced_to_old_residual_ratio.png` | Not yet copied | Appendix | Distribution of per-sample affected-region MSE ratios for Thayer-BR v0.1 versus Thayer-Residual. |
| Worse-than-identity counts | `outputs/runs/balanced_residual_20260708_184632/paper_figures/worse_than_identity_counts.png` | Not yet copied | Appendix or results | Number of samples where each method has higher affected-region MSE than identity. |
| Stress performance by blend severity | `outputs/runs/balanced_residual_20260708_184632/paper_figures/stress_performance_by_blend_severity_bin.png` | Not yet copied | Appendix | Stress-test performance grouped by measured blend severity bin. |

## Legacy Public Figures

| Figure | Path | Intended use | Note |
| --- | --- | --- | --- |
| Earlier affected-region MSE bar chart | `reports/figures/affected_region_mse_bar.png` | Legacy comparison only | Earlier Thayer-Direct/Thayer-Residual figure; do not mix with current Thayer-BR v0.1 table without caption caveat. |
| Earlier normal vs stress improvement chart | `reports/figures/normal_vs_stress_improvement_ratio.png` | Legacy comparison only | Earlier Thayer-Direct/Thayer-Residual figure. |
| Residual success over direct | `reports/figures/residual_success_over_direct.png` | Legacy qualitative example | Useful for explaining Thayer-Residual before Thayer-BR v0.1. |
| Thayer-Direct success | `reports/figures/direct_unet_success.png` | Appendix or history | Legacy Thayer-Direct example. |
| Thayer-Direct partial failure | `reports/figures/direct_unet_partial_failure.png` | Appendix or limitations | Legacy Thayer-Direct failure example. |

## README Recommendation

Use one or two figures maximum:

- `reports/figures/balanced_normal_vs_stress_improvement_ratio.png`
- Optional: `reports/figures/balanced_residual_improvement_example.png`

Avoid leading with the most confusing failure case. Failure and counterexample
figures belong in the limitations section or appendix.
