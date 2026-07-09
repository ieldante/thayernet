# Figure Inventory

This inventory separates reviewed public figures from raw generated output
figures. Raw `outputs/` directories remain ignored. Public figures copied into
`reports/figures/` should be reviewed before use in the README or paper.

## README Figures

| Figure | Path | Intended use | Recommended caption |
| --- | --- | --- | --- |
| v0.2 normal vs stress improvement | `reports/figures/v02_improvement_ratio.png` | README main figure | Thayer-BR v0.2 Moderate improves affected-region MSE over identity on both normal held-out and hard stress-test blends. |
| v0.2 qualitative improvement | `reports/figures/v02_weighted_improvement_example.png` | README optional | Example where affected/core-weighted residual loss improves reconstruction relative to Thayer-BR v0.1. |

Use one figure near the top of the README. Keep counterexamples for limitations
or appendix sections.

## Main Paper Figures

| Figure | Source path | Public destination | Intended use | Recommended caption |
| --- | --- | --- | --- | --- |
| v0.2 affected-region MSE bar chart | `outputs/runs/weighted_residual_20260709_030245/paper_figures/affected_region_mse_bar.png` | `reports/figures/v02_affected_mse_bar.png` | Main results | Affected-region MSE for identity, threshold, direct, residual, Thayer-BR v0.1, and Thayer-BR v0.2 Moderate. |
| v0.2 normal vs stress improvement ratio | `outputs/runs/weighted_residual_20260709_030245/paper_figures/normal_vs_stress_improvement_ratio.png` | `reports/figures/v02_improvement_ratio.png` | Main results, README | Improvement ratio versus identity on normal held-out and hard stress-test blends. |
| v0.2 core affected MSE | `outputs/runs/weighted_residual_20260709_030245/paper_figures/core_affected_mse_comparison.png` | `reports/figures/v02_core_mse.png` | Core-overlap results | Core affected MSE comparison showing v0.2 Moderate's improvement over v0.1. |
| v0.2 versus v0.1 scatter | `outputs/runs/weighted_residual_20260709_030245/paper_figures/weighted_vs_br_v01_per_sample_scatter.png` | `reports/figures/v02_weighted_vs_v01_scatter.png` | Per-sample model comparison | Per-sample affected-region MSE for Thayer-BR v0.2 Moderate versus Thayer-BR v0.1. |
| v0.2 multi-seed summary | `outputs/runs/weighted_residual_20260709_030245/tables/multiseed_summary.csv` | `reports/figures/v02_multiseed_summary.png` | Robustness results | Multi-seed improvement ratio for Thayer-BR v0.2 Moderate on normal and stress evaluations. |
| v0.2 qualitative success | `outputs/runs/weighted_residual_20260709_030245/paper_figures/qualitative_weighted_improves_over_br_v01.png` | `reports/figures/v02_weighted_improvement_example.png` | Qualitative results | Example where Thayer-BR v0.2 Moderate improves over Thayer-BR v0.1. |
| v0.2 counterexample | `outputs/runs/weighted_residual_20260709_030245/paper_figures/qualitative_br_v01_beats_weighted.png` | `reports/figures/v02_counterexample.png` | Limitations or appendix | Example where Thayer-BR v0.1 beats Thayer-BR v0.2 Moderate on an individual sample. |

## Appendix Figures

| Figure | Source path | Intended use | Recommended caption |
| --- | --- | --- | --- |
| v0.2 non-core affected MSE | `outputs/runs/weighted_residual_20260709_030245/paper_figures/noncore_affected_mse_comparison.png` | Appendix | Non-core affected MSE comparison for weighted residual evaluation. |
| v0.2 weighted/BR v0.1 ratio histogram | `outputs/runs/weighted_residual_20260709_030245/paper_figures/hist_weighted_to_br_v01_affected_mse_ratio.png` | Appendix | Distribution of per-sample affected MSE ratios for v0.2 Moderate relative to v0.1. |
| Worse-than-identity counts | `outputs/runs/weighted_residual_20260709_030245/paper_figures/worse_than_identity_count_chart.png` | Appendix or results | Number of samples where each method performs worse than identity. |
| Stress performance by core overlap | `outputs/runs/weighted_residual_20260709_030245/paper_figures/stress_performance_by_core_overlap_bin.png` | Appendix or core-overlap section | Stress-test performance grouped by target-core overlap bin. |
| Stress performance by blend severity | `outputs/runs/weighted_residual_20260709_030245/paper_figures/stress_performance_by_blend_severity_bin.png` | Appendix | Stress-test performance grouped by measured blend severity bin. |
| Weighted variant comparison | `outputs/runs/weighted_residual_20260709_030245/paper_figures/weighted_variant_comparison.png` | Ablation appendix | Moderate and Strong weighting comparison; Strong is not the current best model. |
| Visual-vs-metric disagreement grids | `outputs/runs/size_visual_audit_20260709_102251/example_grids/visual_metric_disagreements/` | Appendix limitations | Selected examples where visual judgment and affected-region MSE ranking may disagree. |
| Apparent-size audit figures | `outputs/runs/size_visual_audit_20260709_102251/figures/` | Appendix robustness audit | Apparent-size, centrality, core-obstruction, and halo-band diagnostics. |

## Historical Figures

| Figure | Path | Intended use | Note |
| --- | --- | --- | --- |
| v0.1 affected-region MSE bar chart | `reports/figures/balanced_affected_region_mse_bar.png` | Historical comparison | Use only when discussing Thayer-BR v0.1 as the previous best model. |
| v0.1 normal vs stress improvement chart | `reports/figures/balanced_normal_vs_stress_improvement_ratio.png` | Historical comparison | Superseded by v0.2 improvement chart for current headline. |
| Thayer-Direct vs Thayer-BR v0.1 scatter | `reports/figures/balanced_direct_vs_balanced_scatter.png` | Historical comparison | Useful for explaining the v0.1 balanced-training step. |
| Thayer-Residual vs Thayer-BR v0.1 scatter | `reports/figures/balanced_old_residual_vs_balanced_scatter.png` | Historical comparison | Useful for explaining the v0.1 balanced-training step. |
| Earlier direct/residual figures | `reports/figures/affected_region_mse_bar.png`, `reports/figures/normal_vs_stress_improvement_ratio.png`, `reports/figures/residual_success_over_direct.png` | Legacy/provenance | Do not use as current headline figures. |

## Recommendation

Main paper figure set:

- `v02_affected_mse_bar.png`
- `v02_improvement_ratio.png`
- `v02_weighted_vs_v01_scatter.png`
- `v02_core_mse.png`
- `v02_multiseed_summary.png`
- `v02_weighted_improvement_example.png`
- `v02_counterexample.png`

Appendix figure set:

- threshold and dilation sensitivity figures from the evaluation audit;
- v0.2 weighted ratio histogram;
- worse-than-identity count chart;
- size audit figures;
- halo-band audit figures;
- visual-vs-metric disagreement grids.
