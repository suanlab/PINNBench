# Statistical Tests

- Tests: exact Wilcoxon signed-rank, bootstrap 95% CI for paired mean difference, paired Cohen's d
- Multiple-comparison correction: Benjamini-Hochberg FDR (q=0.05)
- Seeds: paired by identical seed across compared conditions/models
- Convention: `mean_difference_a_minus_b = group_a - group_b`

## Summary Table

| Comparison | Metric | Group A mean+-std | Group B mean+-std | Mean diff (A-B) | 95% CI | Wilcoxon p | BH-FDR p_adj | Cohen's d | Better |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| RQ1 Heat1D causal ablation | relative_l2_error | 0.3891 +- 0.0537 | 0.3871 +- 0.0524 | 0.0020 | [-0.0008, 0.0047] | 0.3125 | 0.3125 (ns) | 0.5297 | without_causal |
| RQ1b Burgers1D causal ablation | pde_residual_norm | 0.2092 +- 0.0050 | 0.1939 +- 0.0067 | 0.0153 | [0.0108, 0.0200] | 0.0020 | 0.0036 | 1.9700 | without_causal |
| RQ1c AdvDiff1D causal ablation | relative_l2_error | 0.5802 +- 0.1311 | 0.5950 +- 0.1308 | -0.0149 | [-0.0184, -0.0103] | 0.0625 | 0.0677 (ns) | -2.8138 | with_causal |
| RQ3 Heat1D hybrid vs PINN-only | relative_l2_error | 0.0242 +- 0.0104 | 0.4023 +- 0.0480 | -0.3781 | [-0.4041, -0.3506] | 0.0020 | 0.0036 | -8.3272 | hybrid_full |
| RQ4 Wave2D hybrid vs PINN-only | relative_l2_error | 0.4343 +- 0.0780 | 0.7084 +- 0.0102 | -0.2741 | [-0.3174, -0.2270] | 0.0020 | 0.0036 | -3.5534 | hybrid_full |
| RQ6 Wave2D OOD hybrid vs PINN-only | relative_l2_error | 0.9774 +- 0.0264 | 1.0405 +- 0.0128 | -0.0631 | [-0.0889, -0.0373] | 0.0625 | 0.0677 (ns) | -1.8862 | hybrid_full |
| RQ1e KdV1D causal ablation (10-seed) | pde_residual_norm | 0.0096 +- 0.0014 | 0.0105 +- 0.0017 | -0.0008 | [-0.0010, -0.0007] | 0.0020 | 0.0036 | -2.5152 | with_causal |
| RQ1f AllenCahn1D causal ablation (10-seed) | pde_residual_norm | 0.1891 +- 0.0142 | 0.1913 +- 0.0144 | -0.0022 | [-0.0039, -0.0007] | 0.0273 | 0.0355 | -0.7990 | with_causal |
| RQ1e-h KdV1D hybrid vs PINN-only (10-seed) | relative_l2_error | 0.0034 +- 0.0019 | 0.0862 +- 0.2167 | -0.0828 | [-0.2222, -0.0085] | 0.0039 | 0.0056 | -0.3821 | hybrid_full |
| RQ1f-h AllenCahn1D hybrid vs PINN-only (10-seed) | relative_l2_error | 0.0247 +- 0.0132 | 0.2261 +- 0.0652 | -0.2014 | [-0.2449, -0.1725] | 0.0020 | 0.0036 | -3.1239 | hybrid_full |
| RQ-PINO Heat1D hybrid vs PINO (accuracy, 10-seed) | relative_l2_error | 0.0178 +- 0.0046 | 0.7631 +- 0.0166 | -0.7453 | [-0.7552, -0.7343] | 0.0020 | 0.0036 | -41.2846 | hybrid |
| RQ-PINO AdvDiff1D hybrid vs PINO (accuracy, 10-seed) | relative_l2_error | 0.0178 +- 0.0046 | 0.0347 +- 0.0131 | -0.0169 | [-0.0239, -0.0101] | 0.0039 | 0.0056 | -1.4017 | hybrid |
| RQ-PINO Heat1D PINO vs hybrid (residual, 10-seed) | pde_residual_norm | 0.0073 +- 0.0010 | 0.0105 +- 0.0017 | -0.0032 | [-0.0042, -0.0023] | 0.0020 | 0.0036 | -1.9547 | pino |

BH-FDR correction: 10/13 tests remain significant at q=0.05.

## Notes

- **RQ1 Heat1D causal ablation**: Heat1D causal loss should remain a null or near-null result. (p=0.3125, p_adj=0.3125, d=0.5297) **[not significant after FDR]**
- **RQ1b Burgers1D causal ablation**: Burgers1D causal loss should remain a null or slightly negative result. (p=0.0020, p_adj=0.0036, d=1.9700)
- **RQ1c AdvDiff1D causal ablation**: AdvectionDiffusion1D causal loss should remain a weak/null effect. (p=0.0625, p_adj=0.0677, d=-2.8138) **[not significant after FDR]**
- **RQ3 Heat1D hybrid vs PINN-only**: Heat1D hybrid should show a large accuracy gain over PINN-only. (p=0.0020, p_adj=0.0036, d=-8.3272)
- **RQ4 Wave2D hybrid vs PINN-only**: Wave2D hybrid should retain an in-domain accuracy gain over PINN-only. (p=0.0020, p_adj=0.0036, d=-3.5534)
- **RQ6 Wave2D OOD hybrid vs PINN-only**: OOD coefficient shift should reduce the hybrid advantage relative to in-domain. (p=0.0625, p_adj=0.0677, d=-1.8862) **[not significant after FDR]**
- **RQ1e KdV1D causal ablation (10-seed)**: KdV1D causal loss should remain at most a weak directional effect relative to its runtime cost. (p=0.0020, p_adj=0.0036, d=-2.5152)
- **RQ1f AllenCahn1D causal ablation (10-seed)**: AllenCahn1D causal loss should remain a near-null result. (p=0.0273, p_adj=0.0355, d=-0.7990)
- **RQ1e-h KdV1D hybrid vs PINN-only (10-seed)**: KdV1D hybrid should improve accuracy over PINN-only. (p=0.0039, p_adj=0.0056, d=-0.3821)
- **RQ1f-h AllenCahn1D hybrid vs PINN-only (10-seed)**: AllenCahn1D hybrid should improve accuracy over PINN-only. (p=0.0020, p_adj=0.0036, d=-3.1239)
- **RQ-PINO Heat1D hybrid vs PINO (accuracy, 10-seed)**: Heat1D hybrid should strongly outperform PINO on solution accuracy. (p=0.0020, p_adj=0.0036, d=-41.2846)
- **RQ-PINO AdvDiff1D hybrid vs PINO (accuracy, 10-seed)**: AdvDiff1D hybrid should outperform PINO on solution accuracy. (p=0.0039, p_adj=0.0056, d=-1.4017)
- **RQ-PINO Heat1D PINO vs hybrid (residual, 10-seed)**: Heat1D PINO should achieve lower PDE residual than the hybrid model. (p=0.0020, p_adj=0.0036, d=-1.9547)
