# Frozen deterministic Gaussian position analysis

Candidate full passes all frozen criteria against validation-selected fixed and the PCK-optimal point guard. No alternative candidate is selected from test.

| KL condition | full | baseline | difference | relative change | paired named-scan central95% CI |
|---|---:|---:|---:|---:|---|
| all | 0.304822 | 0.324541 | -0.019719 | -6.08% | [-0.025666,-0.013993] |
| q8_k16 | 0.239970 | 0.247905 | -0.007934 | -3.20% | [-0.010922,-0.005220] |
| q16_k8 | 0.315858 | 0.349060 | -0.033201 | -9.51% | [-0.044908,-0.021494] |
| q8_k8 | 0.466723 | 0.482461 | -0.015738 | -3.26% | [-0.021589,-0.010324] |
| q16_k16 | 0.196736 | 0.218738 | -0.022002 | -10.06% | [-0.029868,-0.014149] |

PCK@8px guards use q8_k8 only. Differences below are percentage points, not percent relative changes.

- Full minus fixed: -0.008 pp, central95% CI [-0.064,+0.046] pp.
- Full minus point_pck: -0.089 pp, central95% CI [-0.176,-0.013] pp.

The sampling units are five named scan/place groups, with limited bootstrap precision and no guarantee that all are independent physical rooms. Inference has no trained parameters and no training-seed replication. Isotropic and other comparisons are exploratory, without simultaneous interval coverage. Brown/MIT/home and leave-one-scan sensitivities are descriptive. CE, JS, point EPE, equal-query KL, pooled ratios and coverage diagnostics are retained in summary.json.
