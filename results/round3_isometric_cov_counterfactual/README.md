# Frozen isometric PE: internal-GS covariance counterfactual

Only the key projected covariance is replaced with the regional covariance of primitive centers. This is a frozen checkpoint intervention, not retraining. Query covariance, all means, labels, masks, camera, content encoder and checkpoint parameters remain identical. Resolution 0.25 is added exactly once inside the unchanged model.

| partition | scene | full CE | centeronly CE | full−centeronly | crossed seed/camera 95% CI |
|---|---|---:|---:|---:|---|
| primary | all | 1.148221 | 1.146107 | +0.002113 | [+0.001032,+0.003539] |
| primary | teatime | 1.237023 | 1.234946 | +0.002077 | [+0.000085,+0.005180] |
| primary | waldo_kitchen | 1.061818 | 1.059670 | +0.002148 | [+0.001642,+0.002603] |
| alt | all | 1.146965 | 1.145157 | +0.001808 | [+0.001198,+0.002376] |
| alt | teatime | 1.233065 | 1.231845 | +0.001220 | [+0.000271,+0.001992] |
| alt | waldo_kitchen | 1.063466 | 1.061087 | +0.002378 | [+0.001812,+0.003010] |

All original full aggregate and per-pair archived metrics match. The original experiment did not save token probabilities, so archive validation cannot establish bitwise identity to a historical token-probability file. This replay saves both probability tensors and verifies an independent repeated original-full forward pass exactly. All checkpoint and source hashes, row metrics, projection errors and seed/camera differences are retained in summary.json.
