# SkinPi Results Summary

Model: MobileViTv2-100, trained on BCN20000. Selected checkpoint: epoch 53 of 100. Decision threshold: 0.64 (Youden's index on the BCN20000 validation split).

| Dataset | Images | Sensitivity | Specificity | AUC-ROC | Role |
|---|---|---|---|---|---|
| BCN20000 (validation split) | 2,047 | 88.09% | 81.83% | 0.9281 | Validation (threshold tuned here) |
| HAM10000 | 10,015 | 77.38% | 72.31% | 0.8318 | Development set (used for checkpoint selection) |
| PH2 | 200 | 92.50% | 67.50% | 0.9064 | External test |
| PAD-UFES-20 | 2,298 | 75.39% | 44.00% | 0.6563 | External test |

Note: HAM10000 sensitivity was monitored each epoch and influenced checkpoint selection, so its numbers are optimistic. PH2 and PAD-UFES-20 were each evaluated once with the fixed checkpoint and threshold.