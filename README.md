# Formula 1 Podium Prediction

**CSEN 903: Advanced Computer Lab — Milestone 1**  
**Kaggle Notebook:** [ambitious-intelligence-ms1](https://www.kaggle.com/code/amrelmehrath/ambitious-intelligence-ms1)  
**GitHub Repository:** [MennaAshraf1007/ACL_9th-F1](https://github.com/MennaAshraf1007/ACL_9th-F1)

## Overview
A leakage-safe machine learning pipeline that predicts whether a Formula 1 driver finishes on the podium (`positionOrder <= 3`) using strictly pre-race information available after qualifying. Built on the 1950–2024 Formula 1 World Championship dataset (14 relational tables consolidated into 26,668 driver-race rows), the final race-aware ensemble achieves **0.717 PR-AUC** and **0.939 ROC-AUC** on unseen 2022–2024 seasons.

## Pipeline Summary
* **Data Audit & Cleaning:** Replaced `\N` sentinels, deduplicated 1950s shared drives (keeping the best `positionOrder`), parsed lap and qualifying durations into seconds, shifted `grid = 0` starts behind the field (`grid_pos`), and flagged red-flag pit stop outliers (`> 60s`).
* **Leakage-Safe Feature Engineering:** Constructed 21 historical and pre-race features using grouped `shift(1)` rolling windows (qualifying/grid positions, penalty deltas, recent driver & constructor form, running lap positions, career stats, and normalized `points_share`) plus 13 race-relative features (within-race ranks, gaps to the leader, and field size).
* **Temporal Split:** Trained on **2003–2019** (321 races, 6,831 rows), tuned on **2020–2021** (39 races, 780 rows), and evaluated once on **2022–2024** (68 races, 1,359 rows).
* **Explainability & Inference:** Audited model behavior via permutation importance, global/local SHAP, LIME, and feature-group ablations, and packaged a missing-value-resilient `predict_podium(entries)` inference function.

## Test Results (2022–2024 Seasons)

| Model | PR-AUC (Primary) | ROC-AUC | F1 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline: `grid <= 3` Rule | 0.647 | 0.914 | 0.649 | 0.650 | 0.647 |
| Baseline: Logistic Regression (Grid + Quali) | 0.652 | 0.919 | 0.672 | 0.600 | 0.765 |
| Shallow Neural Network (Keras) | 0.677 | 0.929 | 0.661 | 0.617 | 0.711 |
| Logistic Regression | 0.696 | 0.935 | 0.671 | 0.624 | 0.725 |
| Random Forest | 0.707 | **0.939** | 0.675 | 0.650 | 0.701 |
| Gradient Boosting | 0.709 | 0.937 | 0.662 | 0.644 | 0.681 |
| **Final: Race-Aware Ensemble (Bonus)** | **0.717** | **0.939** | **0.683** | 0.617 | **0.765** |

## Usage
```python
# Score a pre-race entry table (handles missing features via conservative fill rules)
predictions = predict_podium(entries_df)
```

## Team — Ambitious Intelligence
* **Lama Abdeldayem** (58-2085)
* **Menna Ashraf** (58-1027)
* **Amr Osama** (58-9260)
* **Amir Osama Mohamed** (58-1793)