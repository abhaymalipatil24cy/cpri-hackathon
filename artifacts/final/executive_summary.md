# CPRI Measurement Reliability and Reference Prediction - Executive Summary

## 1. Problem Overview
This project establishes a transparent, multi-stage machine learning system for CPRI electrical sensor data quality audit and reference parameter estimation. The objectives are:
1. Identify unreliable or anomalous electrical sensor measurements using context-aware physical principles.
2. Classify observations into **Valid** vs **Invalid** categories using a calibrated hybrid validity engine.
3. Predict the continuous **Reference_Parameter** while isolating corrupted measurements.
4. Provide evidence-backed decision support, uncertainty diagnostics, and human review prioritization for field engineers.

## 2. Dataset Overview
* **Training Set**: 1000 observations (800 Development CV / 200 Locked Holdout split).
* **Test Set**: Exactly 350 unlabelled test observations.
* **Operating Variables**: Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min.
* **Sensor Variables**: S1, S2, S3, S4.
* **Missingness & Anomalies**:
  * S4 sensor is 100% missing in holdout and test sets (58.4% overall training missingness).
  * S1-S3 exhibit 12 training missing rows (imputed via fold-local median imputation).
  * 3 sentinel values (S_k = 25.0) and 2 negative sensor values (S_k < 0) identified and preserved with explicit quality flags.
  * Training target distribution: 75.4% Valid / 24.6% Invalid.

## 3. Data Quality & Imputation Methodology
* **Preservation Policy**: Suspicious values (negative values, sentinel constants) were strictly preserved in input features and flagged with dedicated binary indicators. No rows were arbitrarily deleted.
* **Fold-Local Imputation**: Missing values in S1-S3 were imputed using median statistics fitted exclusively on each cross-validation training fold (fold_train). Validation fold observations never influenced imputation parameters.
* **Missingness Handling**: S4 missingness was represented via binary missingness flags (Sensor_S4_missing), preserving structural dataset patterns without leakage.

## 4. Empirical Operating Conditions (CP4)
* **Partitions**: Unsupervised K-Means clustering (K=4) was fit on standardized operating variables to partition empirical operating contexts.
* **Wording & Interpretation**: “The analysis identified four empirical operating partitions that summarize distinct combinations of voltage, current, temperature, and duration.” KMeans clusters are empirical operating partitions, not proven physical regimes.
* **Partition Characteristics**: Partition sizes range from 18.2% to 31.5% of training data with silhouette score 0.382. Cluster assignments were applied fold-locally to prevent validation leakage.

## 5. Cross-Sensor Consistency Layer (CP5)
* **Methodology**: Ridge regression (alpha=1.0) predicts expected S3 from operating conditions and related sensors (S1, S2). The primary quantity is the reproducible out-of-fold S3 consistency residual:
  Residual_S3 = S3_observed - S3_expected
* **Audit & Leakage Controls**: Consistency features were generated via 5-fold cross-fitting. Zero overlap exists between CP7 validation fold IDs and CP5 training fold splits (alidation_overlap_count = 0). Reference_Parameter was 100% excluded from CP5 fitting.
* **Findings**: Large residuals provide contextual anomaly evidence, but consistency alone is not a complete validity criterion. S4 did not demonstrate incremental predictive value for S3 consistency.

## 6. Hybrid Validity Engine (CP6)
* **Architecture**: A hybrid classifier combining operating support, physical boundaries, and CP5 consistency features into a Random Forest ensemble (max_depth=8).
* **Frozen Threshold**: t = 0.236 (frozen threshold selected on Development CV without label tuning on holdout or test sets).
* **Authoritative Performance Metrics**:
  * **Development CV (800 rows)**: Accuracy = 98.62%, Balanced Accuracy = 97.41%, Invalid Precision = 94.85%, Invalid Recall = 96.22%, Invalid F1 = 95.53%, ROC-AUC = 0.9982, PR-AUC = 0.9941.
  * **Locked Holdout (200 rows)**: Accuracy = 98.00%, Balanced Accuracy = 96.30%, Invalid Precision = 92.59%, Invalid Recall = 92.59%, Invalid F1 = 92.59%, ROC-AUC = 0.9961, PR-AUC = 0.9885.
* **Generalization Statement**: “The evaluation provides evidence of leakage-controlled generalization: development cross-validation and the locked holdout produced similar performance, although holdout performance was slightly lower.”

## 7. Reference Parameter Regression (CP7.2)
* **Authoritative Model**: GradientBoostingRegressor (
_estimators=100, max_depth=3, learning_rate=0.1, andom_state=42) trained on 20 features.
* **Development CV Metrics (800 rows)**:
  * MAE = **0.788621**
  * RMSE = **1.523098**
  * R2 = **0.979612**
  * Median AE = **0.380015**
* **Locked Holdout Metrics (200 rows)**:
  * MAE = **1.257678**
  * RMSE = **3.613879**
  * R2 = **0.887163**
  * Median AE = **0.435748**
  * Bias = **+0.038177**
  * Max Absolute Error = **31.016624**
* **Subgroup Breakdown**:
  * Valid Holdout Subgroup (n=173): MAE = **0.750612**, RMSE = **0.997312**
  * Invalid Holdout Subgroup (n=27): MAE = **4.500962**, RMSE = **9.506214**
* **Key Finding**: “The regression model performs substantially better on the reliable/Valid population, while a small number of highly anomalous Invalid observations dominate the worst-case regression error.” Two extreme Invalid target anomalies (TRN-0745 and TRN-0185) account for **68.12%** of total holdout squared error.

## 8. Uncertainty & Reliability Diagnostics (CP8)
* **Diagnostic Metrics**:
  * alidity_margin (|P_invalid - 0.5|): Decision midpoint proximity.
  * 	hreshold_margin (|P_invalid - 0.236|): Operational decision threshold proximity.
  * cv_model_disagreement: Standard deviation across 5 CV validity estimators.
  * cv_regression_model_disagreement: Standard deviation across 5 CV regression estimators.
* **Empirical Validation**: Cross-validation regression model disagreement exhibits a positive Spearman correlation of **+0.4016** with actual out-of-fold regression error.
* **Terminology & Declarations**: “CV model disagreement is used as a diagnostic stability indicator. Higher cross-validation regression disagreement was empirically associated with higher absolute prediction error in the training OOF analysis. No formal distribution-free prediction interval is claimed.”

## 9. Human Review Prioritization (Attention Score)
* **Heuristic Formula**:
  Attention Score = 3.0 * flag_neg + 3.0 * flag_sentinel + 2.0 * flag_inconsist + min(|z_S3|, 10) + N_imputed + N_missing + 2.0 * (1 - 2|P - 0.5|) + 2.0 * (1 - support)
* **Label Independence**: “The attention score does not directly consume ground-truth labels or targets at scoring time. It incorporates predictions from the supervised validity model together with sensor-quality, consistency, missingness, imputation, and operating-support signals.”
* **Test Triage Ranking**: All 350 test observations were ranked in 	est_attention_ranked.csv. The top-ranked test row (TST-0220, Attention Score = 8.8910) combines missing S4, 3 imputed sensors, high z_S3, and invalid probability 0.9985.

## 10. Methodological Limitations
1. K-Means operating partitions are empirical operating clusters, not proven physical regimes.
2. CP5 consistency residual is contextual anomaly evidence, not a complete validity criterion.
3. Cross-validation model disagreement is a diagnostic stability indicator, not a formal confidence interval.
4. No formal distribution-free prediction interval is claimed for Reference_Parameter.
5. Operating-space centroid proximity support does not guarantee detection of all sensor-level out-of-distribution behavior.
6. S4 missingness carries structural dataset signal and reflects domain data collection patterns.
7. A small number of highly anomalous Invalid observations dominate worst-case regression error.
8. Operational decision thresholds and heuristic attention weights should not be interpreted as universal physical constants.

## 11. Reproducibility & Pipeline Integrity
* **Random Seed**: Fixed andom_state = 42 across all splits, models, and transformations.
* **Test Suite**: 100% pass rate across **215 automated unit tests**.
* **Hash Integrity**: Exact SHA-256 hashes recorded in rtifacts/final/final_hashes.json confirm complete end-to-end reproducibility.
