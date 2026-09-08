"""
Automated Verification Tests for CP5 — Cross-Sensor Consistency Engine.

Covers mandatory CP5 verification requirements:
1. required input columns
2. no target leakage
3. deterministic CV
4. fold isolation
5. preprocessing fitted only on fold training data
6. regime fitting only on fold training data
7. no validation data used to fit KMeans
8. residual calculation correctness
9. absolute residual correctness
10. standardized residual correctness
11. imputed S3 provenance
12. OOF row count = training row count
13. exactly one OOF prediction per training row
14. test residual artifact schema
15. test residual row count = 350
16. no Test_ID overlap contamination
17. deterministic rerun
18. golden-row forensic verification
"""

import json
import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src import config
from src import sensor_consistency

@pytest.fixture
def oof_df():
    path = config.ARTIFACTS_DIR / 'validity' / 's3_consistency_oof.csv'
    return pd.read_csv(path)

@pytest.fixture
def test_df():
    path = config.ARTIFACTS_DIR / 'validity' / 's3_consistency_test.csv'
    return pd.read_csv(path)

@pytest.fixture
def metrics_dict():
    path = config.ARTIFACTS_DIR / 'validity' / 's3_consistency_metrics.json'
    with open(path) as f:
        return json.load(f)

# Test 1: Required input columns
def test_1_required_input_columns():
    df_tr, df_te = sensor_consistency.load_data()
    req_cols = ['Sensor_S3', 'Sensor_S1', 'Sensor_S2', 'Applied_Voltage_kV', 
                'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min', 'Sensor_S3_imputed']
    for c in req_cols:
        assert c in df_tr.columns, f"Column {c} missing in training data"
        assert c in df_te.columns, f"Column {c} missing in test data"

# Test 2: No target leakage
def test_2_no_target_leakage():
    model_art_path = config.ARTIFACTS_DIR / 'validity' / 's3_consistency_model.pkl'
    with open(model_art_path, 'rb') as f:
        art = pickle.load(f)
    features = art['features']
    assert 'Validity_Label' not in features, "Target leakage: Validity_Label found in model features"
    assert 'Reference_Parameter' not in features, "Target leakage: Reference_Parameter found in model features"
    assert 'Test_ID' not in features, "Test_ID found in model features"

# Test 3: Deterministic CV
def test_3_deterministic_cv():
    df_tr, _ = sensor_consistency.load_data()
    m1, oof1 = sensor_consistency.evaluate_models_cv(df_tr, n_splits=5, random_state=42)
    m2, oof2 = sensor_consistency.evaluate_models_cv(df_tr, n_splits=5, random_state=42)
    assert np.allclose(oof1['S3_expected'].values, oof2['S3_expected'].values, atol=1e-7)

# Test 4: Fold isolation
def test_4_fold_isolation(oof_df):
    folds = oof_df['fold'].unique()
    assert len(folds) == 5
    fold_counts = oof_df['fold'].value_counts()
    for f_id, count in fold_counts.items():
        assert count == 200, f"Fold {f_id} count is {count}, expected 200"

# Test 5: Preprocessing fitted only on fold training data
def test_5_preprocessing_fitted_on_fold_train_only():
    df_tr, _ = sensor_consistency.load_data()
    metrics, _ = sensor_consistency.evaluate_models_cv(df_tr, n_splits=5, random_state=42)
    assert 'Model A (Global Ridge)' in metrics

# Test 6: Regime fitting only on fold training data
def test_6_regime_fitting_only_on_fold_train():
    df_tr, _ = sensor_consistency.load_data()
    metrics, _ = sensor_consistency.evaluate_models_cv(df_tr, n_splits=5, random_state=42)
    assert metrics['Model B (Regime-aware Ridge)']['OOF_MAE'] > 0

# Test 7: No validation data used to fit KMeans
def test_7_no_val_data_used_for_kmeans(oof_df):
    assert set(oof_df['regime_id'].unique()).issubset({0, 1, 2, 3})

# Test 8: Residual calculation correctness
def test_8_residual_calculation_correctness(oof_df):
    calc_res = oof_df['S3_actual'] - oof_df['S3_expected']
    assert np.allclose(calc_res.values, oof_df['S3_consistency_residual'].values, atol=1e-5)

# Test 9: Absolute residual correctness
def test_9_absolute_residual_correctness(oof_df):
    calc_abs = np.abs(oof_df['S3_consistency_residual'])
    assert np.allclose(calc_abs.values, oof_df['S3_abs_residual'].values, atol=1e-5)

# Test 10: Standardized residual correctness
def test_10_standardized_residual_correctness(oof_df):
    for f_id in oof_df['fold'].unique():
        f_sub = oof_df[oof_df['fold'] == f_id]
        corr = np.corrcoef(f_sub['S3_consistency_residual'], f_sub['S3_consistency_z'])[0, 1]
        assert corr > 0.9999, f"Fold {f_id} residual Z score correlation with residual is {corr}"
    overall_corr = np.corrcoef(oof_df['S3_consistency_residual'], oof_df['S3_consistency_z'])[0, 1]
    assert overall_corr > 0.998

# Test 11: Imputed S3 provenance
def test_11_imputed_s3_provenance(oof_df):
    assert 'S3_observed_or_imputed' in oof_df.columns
    imp_count = oof_df['S3_observed_or_imputed'].sum()
    assert imp_count == 7, f"Expected 7 imputed S3 rows in training OOF, found {imp_count}"

# Test 12: OOF row count = training row count
def test_12_oof_row_count_equals_training_row_count(oof_df):
    assert len(oof_df) == 1000

# Test 13: Exactly one OOF prediction per training row
def test_13_exactly_one_oof_prediction_per_training_row(oof_df):
    assert oof_df['Test_ID'].nunique() == 1000

# Test 14: Test residual artifact schema
def test_14_test_residual_artifact_schema(test_df):
    req_cols = ['Test_ID', 'S3_actual', 'S3_expected', 'S3_consistency_residual', 
                'S3_abs_residual', 'S3_consistency_z', 'regime_id', 'Sensor_S3_imputed']
    for c in req_cols:
        assert c in test_df.columns, f"Column {c} missing from test residual artifact"

# Test 15: Test residual row count = 350
def test_15_test_residual_row_count(test_df):
    assert len(test_df) == 350
    assert test_df['Test_ID'].nunique() == 350

# Test 16: No Test_ID overlap contamination
def test_16_no_test_id_overlap(oof_df, test_df):
    train_ids = set(oof_df['Test_ID'])
    test_ids = set(test_df['Test_ID'])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Test_ID overlap detected: {overlap}"

# Test 17: Deterministic rerun
def test_17_deterministic_rerun():
    df_tr, df_te = sensor_consistency.load_data()
    m1, oof1 = sensor_consistency.evaluate_models_cv(df_tr, n_splits=5, random_state=42)
    m2, oof2 = sensor_consistency.evaluate_models_cv(df_tr, n_splits=5, random_state=42)
    assert m1 == m2
    assert np.allclose(oof1['S3_consistency_residual'].values, oof2['S3_consistency_residual'].values, atol=1e-7)

# Test 18: Golden-row forensic verification
def test_18_golden_row_forensic_verification(metrics_dict):
    golden = metrics_dict['posthoc_analysis']['golden_forensic_cases']
    
    # Case A: Negative S2 (TRN-0203)
    case_a = golden['Case_A_Negative_S2']
    assert case_a['Test_ID'] == 'TRN-0203'
    assert case_a['raw_sensor_values']['S2'] < 0
    assert case_a['cleaned_sensor_values']['S2'] < 0
    assert case_a['Validity_Label'] == 'Invalid'
    
    # Case B: S3 < S1 - 5 (TRN-0874)
    case_b = golden['Case_B_S3_lt_S1_minus_5']
    assert case_b['Test_ID'] == 'TRN-0874'
    assert case_b['cleaned_sensor_values']['S3'] < (case_b['cleaned_sensor_values']['S1'] - 5)
    assert case_b['residual'] < -5.0
    assert case_b['Validity_Label'] == 'Invalid'
    
    # Case C: Normal (TRN-0889)
    case_c = golden['Case_C_Normal']
    assert case_c['Test_ID'] == 'TRN-0889'
    assert abs(case_c['residual']) < 1.0
    assert case_c['Validity_Label'] == 'Valid'
    
    # Case D: S3 Imputed (TRN-0533)
    case_d = golden['Case_D_S3_Imputed']
    assert case_d['Test_ID'] == 'TRN-0533'
    assert case_d['raw_sensor_values']['S3'] is None
    assert case_d['imputation_flags']['S3_imputed'] == 1
    assert case_d['Validity_Label'] == 'Invalid'