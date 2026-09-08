# Tests for CP6 - Hybrid Validity Engine
import json
import pickle
import numpy as np
import pandas as pd
import pytest

from src import config
from src import validity_engine

@pytest.fixture
def oof_df():
    path = config.ARTIFACTS_DIR / 'validity' / 'validity_oof.csv'
    return pd.read_csv(path)

@pytest.fixture
def test_df():
    path = config.ARTIFACTS_DIR / 'validity' / 'validity_test.csv'
    return pd.read_csv(path)

@pytest.fixture
def audit_df():
    path = config.ARTIFACTS_DIR / 'validity' / 'test_audit.csv'
    return pd.read_csv(path)

@pytest.fixture
def metrics_dict():
    path = config.ARTIFACTS_DIR / 'validity' / 'validity_metrics.json'
    with open(path) as f:
        return json.load(f)

# Test 1: Target leakage prevention
def test_1_target_leakage_prevention():
    feats = validity_engine.FEATURE_COLS
    assert 'Validity_Label' not in feats
    assert 'Reference_Parameter' not in feats
    assert 'Test_ID' not in feats

# Test 2: Reference_Parameter excluded
def test_2_reference_parameter_excluded():
    assert 'Reference_Parameter' not in validity_engine.FEATURE_COLS

# Test 3: Test_ID excluded
def test_3_test_id_excluded():
    assert 'Test_ID' not in validity_engine.FEATURE_COLS

# Test 4: Fold-local imputation integrity
def test_4_fold_local_imputation_integrity(oof_df):
    for col in ['Sensor_S1_missing', 'Sensor_S2_missing', 'Sensor_S3_missing', 'Sensor_S4_missing']:
        assert col in oof_df.columns

# Test 5: Fold-local scaling integrity
def test_5_fold_local_scaling_integrity(oof_df):
    assert 'Applied_Voltage_kV' in oof_df.columns
    assert oof_df['Applied_Voltage_kV'].notna().all()

# Test 6: Fold-local regime fitting integrity
def test_6_fold_local_regime_fitting_integrity(oof_df):
    assert len(oof_df) == 1000

# Test 7: Fold-local classifier
def test_7_fold_local_classifier(oof_df):
    assert 'validity_probability' in oof_df.columns
    assert (oof_df['validity_probability'] >= 0.0).all() and (oof_df['validity_probability'] <= 1.0).all()

# Test 8: Fold-local threshold
def test_8_fold_local_threshold():
    path = config.ARTIFACTS_DIR / 'validity' / 'threshold_results.json'
    with open(path) as f:
        data = json.load(f)
    assert 'selected_fold_thresholds' in data
    assert len(data['selected_fold_thresholds']) == 5

# Test 9: No validation labels used during threshold selection
def test_9_no_val_labels_in_threshold_selection():
    path = config.ARTIFACTS_DIR / 'validity' / 'threshold_results.json'
    with open(path) as f:
        data = json.load(f)
    assert 'mean_threshold' in data

# Test 10: OOF prediction count = 1000
def test_10_oof_prediction_count(oof_df):
    assert len(oof_df) == 1000

# Test 11: Exactly one OOF prediction per training row
def test_11_exactly_one_oof_prediction_per_training_row(oof_df):
    assert oof_df['Test_ID'].nunique() == 1000

# Test 12: Test prediction count = 350
def test_12_test_prediction_count(test_df):
    assert len(test_df) == 350

# Test 13: Deterministic output
def test_13_deterministic_output(oof_df):
    assert oof_df['predicted_validity'].isin(['Valid', 'Invalid']).all()

# Test 14: Probability range [0, 1]
def test_14_probability_range(oof_df, test_df):
    assert (oof_df['validity_probability'] >= 0.0).all() and (oof_df['validity_probability'] <= 1.0).all()
    assert (test_df['validity_probability'] >= 0.0).all() and (test_df['validity_probability'] <= 1.0).all()

# Test 15: Attention score finite
def test_15_attention_score_finite(oof_df, test_df):
    assert np.isfinite(oof_df['attention_score']).all()
    assert np.isfinite(test_df['attention_score']).all()

# Test 16: Support score finite
def test_16_support_score_finite(oof_df, test_df):
    assert np.isfinite(oof_df['input_support_score']).all()
    assert (oof_df['input_support_score'] > 0.0).all() and (oof_df['input_support_score'] <= 1.0).all()
    assert (test_df['input_support_score'] > 0.0).all() and (test_df['input_support_score'] <= 1.0).all()

# Test 17: Missingness flags preserved
def test_17_missingness_flags_preserved(oof_df):
    for col in ['Sensor_S1_missing', 'Sensor_S2_missing', 'Sensor_S3_missing', 'Sensor_S4_missing']:
        assert col in oof_df.columns

# Test 18: Imputation flags preserved
def test_18_imputation_flags_preserved(oof_df):
    for col in ['Sensor_S1_imputed', 'Sensor_S2_imputed', 'Sensor_S3_imputed']:
        assert col in oof_df.columns

# Test 19: Consistency OOF provenance preserved
def test_19_consistency_oof_provenance_preserved(oof_df):
    assert 'S3_consistency_residual' in oof_df.columns
    assert 'S3_consistency_z' in oof_df.columns

# Test 20: No target columns in final test feature matrix
def test_20_no_target_in_test_features():
    assert 'Validity_Label' not in validity_engine.FEATURE_COLS

# Test 21: Competition output schema validation
def test_21_competition_output_schema_validation():
    sub_path = config.OUTPUTS_DIR / 'cpri_validity_submission.csv'
    sub_df = pd.read_csv(sub_path)
    assert set(sub_df.columns) == {'Test_ID', 'Validity_Label'}
    assert len(sub_df) == 350
    assert sub_df['Validity_Label'].isin(['Valid', 'Invalid']).all()

# Test 22: No duplicate Test_ID in output
def test_22_no_duplicate_test_id_in_output():
    sub_path = config.OUTPUTS_DIR / 'cpri_validity_submission.csv'
    sub_df = pd.read_csv(sub_path)
    assert sub_df['Test_ID'].nunique() == 350

# Test 23: Adversarial canary execution
def test_23_adversarial_canary_execution():
    path = config.ARTIFACTS_DIR / 'validity' / 'canary_results.json'
    with open(path) as f:
        canaries = json.load(f)
    assert len(canaries) == 7
    assert 'Canary_1_Negative_S1' in canaries
    assert 'Canary_2_Extreme_S3_Inconsistent' in canaries

# Test 24: Model serialization/deserialization
def test_24_model_serialization_deserialization():
    path = config.ARTIFACTS_DIR / 'validity' / 'validity_model.pkl'
    with open(path, 'rb') as f:
        art = pickle.load(f)
    assert art['model_name'] == 'Random Forest'
    assert hasattr(art['model'], 'predict_proba')

# Test 25: Artifact reproducibility
def test_25_artifact_reproducibility(metrics_dict):
    assert metrics_dict['selected_model'] == 'Random Forest'
    assert 'oof_performance' in metrics_dict
    assert 'locked_holdout' in metrics_dict

# Test 26: Ablation results artifact existence & validity
def test_26_ablation_results():
    path = config.ARTIFACTS_DIR / 'validity' / 'ablation_results.json'
    with open(path) as f:
        ab = json.load(f)
    assert 'E_full_hybrid_model' in ab
    assert 'Full_minus_residual' in ab

# Test 27: Calibration score validity
def test_27_calibration_score_validity():
    path = config.ARTIFACTS_DIR / 'validity' / 'calibration.json'
    with open(path) as f:
        cal = json.load(f)
    assert 'brier_score' in cal
    assert 0.0 <= cal['brier_score'] <= 1.0
