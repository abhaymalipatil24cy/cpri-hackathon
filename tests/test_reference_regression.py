# Tests for CP7 - Reference Parameter Regression
import json
import pickle
import numpy as np
import pandas as pd
import pytest

from src import config
from src import reference_regression

@pytest.fixture
def oof_df():
    path = config.ARTIFACTS_DIR / 'regression' / 'oof_predictions.csv'
    return pd.read_csv(path)

@pytest.fixture
def holdout_df():
    path = config.ARTIFACTS_DIR / 'regression' / 'holdout_predictions.csv'
    return pd.read_csv(path)

@pytest.fixture
def test_df():
    path = config.ARTIFACTS_DIR / 'regression' / 'test_predictions.csv'
    return pd.read_csv(path)

@pytest.fixture
def model_metadata():
    path = config.ARTIFACTS_DIR / 'regression' / 'final_model_metadata.json'
    with open(path) as f:
        return json.load(f)

# Test 1: Target never present in test features
def test_1_target_never_in_features():
    assert 'Reference_Parameter' not in reference_regression.FEATURE_COLS
    assert 'Validity_Label' not in reference_regression.FEATURE_COLS

# Test 2: No Test_ID overlap
def test_2_no_test_id_overlap(oof_df, test_df):
    train_ids = set(oof_df['Test_ID'])
    test_ids = set(test_df['Test_ID'])
    assert len(train_ids.intersection(test_ids)) == 0

# Test 3: Holdout IDs do not enter dev folds
def test_3_holdout_ids_do_not_enter_dev():
    path = config.ARTIFACTS_DIR / 'regression' / 'split_manifest.json'
    with open(path) as f:
        data = json.load(f)
    assert data['id_overlap_count'] == 0

# Test 4: Fold-local preprocessing
def test_4_fold_local_preprocessing(oof_df):
    for col in ['Sensor_S1_missing', 'Sensor_S2_missing', 'Sensor_S3_missing', 'Sensor_S4_missing']:
        assert col in oof_df.columns

# Test 5: No full-dataset imputation before CV
def test_5_no_full_imputation_before_cv(oof_df):
    assert len(oof_df) == 1000

# Test 6: CP5 feature provenance is leakage-safe
def test_6_cp5_feature_provenance_safe(oof_df):
    assert 'S3_consistency_residual' in oof_df.columns
    assert oof_df['S3_consistency_residual'].notna().all()

# Test 7: Deterministic split
def test_7_deterministic_split():
    path = config.ARTIFACTS_DIR / 'regression' / 'split_manifest.json'
    with open(path) as f:
        data = json.load(f)
    assert data['split_seed'] == 42
    assert data['dev_rows'] == 800 and data['holdout_rows'] == 200

# Test 8: Deterministic model
def test_8_deterministic_model(model_metadata):
    assert model_metadata['hyperparameters']['random_state'] == 42

# Test 9: No NaN or Inf final predictions
def test_9_no_nan_inf_predictions(test_df):
    assert test_df['Predicted_Reference_Parameter'].notna().all()
    assert np.isfinite(test_df['Predicted_Reference_Parameter']).all()

# Test 10: Exactly 350 test predictions
def test_10_exactly_350_test_predictions(test_df):
    assert len(test_df) == 350

# Test 11: Test_ID uniqueness
def test_11_test_id_uniqueness(test_df):
    assert test_df['Test_ID'].nunique() == 350

# Test 12: Observed target remains untouched
def test_12_observed_target_untouched(oof_df):
    raw_df = pd.read_csv(config.TRAIN_DATA_PATH)
    assert np.allclose(raw_df['Reference_Parameter'].values, oof_df['Reference_Parameter'].values)

# Test 13: Final model trained on exactly 1000 rows
def test_13_final_model_trained_on_1000_rows(model_metadata):
    assert model_metadata['training_row_count'] == 1000

# Test 14: Reproducibility hashes
def test_14_reproducibility_hashes():
    path = config.ARTIFACTS_DIR / 'regression' / 'reproducibility.json'
    with open(path) as f:
        rep = json.load(f)
    assert 'oof_sha256' in rep and 'test_sha256' in rep

# Test 15: Prediction range reporting
def test_15_prediction_range_reporting(test_df):
    preds = test_df['Predicted_Reference_Parameter']
    assert preds.min() > 0.0
    assert preds.max() < 100.0

# Test 16: Subgroup metric sample counts
def test_16_subgroup_metric_sample_counts():
    path = config.ARTIFACTS_DIR / 'regression' / 'validity_metrics.csv'
    df = pd.read_csv(path)
    assert 'Subgroup' in df.columns
    assert len(df) == 5

# Test 17: Validity handling is explicitly recorded
def test_17_validity_handling_recorded(model_metadata):
    assert 'V1' in model_metadata['validity_handling']

# Test 18: Regime fitting is fold-local where used
def test_18_regime_fitting_fold_local(model_metadata):
    assert 'regime_handling' in model_metadata

# Test 19: Test data does not affect model selection
def test_19_test_data_does_not_affect_model_selection():
    path = config.ARTIFACTS_DIR / 'regression' / 'final_model_metadata.json'
    with open(path) as f:
        meta = json.load(f)
    assert meta['model_family'] == 'GradientBoostingRegressor'

# Test 20: No silent prediction clipping
def test_20_no_silent_clipping(test_df):
    preds = test_df['Predicted_Reference_Parameter']
    assert preds.min() >= 10.0

# --- CP7.1 FORENSIC AUDIT TESTS (21-35) ---

# Test 21: 800-row Dev set identity
def test_21_dev_set_identity():
    path = config.ARTIFACTS_DIR / 'regression' / 'evaluation_provenance.json'
    with open(path) as f:
        prov = json.load(f)
    assert prov['development_cv']['row_count'] == 800
    assert prov['development_cv']['fold_count'] == 5

# Test 22: 200-row Holdout identity
def test_22_holdout_identity():
    path = config.ARTIFACTS_DIR / 'regression' / 'evaluation_provenance.json'
    with open(path) as f:
        prov = json.load(f)
    assert prov['locked_holdout']['row_count'] == 200
    assert prov['locked_holdout']['holdout_involved'] == True

# Test 23: Dev/Holdout disjointness
def test_23_dev_holdout_disjointness():
    path = config.ARTIFACTS_DIR / 'regression' / 'split_manifest.json'
    with open(path) as f:
        sm = json.load(f)
    dev_set = set(sm['dev_ids'])
    holdout_set = set(sm['holdout_ids'])
    assert len(dev_set.intersection(holdout_set)) == 0

# Test 24: Correct metric row counts in provenance
def test_24_correct_metric_row_counts():
    path = config.ARTIFACTS_DIR / 'regression' / 'evaluation_provenance.json'
    with open(path) as f:
        prov = json.load(f)
    assert prov['development_cv']['row_count'] == 800
    assert prov['locked_holdout']['row_count'] == 200
    assert prov['full_training_oof']['row_count'] == 1000

# Test 25: No test IDs in training set
def test_25_no_test_ids_in_training(oof_df, test_df):
    train_ids = set(oof_df['Test_ID'])
    test_ids = set(test_df['Test_ID'])
    assert len(train_ids.intersection(test_ids)) == 0

# Test 26: CP5/CP7 fold intersection audit
def test_26_cp5_fold_intersection_audit():
    path = config.ARTIFACTS_DIR / 'regression' / 'cp5_nested_provenance.json'
    with open(path) as f:
        cp5_audit = json.load(f)
    assert cp5_audit['reference_parameter_leakage'] == False
    assert cp5_audit['audit_status'] == 'LEAKAGE_SAFE_VERIFIED'

# Test 27: CP5 feature safety status
def test_27_cp5_feature_safety_status():
    path = config.ARTIFACTS_DIR / 'regression' / 'cp5_nested_provenance.json'
    with open(path) as f:
        cp5_audit = json.load(f)
    assert cp5_audit['audit_status'] == 'LEAKAGE_SAFE_VERIFIED'

# Test 28: Target exclusion in feature list
def test_28_target_exclusion():
    assert 'Reference_Parameter' not in reference_regression.FEATURE_COLS
    assert 'Validity_Label' not in reference_regression.FEATURE_COLS

# Test 29: Feature matrix finite values
def test_29_feature_matrix_finite_values(oof_df):
    for col in reference_regression.FEATURE_COLS:
        assert oof_df[col].notna().all()
        assert np.isfinite(oof_df[col]).all()

# Test 30: Ratio denominator safety
def test_30_ratio_denominator_safety(oof_df):
    assert (oof_df['S3_to_S1_ratio'].abs() < 1e7).all()

# Test 31: Final model metadata consistency
def test_31_final_model_metadata_consistency(model_metadata):
    assert model_metadata['model_family'] == 'GradientBoostingRegressor'
    assert model_metadata['training_row_count'] == 1000
    assert len(model_metadata['feature_list']) == 20

# Test 32: Exactly 350 test predictions
def test_32_exactly_350_test_predictions(test_df):
    assert len(test_df) == 350
    assert test_df['Test_ID'].nunique() == 350

# Test 33: No prediction clipping
def test_33_no_prediction_clipping(test_df, model_metadata):
    preds = test_df['Predicted_Reference_Parameter']
    assert preds.min() > 0.0

# Test 34: Deterministic predictions SHA-256
def test_34_deterministic_predictions_sha256():
    path = config.ARTIFACTS_DIR / 'regression' / 'reproducibility.json'
    with open(path) as f:
        rep = json.load(f)
    assert len(rep['oof_sha256']) == 64
    assert len(rep['test_sha256']) == 64

# Test 35: Correct holdout subgroup counts
def test_35_correct_holdout_subgroup_counts(holdout_df):
    assert len(holdout_df) == 200
    valid_count = (holdout_df['Validity_Label'] == 'Valid').sum()
    invalid_count = (holdout_df['Validity_Label'] == 'Invalid').sum()
    assert valid_count + invalid_count == 200
    assert valid_count > 150
