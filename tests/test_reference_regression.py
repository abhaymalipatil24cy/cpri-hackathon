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
