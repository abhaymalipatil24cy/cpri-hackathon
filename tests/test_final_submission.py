# Tests for Checkpoint CP9 - Final Submission Output + Executive Summary
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

from src import config

@pytest.fixture
def raw_test_df():
    return pd.read_csv(config.TEST_DATA_PATH)

@pytest.fixture
def raw_train_df():
    return pd.read_csv(config.TRAIN_DATA_PATH)

@pytest.fixture
def submission_df():
    sub_path = config.OUTPUTS_DIR / 'TeamName.csv'
    return pd.read_csv(sub_path)

@pytest.fixture
def audit_df():
    audit_path = config.ARTIFACTS_DIR / 'final' / 'final_test_audit.csv'
    return pd.read_csv(audit_path)

@pytest.fixture
def freeze_manifest():
    manifest_path = config.ARTIFACTS_DIR / 'final' / 'cp9_freeze_manifest.json'
    with open(manifest_path) as f:
        return json.load(f)

@pytest.fixture
def quality_report():
    report_path = config.ARTIFACTS_DIR / 'final' / 'final_quality_report.json'
    with open(report_path) as f:
        return json.load(f)

# Test 1: Submission file exists
def test_1_submission_file_exists():
    sub_path = config.OUTPUTS_DIR / 'TeamName.csv'
    assert sub_path.exists()

# Test 2: Exact row count = 350
def test_2_exact_350_rows(submission_df):
    assert len(submission_df) == 350

# Test 3: Exact required columns
def test_3_exact_required_columns(submission_df):
    expected_cols = ['Test_ID', 'Validity_Label', 'Reference_Parameter']
    assert list(submission_df.columns) == expected_cols

# Test 4: Exact column ordering
def test_4_exact_column_ordering(submission_df):
    assert list(submission_df.columns)[0] == 'Test_ID'
    assert list(submission_df.columns)[1] == 'Validity_Label'
    assert list(submission_df.columns)[2] == 'Reference_Parameter'

# Test 5: Test_ID set equality
def test_5_test_id_set_equality(submission_df, raw_test_df):
    assert set(submission_df['Test_ID']) == set(raw_test_df['Test_ID'])

# Test 6: Test_ID ordering equality
def test_6_test_id_ordering_equality(submission_df, raw_test_df):
    assert (submission_df['Test_ID'].values == raw_test_df['Test_ID'].values).all()

# Test 7: No duplicate Test_IDs
def test_7_no_duplicate_test_ids(submission_df):
    assert submission_df['Test_ID'].is_unique

# Test 8: No NaN values in submission
def test_8_no_nan_values(submission_df):
    assert submission_df.notna().all().all()

# Test 9: Finite regression predictions
def test_9_finite_regression_predictions(submission_df):
    preds = submission_df['Reference_Parameter']
    assert np.isfinite(preds).all()

# Test 10: Valid classification values
def test_10_valid_classification_values(submission_df):
    labels = set(submission_df['Validity_Label'].unique())
    assert labels.issubset({'Valid', 'Invalid'})

# Test 11: No training IDs in submission
def test_11_no_training_ids_in_submission(submission_df, raw_train_df):
    train_ids = set(raw_train_df['Test_ID'])
    sub_ids = set(submission_df['Test_ID'])
    assert len(train_ids.intersection(sub_ids)) == 0

# Test 12: Audit/submission join completeness
def test_12_audit_submission_join_completeness(submission_df, audit_df):
    assert len(submission_df.merge(audit_df, on='Test_ID')) == 350

# Test 13: Audit/submission prediction equality
def test_13_audit_submission_prediction_equality(submission_df, audit_df):
    merged = submission_df.merge(audit_df, on='Test_ID', suffixes=('_sub', '_audit'))
    assert (merged['Validity_Label_sub'].values == merged['Validity_Label_audit'].values).all()
    assert np.allclose(merged['Reference_Parameter_sub'].values, merged['Reference_Parameter_audit'].values)

# Test 14: Freeze manifest exists
def test_14_freeze_manifest_exists():
    path = config.ARTIFACTS_DIR / 'final' / 'cp9_freeze_manifest.json'
    assert path.exists()

# Test 15: Model hashes match CP8 frozen manifest
def test_15_model_hashes_match_cp8_manifest(freeze_manifest):
    cp8_path = config.ARTIFACTS_DIR / 'uncertainty' / 'frozen_model_manifest.json'
    with open(cp8_path) as f:
        cp8_m = json.load(f)
    assert freeze_manifest['cp6_validity_classifier']['sha256'] == cp8_m['cp6_validity_model']['sha256']
    assert freeze_manifest['cp7_regression_model']['sha256'] == cp8_m['cp7_regression_model']['sha256']

# Test 16: Decision threshold = 0.236
def test_16_decision_threshold_equals_0_236(freeze_manifest):
    assert freeze_manifest['cp6_validity_classifier']['decision_threshold'] == 0.236

# Test 17: Final model hyperparameters match CP8
def test_17_final_hyperparameters_match(freeze_manifest):
    assert freeze_manifest['cp7_regression_model']['model_family'] == 'GradientBoostingRegressor'
    assert freeze_manifest['cp7_regression_model']['hyperparameters']['n_estimators'] == 100

# Test 18: Random seed = 42
def test_18_random_seed_equals_42(freeze_manifest):
    assert freeze_manifest['cp6_validity_classifier']['random_seed'] == 42
    assert freeze_manifest['cp7_regression_model']['random_seed'] == 42

# Test 19: Hashes are reproducible and recorded
def test_19_hashes_reproducible():
    path = config.ARTIFACTS_DIR / 'final' / 'final_hashes.json'
    with open(path) as f:
        h = json.load(f)
    assert 'TeamName.csv_sha256' in h
    assert len(h['TeamName.csv_sha256']) == 64

# Test 20: No accidental index column in submission
def test_20_no_accidental_index_column():
    sub_path = config.OUTPUTS_DIR / 'TeamName.csv'
    with open(sub_path) as f:
        first_line = f.readline().strip()
    assert first_line == 'Test_ID,Validity_Label,Reference_Parameter'

# Test 21: No debug columns in submission
def test_21_no_debug_columns(submission_df):
    assert len(submission_df.columns) == 3

# Test 22: Final quality report is internally consistent
def test_22_final_quality_report_internally_consistent(quality_report):
    assert quality_report['overall_status'] == 'PASS'
    assert quality_report['checks']['submission_exact_350_rows'] == True
    assert quality_report['checks']['submission_exact_schema'] == True
    assert quality_report['checks']['no_training_id_leakage'] == True
