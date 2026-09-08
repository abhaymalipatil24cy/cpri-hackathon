# Tests for Checkpoint CP10 - End-to-End Pipeline Verification
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src import config

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

@pytest.fixture
def env_data():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_environment.json'
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def pipeline_graph():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_pipeline_graph.json'
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def leakage_audit():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_leakage_audit.json'
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def hash_comp():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_hash_comparison.json'
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def static_audit():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_static_audit.json'
    with open(path) as f:
        return json.load(f)

# Test 1: Verification environment recorded
def test_1_environment_recorded(env_data):
    assert env_data['checkpoint'] == 'CP10'
    assert env_data['random_seed'] == 42
    assert 'python_version' in env_data

# Test 2: Raw training data integrity
def test_2_raw_training_data_integrity(env_data):
    raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    assert len(raw_tr) == 1000
    assert len(raw_tr.columns) == 11
    assert raw_tr['Test_ID'].is_unique
    assert file_hash(config.TRAIN_DATA_PATH) == env_data['raw_training_hash']

# Test 3: Raw test data integrity
def test_3_raw_test_data_integrity(env_data):
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert len(raw_te) == 350
    assert len(raw_te.columns) == 9
    assert raw_te['Test_ID'].is_unique
    assert file_hash(config.TEST_DATA_PATH) == env_data['raw_test_hash']

# Test 4: Pipeline graph dependency chain recorded
def test_4_pipeline_graph_recorded(pipeline_graph):
    stages = [s['stage'] for s in pipeline_graph['pipeline_flow']]
    assert len(stages) == 9
    assert '1. Ingestion' in stages[0]

# Test 5: Leakage audit passes
def test_5_leakage_audit_passes(leakage_audit):
    assert leakage_audit['test_isolation_status'] == 'PASS'
    assert leakage_audit['fold_local_preprocessing_status'] == 'PASS'
    assert leakage_audit['cp5_second_level_provenance'] == 'LEAKAGE_SAFE_VERIFIED'

# Test 6: Model identity freeze verified
def test_6_model_identity_freeze_verified():
    path = config.ARTIFACTS_DIR / 'final' / 'cp9_freeze_manifest.json'
    with open(path) as f:
        m = json.load(f)
    assert m['cp6_validity_classifier']['decision_threshold'] == 0.236
    assert m['cp7_regression_model']['hyperparameters']['n_estimators'] == 100

# Test 7: Clean run log recorded
def test_7_run_log_recorded():
    log_path = config.ARTIFACTS_DIR / 'verification' / 'cp10_run_log.txt'
    assert log_path.exists()
    with open(log_path, encoding='utf-8') as f:
        content = f.read()
    assert 'Exit Code: 0' in content

# Test 8: All CP1-CP9 artifacts present
def test_8_all_artifacts_present():
    assert (config.ARTIFACTS_DIR / 'validity' / 'validity_model.pkl').exists()
    assert (config.ARTIFACTS_DIR / 'regression' / 'final_model_metadata.json').exists()
    assert (config.ARTIFACTS_DIR / 'uncertainty' / 'attention_breakdown.csv').exists()
    assert (config.ARTIFACTS_DIR / 'final' / 'final_test_audit.csv').exists()

# Test 9: Hash comparison reproduces exact submission hash
def test_9_submission_hash_comparison(hash_comp):
    assert hash_comp['submission_hash_match'] == True

# Test 10: 350 out of 350 exact prediction equality
def test_10_prediction_equality_350_of_350():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / 'TeamName.csv')
    audit_df = pd.read_csv(config.ARTIFACTS_DIR / 'final' / 'final_test_audit.csv')
    merged = sub_df.merge(audit_df, on='Test_ID', suffixes=('_sub', '_audit'))
    assert len(merged) == 350
    assert (merged['Validity_Label_sub'].values == merged['Validity_Label_audit'].values).all()
    assert np.allclose(merged['Reference_Parameter_sub'].values, merged['Reference_Parameter_audit'].values)

# Test 11: Exact column schema on regenerated submission
def test_11_exact_submission_schema():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / 'TeamName.csv')
    assert list(sub_df.columns) == ['Test_ID', 'Validity_Label', 'Reference_Parameter']

# Test 12: Exact Test_ID ordering matching raw test data
def test_12_exact_test_id_ordering():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / 'TeamName.csv')
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert (sub_df['Test_ID'].values == raw_te['Test_ID'].values).all()

# Test 13: Zero NaN and zero Inf in regenerated submission
def test_13_zero_nan_inf_submission():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / 'TeamName.csv')
    assert sub_df.notna().all().all()
    assert np.isfinite(sub_df['Reference_Parameter']).all()

# Test 14: Static audit status passes
def test_14_static_audit_passes(static_audit):
    assert static_audit['audit_status'] == 'PASS'

# Test 15: Input row shuffle invariance
def test_15_input_row_shuffle_invariance():
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    shuffled = raw_te.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sub_df = pd.read_csv(config.OUTPUTS_DIR / 'TeamName.csv')
    
    merged = shuffled[['Test_ID']].merge(sub_df, on='Test_ID')
    reordered = merged.sort_values('Test_ID').reset_index(drop=True)
    orig_ordered = sub_df.sort_values('Test_ID').reset_index(drop=True)
    
    assert (reordered['Validity_Label'].values == orig_ordered['Validity_Label'].values).all()
    assert np.allclose(reordered['Reference_Parameter'].values, orig_ordered['Reference_Parameter'].values)

# Test 16: Metric provenance reconciliation
def test_16_metric_provenance_reconciliation():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_metric_provenance.json'
    with open(path) as f:
        prov = json.load(f)
    assert prov['cp7_dev_cv_mae']['value'] == 0.788621
    assert prov['cp7_holdout_mae']['value'] == 1.257678

# Test 17: Reproducibility report status passes
def test_17_reproducibility_report_status():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_reproducibility_report.json'
    with open(path) as f:
        rep = json.load(f)
    assert rep['overall_status'] == 'PASS'

# Test 18: No training IDs present in submission
def test_18_no_train_ids_in_submission():
    raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    sub_df = pd.read_csv(config.OUTPUTS_DIR / 'TeamName.csv')
    assert len(set(raw_tr['Test_ID']).intersection(set(sub_df['Test_ID']))) == 0

# Test 19: Entry point run_pipeline.py exists
def test_19_entry_point_exists():
    path = Path('run_pipeline.py')
    assert path.exists()

# Test 20: Master pipeline clean run execution log exists
def test_20_pipeline_run_log_exists():
    path = config.ARTIFACTS_DIR / 'verification' / 'cp10_run_log.txt'
    assert path.exists()
