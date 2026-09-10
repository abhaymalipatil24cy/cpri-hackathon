# Forensic Reconciliation Unit Tests for Checkpoint CP11.1
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src import config

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

@pytest.fixture
def att_reconcil():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_attention_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def cp6_reconcil():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp6_metric_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def cp7_reconcil():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp7_metric_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def s4_reconcil():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_s4_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def s4_imp_audit():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_s4_imputation_audit.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def dupe_reconcil():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_duplicate_reconciliation.json"
    with open(path) as f:
        return json.load(f)

# Test 1: Frozen attention formula reaffirmed
def test_1_attention_formula_reaffirmed(att_reconcil):
    assert att_reconcil["reconciliation_status"] == "RESOLVED"
    assert "3.0 * flag_negative" in att_reconcil["authoritative_production_formula"]

# Test 2: Production attention outputs present
def test_2_production_attention_outputs():
    audit_df = pd.read_csv(config.ARTIFACTS_DIR / "final" / "final_test_audit.csv")
    assert "total_attention_score" in audit_df.columns
    assert (audit_df["total_attention_score"] > 0).all()

# Test 3: CP6 authoritative holdout Invalid F1 = 0.8889
def test_3_cp6_authoritative_holdout_f1(cp6_reconcil):
    assert cp6_reconcil["authoritative_holdout_metrics"]["Invalid_F1"] == pytest.approx(0.8889, abs=1e-3)
    assert cp6_reconcil["authoritative_holdout_metrics"]["Accuracy"] == 0.9700

# Test 4: CP7 authoritative holdout Invalid MAE = 4.500962
def test_4_cp7_authoritative_holdout_invalid_mae(cp7_reconcil):
    assert cp7_reconcil["authoritative_metrics"]["holdout_invalid_subgroup_mae"] == 4.500962

# Test 5: CP7 authoritative total holdout MAE = 1.257678
def test_5_cp7_authoritative_total_holdout_mae(cp7_reconcil):
    assert cp7_reconcil["authoritative_metrics"]["holdout_total_mae"] == 1.257678
    assert cp7_reconcil["authoritative_metrics"]["holdout_valid_subgroup_mae"] == pytest.approx(0.7506, abs=1e-3)

# Test 6: Raw test set S4 missing count is exactly 11 (3.14%)
def test_6_raw_test_s4_missing(s4_reconcil):
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert int(raw_te["Sensor_S4"].isna().sum()) == 11
    assert s4_reconcil["authoritative_raw_counts"]["test_s4_missing_count"] == 11

# Test 7: Raw training set S4 missing count is exactly 29 (2.90%)
def test_7_raw_train_s4_missing(s4_reconcil):
    raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    assert int(raw_tr["Sensor_S4"].isna().sum()) == 29
    assert s4_reconcil["authoritative_raw_counts"]["training_s4_missing_count"] == 29

# Test 8: S4 is strictly unimputed in clean artifacts
def test_8_s4_strictly_unimputed(s4_imp_audit):
    clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    clean_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    assert int(clean_tr["Sensor_S4_imputed"].sum()) == 0
    assert int(clean_te["Sensor_S4_imputed"].sum()) == 0
    assert s4_imp_audit["production_implementation"] == "Sensor_S4 is UNIMPUTED"

# Test 9: Duplicate counts are exactly 24 rows in 12 groups
def test_9_duplicate_counts_reconciled(dupe_reconcil):
    assert dupe_reconcil["authoritative_duplicate_count"] == "24 rows in 12 groups"
    assert dupe_reconcil["definitions"]["definition_a_operating_variables_only"]["rows_affected"] == 24

# Test 10: Model hashes match baseline
def test_10_production_model_hashes():
    val_model_path = config.ARTIFACTS_DIR / "validity" / "validity_model.pkl"
    assert file_hash(val_model_path) == "e0aa655a3b19d587a1fa0901ee7a600a6eba3e13171e65fadc60401703f569b1"

# Test 11: Submission immutability SHA-256 hash match
def test_11_submission_immutability():
    sub_path = config.OUTPUTS_DIR / "TeamName.csv"
    expected_hash = "5586561015bec10b02ebf00a2491e42b9670ab6b8ce54aed89e0393e3dfaf67c"
    assert file_hash(sub_path) == expected_hash

# Test 12: Decision threshold = 0.236 locked
def test_12_decision_threshold_locked():
    manifest_path = config.ARTIFACTS_DIR / "final" / "cp9_freeze_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)
    assert m["cp6_validity_classifier"]["decision_threshold"] == 0.236

# Test 13: Zero target leakage in test feature set
def test_13_zero_target_leakage_test():
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert "Validity_Label" not in raw_te.columns
    assert "Reference_Parameter" not in raw_te.columns
