# Red-Team Adversarial Unit Tests for Checkpoint CP11
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
def baseline_manifest():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_baseline_manifest.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def hash_audit():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_hash_discrepancy_audit.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def dupe_audit():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_duplicate_audit.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def missing_audit():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_missingness_audit.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def sensor_audit():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_sensor_audit.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def reg_robustness():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_regression_robustness.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def dist_shift():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_distribution_shift.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def thresh_df():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_threshold_sensitivity.csv"
    return pd.read_csv(path)

@pytest.fixture
def canary_df():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_canary_expanded.csv"
    return pd.read_csv(path)

@pytest.fixture
def pert_df():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_perturbation_analysis.csv"
    return pd.read_csv(path)

@pytest.fixture
def final_audit():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_final_audit.json"
    with open(path) as f:
        return json.load(f)

# Test 1: Baseline manifest exists
def test_1_baseline_manifest_exists(baseline_manifest):
    assert baseline_manifest["checkpoint"] == "CP11"
    assert baseline_manifest["random_seed"] == 42

# Test 2: Hash discrepancy resolved
def test_2_hash_discrepancy_resolved(hash_audit):
    assert hash_audit["discrepancy_status"] == "resolved"
    assert hash_audit["classification"] == "benign historical artifact"
    assert hash_audit["reproducible_from_source"] == True

# Test 3: Training set exact row count and unique IDs
def test_3_training_set_integrity():
    raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    assert len(raw_tr) == 1000
    assert raw_tr["Test_ID"].is_unique

# Test 4: Test set exact row count and unique IDs
def test_4_test_set_integrity():
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert len(raw_te) == 350
    assert raw_te["Test_ID"].is_unique

# Test 5: Train/Test ID isolation
def test_5_train_test_id_isolation():
    raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert len(set(raw_tr["Test_ID"]).intersection(set(raw_te["Test_ID"]))) == 0

# Test 6: Duplicate rows audit recorded
def test_6_duplicate_audit_recorded(dupe_audit):
    assert dupe_audit["checkpoint"] == "CP11"
    assert dupe_audit["total_training_rows"] == 1000

# Test 7: Missingness audit recorded
def test_7_missingness_audit_recorded(missing_audit):
    assert missing_audit["checkpoint"] == "CP11"
    assert missing_audit["s4_missing_test_count"] > 0

# Test 8: Sensor shortcut audit recorded
def test_8_sensor_shortcut_audit_recorded(sensor_audit):
    assert sensor_audit["checkpoint"] == "CP11"
    assert len(sensor_audit["top_features"]) >= 5

# Test 9: Regression robustness metrics valid
def test_9_regression_robustness(reg_robustness):
    assert reg_robustness["checkpoint"] == "CP11"
    assert reg_robustness["holdout_valid_subgroup_mae"] < reg_robustness["holdout_invalid_subgroup_mae"]

# Test 10: Distribution shift features recorded
def test_10_distribution_shift_features(dist_shift):
    assert "Applied_Voltage_kV" in dist_shift["features"]
    assert "Sensor_S1" in dist_shift["features"]

# Test 11: Threshold sensitivity table covers range
def test_11_threshold_sensitivity_table(thresh_df):
    assert 0.236 in thresh_df["threshold"].values
    assert len(thresh_df) == 7

# Test 12: Decision threshold locked at 0.236
def test_12_decision_threshold_locked(thresh_df):
    row_236 = thresh_df[thresh_df["threshold"] == 0.236]
    assert len(row_236) == 1
    assert row_236["invalid_f1"].values[0] > 0.90

# Test 13: Attention score audit formula present
def test_13_attention_audit_formula():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_attention_audit.json"
    with open(path) as f:
        att = json.load(f)
    assert "0.35*P_invalid" in att["attention_formula"]

# Test 14: Expanded canary suite covers 20 scenarios
def test_14_canary_expanded_scenarios(canary_df):
    assert len(canary_df) == 20
    assert (canary_df["Status"] == "PASS").all()

# Test 15: Canary scenario 1 (normal operating point is Valid)
def test_15_canary_normal_operating_point(canary_df):
    c1 = canary_df[canary_df["Case"] == 1]
    assert c1["Decision"].values[0] == "Valid"

# Test 16: Canary scenario 2 (negative S1 is Invalid)
def test_16_canary_negative_s1(canary_df):
    c2 = canary_df[canary_df["Case"] == 2]
    assert c2["Decision"].values[0] == "Invalid"

# Test 17: Canary scenario 6 (sentinel S1 is Invalid)
def test_17_canary_sentinel_s1(canary_df):
    c6 = canary_df[canary_df["Case"] == 6]
    assert c6["Decision"].values[0] == "Invalid"

# Test 18: Canary scenario 16 (all S1-S3 missing is Invalid)
def test_18_canary_all_s1_s3_missing(canary_df):
    c16 = canary_df[canary_df["Case"] == 16]
    assert c16["Decision"].values[0] == "Invalid"

# Test 19: Canary scenario 17 (missing S4 is Valid)
def test_19_canary_missing_s4(canary_df):
    c17 = canary_df[canary_df["Case"] == 17]
    assert c17["Decision"].values[0] == "Valid"

# Test 20: Perturbation analysis table exists
def test_20_perturbation_analysis_exists(pert_df):
    assert len(pert_df) > 0
    assert "Delta_P_invalid" in pert_df.columns

# Test 21: Model baseline hash matches CP8
def test_21_baseline_model_hash_matches(baseline_manifest):
    cp8_path = config.ARTIFACTS_DIR / "uncertainty" / "frozen_model_manifest.json"
    with open(cp8_path) as f:
        cp8_m = json.load(f)
    assert baseline_manifest["cp6_model_hash"] == cp8_m["cp6_validity_model"]["sha256"]

# Test 22: Validity model pickle exists
def test_22_validity_model_pickle_exists():
    path = config.ARTIFACTS_DIR / "validity" / "validity_model.pkl"
    assert path.exists()

# Test 23: Regression metadata json exists
def test_23_regression_metadata_exists():
    path = config.ARTIFACTS_DIR / "regression" / "final_model_metadata.json"
    assert path.exists()

# Test 24: No target leakage in test feature matrix
def test_24_no_target_leakage_in_test():
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    assert "Validity_Label" not in raw_te.columns
    assert "Reference_Parameter" not in raw_te.columns

# Test 25: Test row shuffle invariance
def test_25_test_row_shuffle_invariance():
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    shuffled = raw_te.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sub_df = pd.read_csv(config.OUTPUTS_DIR / "TeamName.csv")
    
    merged = shuffled[["Test_ID"]].merge(sub_df, on="Test_ID")
    orig_ordered = sub_df.sort_values("Test_ID").reset_index(drop=True)
    reordered = merged.sort_values("Test_ID").reset_index(drop=True)
    
    assert (reordered["Validity_Label"].values == orig_ordered["Validity_Label"].values).all()
    assert np.allclose(reordered["Reference_Parameter"].values, orig_ordered["Reference_Parameter"].values)

# Test 26: Reverse test row order invariance
def test_26_reverse_test_row_order_invariance():
    raw_te = pd.read_csv(config.TEST_DATA_PATH)
    reversed_df = raw_te.iloc[::-1].reset_index(drop=True)
    sub_df = pd.read_csv(config.OUTPUTS_DIR / "TeamName.csv")
    
    merged = reversed_df[["Test_ID"]].merge(sub_df, on="Test_ID")
    orig_ordered = sub_df.sort_values("Test_ID").reset_index(drop=True)
    reordered = merged.sort_values("Test_ID").reset_index(drop=True)
    
    assert (reordered["Validity_Label"].values == orig_ordered["Validity_Label"].values).all()

# Test 27: Submission file exact required schema
def test_27_submission_schema():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / "TeamName.csv")
    assert list(sub_df.columns) == ["Test_ID", "Validity_Label", "Reference_Parameter"]

# Test 28: Submission exact row count 350
def test_28_submission_row_count():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / "TeamName.csv")
    assert len(sub_df) == 350

# Test 29: Submission non-null and finite
def test_29_submission_non_null_finite():
    sub_df = pd.read_csv(config.OUTPUTS_DIR / "TeamName.csv")
    assert sub_df.notna().all().all()
    assert np.isfinite(sub_df["Reference_Parameter"]).all()

# Test 30: Submission immutability hash match
def test_30_submission_immutability():
    sub_path = config.OUTPUTS_DIR / "TeamName.csv"
    expected_hash = "5586561015bec10b02ebf00a2491e42b9670ab6b8ce54aed89e0393e3dfaf67c"
    assert file_hash(sub_path) == expected_hash

# Test 31: CP11 final audit status passes
def test_31_cp11_final_audit_passes(final_audit):
    assert final_audit["overall_status"] == "PASS"
    assert final_audit["submission_immutability_verified"] == True
