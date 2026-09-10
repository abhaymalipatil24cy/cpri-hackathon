# Focused Unit Tests for CP11.1 Forensic Reconciliation Final Cleanup Blockers
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
def cp7_mae_v2():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp7_mae_reconciliation_v2.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def cp5_hash_rec():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp5_hash_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def cp6_thresh_rec():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp6_threshold_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def cp6_final_rec():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp6_final_metric_reconciliation.json"
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def cp9_hash_rec():
    path = config.ARTIFACTS_DIR / "verification" / "cp11_cp9_hash_reconciliation.json"
    with open(path) as f:
        return json.load(f)

# Test 1: Blocker 1 - CP7 MAE Reconciliation V2 Artifact & Authority
def test_blocker_1_cp7_mae_v2_reconciliation(cp7_mae_v2):
    assert cp7_mae_v2["reconciliation_status"] == "RESOLVED"
    assert cp7_mae_v2["one_authoritative_value"] == 4.506471
    assert cp7_mae_v2["discrepancy_explanation"]["narrative_typo_value"] == 4.500962
    assert cp7_mae_v2["holdout_population"]["total_rows"] == 200
    assert cp7_mae_v2["holdout_population"]["invalid_rows_count"] == 27
    assert len(cp7_mae_v2["holdout_population"]["invalid_test_ids"]) == 27

    # Independent raw calculation check from holdout_predictions.csv
    holdout_path = config.ARTIFACTS_DIR / "regression" / "holdout_predictions.csv"
    holdout_df = pd.read_csv(holdout_path)
    invalid_sub = holdout_df[holdout_df["Validity_Label"] == "Invalid"]
    mae_raw = float(np.abs(invalid_sub["Reference_Parameter"] - invalid_sub["predicted_Reference_Parameter"]).mean())
    assert round(mae_raw, 6) == 4.506471

# Test 2: Blocker 2 - CP5 Model Hash Reconciliation
def test_blocker_2_cp5_hash_reconciliation(cp5_hash_rec):
    assert cp5_hash_rec["reconciliation_status"] == "RESOLVED"
    assert cp5_hash_rec["production_cp5_model_path"] == "artifacts/validity/s3_consistency_model.pkl"
    assert cp5_hash_rec["cp10_baseline_sha256"] == "81fa9d13bbd3eb23b9d714b8d127b51448cf433e5374b6111ab85a49b5c0f213"
    assert cp5_hash_rec["parameter_identical"] is True

    # Actual file on disk hash check (matches pristine CP10 baseline)
    cp5_path = config.ARTIFACTS_DIR / "validity" / "s3_consistency_model.pkl"
    assert file_hash(cp5_path) == "81fa9d13bbd3eb23b9d714b8d127b51448cf433e5374b6111ab85a49b5c0f213"

# Test 3: Blocker 3 - CP6 Threshold Lineage Reconciliation
def test_blocker_3_cp6_threshold_reconciliation(cp6_thresh_rec):
    assert cp6_thresh_rec["reconciliation_status"] == "RESOLVED"
    assert cp6_thresh_rec["frozen_operational_threshold"] == 0.236
    assert "provides evidence of leakage-controlled threshold selection" in cp6_thresh_rec["wording_compliance"]
    assert "proves leakage-free" not in cp6_thresh_rec["wording_compliance"]

# Test 4: CP6 Numerical Integrity & Confusion Matrix Population Sum Check
def test_cp6_confusion_matrix_population_sums(cp6_final_rec):
    assert cp6_final_rec["reconciliation_status"] == "RESOLVED"
    pops = cp6_final_rec["population_provenance"]
    for key, pop in pops.items():
        cm = pop["confusion_matrix"]
        N = pop["total_rows"]
        cm_sum = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        assert cm_sum == N, f"Population sum mismatch for {key}: {cm_sum} != {N}"
        assert pop["sum_check_passed"] is True

# Test 5: Development and Holdout ID Disjointness & Specific Provenance Check
def test_cp6_disjointness_and_holdout_provenance(cp6_final_rec):
    with open(config.ARTIFACTS_DIR / "regression" / "split_manifest.json") as f:
        split = json.load(f)

    dev_ids = set(split["dev_ids"])
    holdout_ids = set(split["holdout_ids"])

    assert len(dev_ids) == 800
    assert len(holdout_ids) == 200
    assert len(dev_ids.intersection(holdout_ids)) == 0, "Development and Holdout ID sets must be strictly disjoint!"

    # Verify holdout Test_ID discrepancy details in JSON
    details = cp6_final_rec["holdout_reconciliation_details"]
    assert "TRN-0837" in details["differing_test_ids"]

# Test 6: CP9 Final Audit Hash Reconciliation
def test_cp9_final_audit_hash_reconciliation(cp9_hash_rec):
    assert cp9_hash_rec["reconciliation_status"] == "RESOLVED"
    assert cp9_hash_rec["target_artifact"] == "artifacts/final/final_test_audit.csv"
    assert cp9_hash_rec["authoritative_frozen_production_hash"] == "a51877875a26f9f9ee3dcd64af30f1daca1de3834d4c35552e8a2d5bb6c281ae"
    
    # Disk check
    audit_path = config.ARTIFACTS_DIR / "final" / "final_test_audit.csv"
    assert file_hash(audit_path) == "a51877875a26f9f9ee3dcd64af30f1daca1de3834d4c35552e8a2d5bb6c281ae"

# Test 7: Production Immutability Verification of All Frozen Artifacts
def test_production_immutability_hashes():
    expected_hashes = {
        config.OUTPUTS_DIR / "TeamName.csv": "5586561015bec10b02ebf00a2491e42b9670ab6b8ce54aed89e0393e3dfaf67c",
        config.ARTIFACTS_DIR / "validity" / "s3_consistency_model.pkl": "81fa9d13bbd3eb23b9d714b8d127b51448cf433e5374b6111ab85a49b5c0f213",
        config.ARTIFACTS_DIR / "validity" / "validity_model.pkl": "e0aa655a3b19d587a1fa0901ee7a600a6eba3e13171e65fadc60401703f569b1",
        config.ARTIFACTS_DIR / "regression" / "final_model_metadata.json": "4af06df714079e8337bbb1c3c0c7fed386edfb37026ffb26e96da529d71edbf7",
        config.ARTIFACTS_DIR / "validity" / "attention_score_spec.json": "f26a8e89109898262ff2ba2df8bd415caa0bf4e40b22d2a3b5072c173af10923",
        config.ARTIFACTS_DIR / "final" / "final_test_audit.csv": "a51877875a26f9f9ee3dcd64af30f1daca1de3834d4c35552e8a2d5bb6c281ae",
    }
    for path, exp_hash in expected_hashes.items():
        assert path.exists(), f"Missing production artifact: {path}"
        assert file_hash(path) == exp_hash, f"Hash mismatch for {path}: got {file_hash(path)}, expected {exp_hash}"
