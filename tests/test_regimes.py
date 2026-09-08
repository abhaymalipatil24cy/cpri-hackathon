import hashlib
import json
import pickle
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src import config
from src import feature_engineering

def get_file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8192):
            h.update(c)
    return h.hexdigest()

def test_1_training_row_count_preserved():
    df_tr = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_training.csv")
    assert len(df_tr) == 1000

def test_2_test_row_count_preserved():
    df_te = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_test.csv")
    assert len(df_te) == 350

def test_3_no_duplicate_test_ids():
    df_tr = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_training.csv")
    df_te = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_test.csv")
    assert df_tr["Test_ID"].duplicated().sum() == 0
    assert df_te["Test_ID"].duplicated().sum() == 0

def test_4_scaler_fitted_on_training_only():
    with open(config.ARTIFACTS_DIR / "regimes" / "regime_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert scaler.n_samples_seen_ == 1000

def test_5_kmeans_fitted_on_training_only():
    with open(config.ARTIFACTS_DIR / "regimes" / "regime_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    assert kmeans.n_features_in_ == 4
    assert len(kmeans.cluster_centers_) == 4

def test_6_test_assignments_reproduce_exact():
    with open(config.ARTIFACTS_DIR / "regimes" / "regime_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(config.ARTIFACTS_DIR / "regimes" / "regime_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
        
    df_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    X_scaled = scaler.transform(df_te[config.OPERATING_VARIABLES])
    expected_preds = kmeans.predict(X_scaled)
    
    df_assign_te = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_test.csv")
    np.testing.assert_array_equal(df_assign_te["regime_id"].values, expected_preds)

def test_7_all_training_rows_assigned():
    df_tr = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_training.csv")
    assert df_tr["regime_id"].isna().sum() == 0

def test_8_all_test_rows_assigned():
    df_te = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_test.csv")
    assert df_te["regime_id"].isna().sum() == 0

def test_9_no_missing_regime_ids():
    df_tr = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_training.csv")
    unique_ids = set(df_tr["regime_id"].unique())
    assert unique_ids == {0, 1, 2, 3}

def test_10_selected_k_is_evaluated_candidate():
    with open(config.ARTIFACTS_DIR / "regimes" / "regime_metrics.json") as f:
        metrics = json.load(f)
    assert metrics["selected_k"] in [4, 5, 6]
    assert f"k={metrics['selected_k']}" in metrics["candidate_evaluations"]

def test_11_regime_metrics_reproduce_independently():
    df_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    evals = feature_engineering.evaluate_regime_candidates(df_tr)
    assert evals["k=4"]["silhouette_score"] == 0.1919
    assert evals["k=4"]["minimum_cluster_size"] == 240

def test_12_training_cluster_sizes_sum_to_1000():
    df_tr = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_training.csv")
    sizes = df_tr["regime_id"].value_counts()
    assert sizes.sum() == 1000

def test_13_test_cluster_sizes_sum_to_350():
    df_te = pd.read_csv(config.ARTIFACTS_DIR / "regimes" / "regime_assignments_test.csv")
    sizes = df_te["regime_id"].value_counts()
    assert sizes.sum() == 350

def test_14_input_hashes_unchanged():
    clean_tr_path = config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv"
    hash_before = get_file_hash(clean_tr_path)
    feature_engineering.run_full_regime_pipeline()
    hash_after = get_file_hash(clean_tr_path)
    assert hash_before == hash_after