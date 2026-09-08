import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src import config
from src import clean_impute

def get_file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8192):
            h.update(c)
    return h.hexdigest()

def test_1_raw_training_hash_unchanged():
    hash_before = get_file_hash(config.TRAIN_DATA_PATH)
    clean_impute.run_full_cleaning_pipeline()
    hash_after = get_file_hash(config.TRAIN_DATA_PATH)
    assert hash_before == hash_after

def test_2_raw_test_hash_unchanged():
    hash_before = get_file_hash(config.TEST_DATA_PATH)
    clean_impute.run_full_cleaning_pipeline()
    hash_after = get_file_hash(config.TEST_DATA_PATH)
    assert hash_before == hash_after

def test_3_row_count_preserved():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_raw_te = pd.read_csv(config.TEST_DATA_PATH)
    
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    df_clean_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    
    assert len(df_clean_tr) == len(df_raw_tr) == 1000
    assert len(df_clean_te) == len(df_raw_te) == 350

def test_4_test_id_coverage_preserved():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_raw_te = pd.read_csv(config.TEST_DATA_PATH)
    
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    df_clean_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    
    assert list(df_clean_tr["Test_ID"]) == list(df_raw_tr["Test_ID"])
    assert list(df_clean_te["Test_ID"]) == list(df_raw_te["Test_ID"])

def test_5_reference_parameter_unchanged():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    
    assert "Reference_Parameter" in df_clean_tr.columns
    np.testing.assert_array_almost_equal(
        df_clean_tr["Reference_Parameter"].values,
        df_raw_tr["Reference_Parameter"].values
    )

def test_6_validity_label_unchanged():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    
    assert "Validity_Label" in df_clean_tr.columns
    assert list(df_clean_tr["Validity_Label"]) == list(df_raw_tr["Validity_Label"])

def test_7_observed_sensor_values_unchanged():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    
    for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
        obs_mask = df_raw_tr[s].notna()
        np.testing.assert_array_almost_equal(
            df_clean_tr.loc[obs_mask, s].values,
            df_raw_tr.loc[obs_mask, s].values
        )

def test_8_imputation_counts_equal_original_missing():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_raw_te = pd.read_csv(config.TEST_DATA_PATH)
    
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    df_clean_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    
    for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3"]:
        raw_miss_tr = int(df_raw_tr[s].isna().sum())
        clean_imp_tr = int(df_clean_tr[f"{s}_imputed"].sum())
        assert raw_miss_tr == clean_imp_tr
        
        raw_miss_te = int(df_raw_te[s].isna().sum())
        clean_imp_te = int(df_clean_te[f"{s}_imputed"].sum())
        assert raw_miss_te == clean_imp_te
        
    assert int(df_clean_tr["Sensor_S4_imputed"].sum()) == 0
    assert int(df_clean_te["Sensor_S4_imputed"].sum()) == 0

def test_9_imputation_flags_correctly_identify_only_imputed():
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    
    for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3"]:
        flagged_as_imp = df_clean_tr[f"{s}_imputed"] == 1
        was_originally_missing = df_raw_tr[s].isna()
        assert flagged_as_imp.equals(was_originally_missing)

def test_10_no_target_leakage_in_imputation_pipeline():
    df_clean_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    assert "Reference_Parameter" not in df_clean_te.columns
    assert "Validity_Label" not in df_clean_te.columns
    
    with open(config.ARTIFACTS_DIR / "cleaning" / "imputation_metrics.json") as f:
        metrics = json.load(f)
    for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3"]:
        predictors = metrics[s]["predictors"]
        assert "Reference_Parameter" not in predictors
        assert "Validity_Label" not in predictors

def test_11_duplicate_observable_rows_retained():
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    assert len(df_clean_tr) == 1000
    assert int(df_clean_tr["observable_duplicate_flag"].sum()) == 24
    assert int(df_clean_tr["observable_duplicate_group"].max()) == 12

def test_12_negative_values_preserved():
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    df_clean_te = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "test_cleaned.csv")
    
    neg_tr = df_clean_tr[df_clean_tr["Sensor_S2"] < 0]
    assert len(neg_tr) == 1
    assert abs(neg_tr["Sensor_S2"].values[0] - (-0.2015)) < 1e-4
    assert neg_tr["negative_sensor_flag"].values[0] == 1
    
    neg_te = df_clean_te[df_clean_te["Sensor_S2"] < 0]
    assert len(neg_te) == 1
    assert abs(neg_te["Sensor_S2"].values[0] - (-1.5509)) < 1e-4
    assert neg_te["negative_sensor_flag"].values[0] == 1

def test_13_sentinel_values_preserved():
    df_clean_tr = pd.read_csv(config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv")
    sentinel_rows = df_clean_tr[df_clean_tr["sentinel_candidate_flag"] == 1]
    assert len(sentinel_rows) > 0
    df_raw_tr = pd.read_csv(config.TRAIN_DATA_PATH)
    for idx in sentinel_rows.index:
        raw_vals = [df_raw_tr.loc[idx, s] for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3"] if pd.notna(df_raw_tr.loc[idx, s])]
        has_25 = any(abs(v - 25.0) < 1e-3 for v in raw_vals)
        assert has_25