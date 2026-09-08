import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src import config
from src import eda

def get_file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8192):
            h.update(c)
    return h.hexdigest()

def test_1_eda_uses_exact_training_rows():
    df = eda.load_analysis_data()
    assert len(df) == 1000

def test_2_validity_counts():
    df = eda.load_analysis_data()
    counts = df["Validity_Label"].value_counts().to_dict()
    assert counts["Valid"] == 866
    assert counts["Invalid"] == 134

def test_3_candidate_rules_reproduce_independently():
    df = eda.load_analysis_data()
    rules = eda.evaluate_candidate_rules(df)
    
    assert rules["Rule_A (Sensor < 0)"]["number_flagged"] == 1
    assert rules["Rule_A (Sensor < 0)"]["invalid_flagged"] == 1
    assert rules["Rule_A (Sensor < 0)"]["precision"] == 1.0
    
    assert rules["Rule_B (Sensor == 25)"]["number_flagged"] == 6
    assert rules["Rule_B (Sensor == 25)"]["invalid_flagged"] == 6
    assert rules["Rule_B (Sensor == 25)"]["precision"] == 1.0
    
    assert rules["Rule_E (S3 < S1 - 5)"]["number_flagged"] == 11
    assert rules["Rule_E (S3 < S1 - 5)"]["invalid_flagged"] == 11
    assert rules["Rule_E (S3 < S1 - 5)"]["precision"] == 1.0

def test_4_sentinel_counts_reproduce_independently():
    df = eda.load_analysis_data()
    susp = eda.analyze_suspicious_values(df)
    assert susp["sentinel_25_count"] == 6
    assert susp["sentinel_25_p_invalid"] == 1.0

def test_5_negative_value_analysis_reproduces():
    df = eda.load_analysis_data()
    susp = eda.analyze_suspicious_values(df)
    negs = susp["negative_occurrences"]
    assert len(negs) == 1
    assert negs[0]["Test_ID"] == "TRN-0203"
    assert abs(negs[0]["value"] - (-0.2015)) < 1e-4

def test_6_zero_value_analysis_reproduces():
    df = eda.load_analysis_data()
    susp = eda.analyze_suspicious_values(df)
    zeros = susp["zero_occurrences"]
    assert len(zeros) == 3

def test_7_duplicate_group_count_remains_12():
    df = eda.load_analysis_data()
    dup_res = eda.analyze_duplicate_targets(df)
    assert dup_res["total_duplicate_groups"] == 12

def test_8_duplicate_rows_not_deleted():
    df = eda.load_analysis_data()
    dup_res = eda.analyze_duplicate_targets(df)
    assert dup_res["total_duplicate_rows"] == 24
    assert len(df) == 1000

def test_9_reference_parameter_stats_reproduce():
    df = eda.load_analysis_data()
    s_ref = df["Reference_Parameter"]
    assert abs(s_ref.mean() - 26.6925) < 1e-3
    assert abs(s_ref.median() - 23.0183) < 1e-3

def test_10_eda_does_not_modify_cleaned_input():
    cleaned_path = config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv"
    hash_before = get_file_hash(cleaned_path)
    eda.generate_eda_artifacts()
    hash_after = get_file_hash(cleaned_path)
    assert hash_before == hash_after