import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src import config
from src import ingest

def test_1_train_row_count():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    assert len(df_train) == 1000

def test_2_test_row_count():
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    assert len(df_test) == 350

def test_3_train_schema():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    expected_cols = ['Test_ID'] + config.INPUT_FEATURES + [config.TARGET_REFERENCE, config.TARGET_VALIDITY]
    schema_res = ingest.validate_schema(df_train, expected_cols)
    assert schema_res['is_valid'] is True
    assert len(schema_res['missing_columns']) == 0
    assert len(schema_res['unexpected_columns']) == 0

def test_4_test_schema():
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    expected_cols = ['Test_ID'] + config.INPUT_FEATURES
    schema_res = ingest.validate_schema(df_test, expected_cols)
    assert schema_res['is_valid'] is True
    assert len(schema_res['missing_columns']) == 0
    assert len(schema_res['unexpected_columns']) == 0

def test_5_numeric_coercion_behavior():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    failures_df, summary = ingest.audit_numeric_coercion(df_train, config.INPUT_FEATURES + [config.TARGET_REFERENCE])
    assert summary['total_coercion_failures'] == 0
    assert len(failures_df) == 0

def test_6_missingness_counts():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    
    train_stats = ingest.audit_missingness_and_stats(df_train, config.INPUT_FEATURES + [config.TARGET_REFERENCE])
    test_stats = ingest.audit_missingness_and_stats(df_test, config.INPUT_FEATURES)
    
    assert train_stats['Sensor_S1']['missing_count'] == 6
    assert train_stats['Sensor_S2']['missing_count'] == 2
    assert train_stats['Sensor_S3']['missing_count'] == 7
    assert train_stats['Sensor_S4']['missing_count'] == 29
    assert train_stats['Reference_Parameter']['missing_count'] == 0
    assert train_stats['Validity_Label']['missing_count'] == 0
    
    assert test_stats['Sensor_S1']['missing_count'] == 1
    assert test_stats['Sensor_S2']['missing_count'] == 3
    assert test_stats['Sensor_S3']['missing_count'] == 2
    assert test_stats['Sensor_S4']['missing_count'] == 11

def test_7_duplicate_detection():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    dup_rows, summary = ingest.audit_exact_duplicates(df_train)
    assert summary['total_duplicate_rows'] == 0
    
    # Verify exact observable duplicate pairs in train data
    feature_cols = config.INPUT_FEATURES
    dups_obs = df_train[df_train.duplicated(subset=feature_cols, keep=False)]
    assert len(dups_obs) == 24

def test_8_sentinel_value_counts():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    sentinels = ingest.audit_suspicious_sentinels(df_train, [18.0, 25.0, 55.0], ['Sensor_S1', 'Sensor_S2', 'Sensor_S3'])
    
    assert sentinels['25.0']['Sensor_S1']['count'] == 1
    assert sentinels['25.0']['Sensor_S2']['count'] == 3
    assert sentinels['25.0']['Sensor_S3']['count'] == 2
    assert sentinels['18.0']['Sensor_S1']['count'] == 0
    assert sentinels['55.0']['Sensor_S1']['count'] == 0

def test_9_sha256_hashing():
    hash_res = ingest.compute_file_hash(config.TRAIN_DATA_PATH)
    assert 'sha256' in hash_res
    assert len(hash_res['sha256']) == 64
    assert hash_res['row_count'] == 1000
    assert hash_res['column_count'] == 11

def test_10_no_raw_data_mutation():
    hash_before = ingest.compute_file_hash(config.TRAIN_DATA_PATH)['sha256']
    ingest.run_full_quality_audit()
    hash_after = ingest.compute_file_hash(config.TRAIN_DATA_PATH)['sha256']
    assert hash_before == hash_after