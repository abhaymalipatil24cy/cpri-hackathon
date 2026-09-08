import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from src import config

def test_imports():
    import sklearn
    import scipy
    import openpyxl
    assert sklearn.__version__ is not None
    assert scipy.__version__ is not None
    assert openpyxl.__version__ is not None

def test_repository_paths():
    assert config.ROOT_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.ARTIFACTS_DIR.exists()
    assert config.OUTPUTS_DIR.exists()

def test_data_files_exist():
    assert config.TRAIN_DATA_PATH.exists()
    assert config.TEST_DATA_PATH.exists()

def test_data_schemas():
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    assert len(df_train) == 1000
    assert len(df_test) == 350
    expected_train_cols = ['Test_ID', 'Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min', 'Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4', 'Reference_Parameter', 'Validity_Label']
    expected_test_cols = ['Test_ID', 'Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min', 'Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']
    assert list(df_train.columns) == expected_train_cols
    assert list(df_test.columns) == expected_test_cols
