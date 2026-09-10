import pytest
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from src.cp12_1_verifier import parse_pdf_training, parse_pdf_test, sha256_file, PDF_PATH

def test_cp12_1_pdf_file_exists():
    assert Path(PDF_PATH).exists(), f"PDF dataset file not found at {PDF_PATH}"

def test_cp12_1_training_dataset_exact_match():
    df_pdf_trn = parse_pdf_training(PDF_PATH)
    df_csv_trn = pd.read_csv('data/training_data.csv')
    
    assert len(df_pdf_trn) == 1000, f"Expected 1000 PDF training rows, got {len(df_pdf_trn)}"
    assert len(df_csv_trn) == 1000, f"Expected 1000 CSV training rows, got {len(df_csv_trn)}"
    assert set(df_pdf_trn['Test_ID']) == set(df_csv_trn['Test_ID']), "Training Test_IDs mismatch"
    
    # Check labels count
    val_counts = df_csv_trn['Validity_Label'].value_counts().to_dict()
    assert val_counts['Valid'] == 866, f"Expected 866 Valid labels, got {val_counts.get('Valid')}"
    assert val_counts['Invalid'] == 134, f"Expected 134 Invalid labels, got {val_counts.get('Invalid')}"

def test_cp12_1_test_dataset_exact_match():
    df_pdf_tst = parse_pdf_test(PDF_PATH)
    df_csv_tst = pd.read_csv('data/test_data.csv')
    
    assert len(df_pdf_tst) == 350, f"Expected 350 PDF test rows, got {len(df_pdf_tst)}"
    assert len(df_csv_tst) == 350, f"Expected 350 CSV test rows, got {len(df_csv_tst)}"
    assert set(df_pdf_tst['Test_ID']) == set(df_csv_tst['Test_ID']), "Test Test_IDs mismatch"

def test_cp12_1_submission_schema_and_hash():
    df_sub = pd.read_csv('outputs/TeamName.csv')
    assert len(df_sub) == 350, f"Submission must have 350 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ['Test_ID', 'Validity_Label', 'Reference_Parameter'], f"Invalid submission columns: {list(df_sub.columns)}"
    assert df_sub.isnull().sum().sum() == 0, "Submission contains nulls"
    assert set(df_sub['Validity_Label']).issubset({'Valid', 'Invalid'}), "Invalid Validity_Label values"
    
    sub_hash = sha256_file('outputs/TeamName.csv')
    expected_sub_hash = '5586561015bec10b02ebf00a2491e42b9670ab6b8ce54aed89e0393e3dfaf67c'
    assert sub_hash == expected_sub_hash, f"Submission hash mismatch: got {sub_hash}, expected {expected_sub_hash}"

def test_cp12_1_frozen_model_hashes():
    expected_hashes = {
        'artifacts/validity/s3_consistency_model.pkl': '81fa9d13bbd3eb23b9d714b8d127b51448cf433e5374b6111ab85a49b5c0f213',
        'artifacts/validity/validity_model.pkl': 'e0aa655a3b19d587a1fa0901ee7a600a6eba3e13171e65fadc60401703f569b1',
        'artifacts/regression/final_model_metadata.json': '4af06df714079e8337bbb1c3c0c7fed386edfb37026ffb26e96da529d71edbf7',
        'artifacts/validity/attention_score_spec.json': 'f26a8e89109898262ff2ba2df8bd415caa0bf4e40b22d2a3b5072c173af10923'
    }
    for file_path, expected_hash in expected_hashes.items():
        actual_hash = sha256_file(file_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {file_path}: got {actual_hash}, expected {expected_hash}"
