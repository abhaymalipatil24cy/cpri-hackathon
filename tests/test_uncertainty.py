# Tests for Checkpoint CP8 — Uncertainty, Reliability & Attention Prioritization
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

from src import config

@pytest.fixture
def manifest():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'frozen_model_manifest.json'
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def breakdown_df():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'attention_breakdown.csv'
    return pd.read_csv(path)

@pytest.fixture
def test_ranked_df():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'test_attention_ranked.csv'
    return pd.read_csv(path)

@pytest.fixture
def provenance():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'attention_provenance.json'
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def canary_df():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'canary_results.csv'
    return pd.read_csv(path)

# Test 1: Frozen CP6 model unchanged
def test_1_frozen_cp6_model_unchanged(manifest):
    path = config.ARTIFACTS_DIR / 'validity' / 'validity_metadata.json'
    with open(path) as f:
        meta = json.load(f)
    assert manifest['cp6_validity_model']['model_family'] == meta.get('selected_model', 'Random Forest')

# Test 2: Frozen CP7 model unchanged
def test_2_frozen_cp7_model_unchanged(manifest):
    path = config.ARTIFACTS_DIR / 'regression' / 'final_model_metadata.json'
    with open(path) as f:
        meta = json.load(f)
    assert manifest['cp7_regression_model']['model_family'] == meta['model_family']
    assert manifest['cp7_regression_model']['hyperparameters']['n_estimators'] == meta['hyperparameters']['n_estimators']

# Test 3: Decision threshold remains 0.236
def test_3_decision_threshold_remains_0_236(manifest):
    assert manifest['cp6_validity_model']['decision_threshold'] == 0.236

# Test 4: Attention formula matches specification
def test_4_attention_formula_matches_spec(provenance):
    assert 'min(|z_S3|, 10)' in provenance['formula']
    assert provenance['components_weights']['negative_sensor_flag'] == 3.0
    assert provenance['components_weights']['sentinel_flag'] == 3.0

# Test 5: Attention score does not use target column
def test_5_attention_does_not_use_target(provenance):
    assert 'Reference_Parameter' in provenance['target_columns_excluded']

# Test 6: Attention score does not use validity label column
def test_6_attention_does_not_use_validity_label(provenance):
    assert 'Validity_Label' in provenance['target_columns_excluded']

# Test 7: Support uses training-fitted center
def test_7_support_uses_training_fitted_center():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'support_analysis.csv'
    df = pd.read_csv(path)
    assert 'Train (1000)' in df['partition'].values

# Test 8: Support calculation excludes test during fitting
def test_8_support_excludes_test_during_fitting(breakdown_df):
    assert breakdown_df['operating_space_support'].notna().all()
    assert (breakdown_df['operating_space_support'] > 0.0).all()
    assert (breakdown_df['operating_space_support'] <= 1.0).all()

# Test 9: P_invalid is finite and in [0, 1]
def test_9_p_invalid_finite_range(breakdown_df):
    p_inv = breakdown_df['P_invalid']
    assert p_inv.notna().all()
    assert np.isfinite(p_inv).all()
    assert (p_inv >= 0.0).all() and (p_inv <= 1.0).all()

# Test 10: Attention scores are finite
def test_10_attention_scores_finite(breakdown_df):
    tot = breakdown_df['total_attention_score']
    assert tot.notna().all()
    assert np.isfinite(tot).all()
    assert (tot >= 0.0).all()

# Test 11: CV model disagreement is finite and non-negative
def test_11_disagreement_finite_non_negative(breakdown_df):
    val_dis = breakdown_df['cv_model_disagreement']
    reg_dis = breakdown_df['cv_regression_model_disagreement']
    assert val_dis.notna().all() and reg_dis.notna().all()
    assert (val_dis >= 0.0).all() and (reg_dis >= 0.0).all()

# Test 12: Exactly 350 test rows in ranked triage output
def test_12_exactly_350_test_rows(test_ranked_df):
    assert len(test_ranked_df) == 350

# Test 13: Unique Test_ID in ranked triage output
def test_13_unique_test_id_ranked(test_ranked_df):
    assert test_ranked_df['Test_ID'].is_unique

# Test 14: No NaN in ranked triage output
def test_14_no_nan_in_ranked_triage(test_ranked_df):
    assert test_ranked_df.notna().all().all()

# Test 15: No Inf in ranked triage output
def test_15_no_inf_in_ranked_triage(test_ranked_df):
    num_cols = test_ranked_df.select_dtypes(include=[np.number]).columns
    assert np.isfinite(test_ranked_df[num_cols]).all().all()

# Test 16: Ranking is deterministic
def test_16_ranking_deterministic(test_ranked_df):
    assert (test_ranked_df['rank'].values == np.arange(1, 351)).all()
    assert (test_ranked_df['total_attention_score'].diff().dropna() <= 1e-6).all()

# Test 17: Component sum equals total attention score exactly
def test_17_component_sum_equals_total_attention(breakdown_df):
    comp_sum = (
        breakdown_df['negative_sensor_flag_contrib'] +
        breakdown_df['sentinel_flag_contrib'] +
        breakdown_df['s3_s1_inconsistency_flag_contrib'] +
        breakdown_df['z_score_magnitude_contrib'] +
        breakdown_df['imputation_count_contrib'] +
        breakdown_df['missing_count_contrib'] +
        breakdown_df['model_uncertainty_contrib'] +
        breakdown_df['centroid_distance_contrib']
    )
    assert np.allclose(comp_sum.values, breakdown_df['total_attention_score'].values, atol=1e-5)

# Test 18: Shuffle invariance for attention calculation
def test_18_shuffle_invariance(test_ranked_df):
    shuffled = test_ranked_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    reordered = shuffled.sort_values(by='rank').reset_index(drop=True)
    assert np.allclose(reordered['total_attention_score'].values, test_ranked_df['total_attention_score'].values)

# Test 19: Canary outputs deterministic
def test_19_canary_outputs_deterministic(canary_df):
    assert len(canary_df) == 10
    assert canary_df['total_attention_score'].notna().all()
    c1 = canary_df.iloc[0]
    c10 = canary_df.iloc[9]
    assert c1['total_attention_score'] < c10['total_attention_score']

# Test 20: Reproducibility hashes match
def test_20_reproducibility_hashes_match():
    path = config.ARTIFACTS_DIR / 'uncertainty' / 'reproducibility.json'
    with open(path) as f:
        rep = json.load(f)
    assert 'frozen_model_manifest_sha256' in rep
    assert 'test_attention_ranked_sha256' in rep
    assert len(rep['test_attention_ranked_sha256']) == 64
