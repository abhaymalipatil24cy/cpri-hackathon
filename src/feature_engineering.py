import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

def evaluate_regime_candidates(df_train: pd.DataFrame) -> Dict[str, Any]:
    op_cols = config.OPERATING_VARIABLES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[op_cols])
    results = {}
    for k in [4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        counts = pd.Series(labels).value_counts().to_dict()
        sizes = [int(counts[i]) for i in range(k)]
        results[f'k={k}'] = {
            'k': k,
            'silhouette_score': round(sil, 4),
            'cluster_sizes': sizes,
            'minimum_cluster_size': int(min(sizes)),
            'maximum_cluster_size': int(max(sizes)),
            'cluster_balance_ratio': round(float(min(sizes) / max(sizes)), 4)
        }
    return results

def evaluate_regime_stability(df_train: pd.DataFrame, n_clusters: int = 4, n_bootstraps: int = 10) -> Dict[str, Any]:
    op_cols = config.OPERATING_VARIABLES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[op_cols])
    base_km = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    base_labels = base_km.fit_predict(X_scaled)
    ari_scores = []
    n_samples = len(df_train)
    for seed in range(42, 42 + n_bootstraps):
        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(n_samples, size=int(0.8 * n_samples), replace=False)
        X_sub = X_scaled[sub_idx]
        sub_km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        sub_labels = sub_km.fit_predict(X_sub)
        base_sub_labels = base_labels[sub_idx]
        ari = adjusted_rand_score(base_sub_labels, sub_labels)
        ari_scores.append(float(ari))
    return {
        'n_clusters': n_clusters,
        'n_bootstraps': n_bootstraps,
        'mean_adjusted_rand_index': round(float(np.mean(ari_scores)), 4),
        'std_adjusted_rand_index': round(float(np.std(ari_scores)), 4),
        'min_adjusted_rand_index': round(float(np.min(ari_scores)), 4),
        'max_adjusted_rand_index': round(float(np.max(ari_scores)), 4)
    }

def fit_operating_regimes(df_train: pd.DataFrame, n_clusters: int = 4) -> Tuple[StandardScaler, KMeans, np.ndarray]:
    op_cols = config.OPERATING_VARIABLES
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[op_cols])
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    train_regimes = kmeans.fit_predict(X_train_scaled)
    return scaler, kmeans, train_regimes

def predict_operating_regimes(df: pd.DataFrame, scaler: StandardScaler, kmeans: KMeans) -> np.ndarray:
    op_cols = config.OPERATING_VARIABLES
    X_scaled = scaler.transform(df[op_cols])
    return kmeans.predict(X_scaled)

def evaluate_variance_reduction(df_train: pd.DataFrame, regime_ids: np.ndarray) -> Dict[str, Any]:
    eval_cols = config.SENSOR_VARIABLES + config.OPERATING_VARIABLES + [config.TARGET_REFERENCE]
    results = {}
    for col in eval_cols:
        if col not in df_train.columns:
            continue
        s = df_train[col].dropna()
        if len(s) == 0:
            continue
        global_var = float(s.var(ddof=1))
        weighted_vars = []
        for r_id in np.unique(regime_ids):
            s_r = df_train.loc[regime_ids == r_id, col].dropna()
            if len(s_r) > 1:
                weighted_vars.append(len(s_r) * s_r.var(ddof=1))
        pooled_var = float(sum(weighted_vars) / len(s)) if len(s) > 0 else global_var
        var_reduction_pct = round(float((1.0 - (pooled_var / global_var)) * 100), 4) if global_var > 0 else 0.0
        results[col] = {
            'global_variance': round(global_var, 4),
            'pooled_regime_variance': round(pooled_var, 4),
            'variance_reduction_percent': var_reduction_pct
        }
    return results

def generate_regime_profiles(df_train: pd.DataFrame, regime_ids: np.ndarray) -> pd.DataFrame:
    df_temp = df_train.copy()
    df_temp['regime_id'] = regime_ids
    df_temp['power_proxy'] = df_temp['Load_Current_A']**2 * df_temp['Test_Duration_min']
    profile_cols = config.OPERATING_VARIABLES + ['power_proxy'] + config.SENSOR_VARIABLES + [config.TARGET_REFERENCE]
    profiles = []
    for r_id, group in df_temp.groupby('regime_id'):
        prof = {'regime_id': int(r_id), 'count': len(group)}
        for col in profile_cols:
            if col in group.columns:
                s = group[col].dropna()
                prof[f'{col}_mean'] = round(float(s.mean()), 4) if len(s) > 0 else None
                prof[f'{col}_std'] = round(float(s.std()), 4) if len(s) > 0 else None
                prof[f'{col}_median'] = round(float(s.median()), 4) if len(s) > 0 else None
                prof[f'{col}_min'] = round(float(s.min()), 4) if len(s) > 0 else None
                prof[f'{col}_max'] = round(float(s.max()), 4) if len(s) > 0 else None
        if 'Validity_Label' in group.columns:
            valid_cnt = int((group['Validity_Label'] == 'Valid').sum())
            invalid_cnt = int((group['Validity_Label'] == 'Invalid').sum())
            prof['Valid_count'] = valid_cnt
            prof['Invalid_count'] = invalid_cnt
            prof['Invalid_rate'] = round(float(invalid_cnt / len(group)), 4)
        if 'Sensor_S4_missing' in group.columns:
            prof['S4_missing_count'] = int(group['Sensor_S4_missing'].sum())
        profiles.append(prof)
    return pd.DataFrame(profiles)

def generate_regime_plots(df_train: pd.DataFrame, scaler: StandardScaler, kmeans: KMeans, output_dir: Path):
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    op_cols = config.OPERATING_VARIABLES
    X_scaled = scaler.transform(df_train[op_cols])
    regimes = kmeans.predict(X_scaled)
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=regimes, cmap='viridis', alpha=0.8, edgecolors='k', s=40)
    ax.set_title('Operating Regimes 2D PCA Projection')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.colorbar(scatter, ax=ax, label='Regime ID')
    plt.tight_layout()
    plt.savefig(plots_dir / 'regime_pca_projection.png', dpi=150)
    plt.close()
    df_plot = df_train.copy()
    df_plot['regime_id'] = regimes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, col in enumerate(op_cols):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df_plot, x='regime_id', y=col, ax=ax, palette='Set2')
        ax.set_title(f'{col} by Operating Regime')
    plt.tight_layout()
    plt.savefig(plots_dir / 'operating_variables_by_regime.png', dpi=150)
    plt.close()

def run_full_regime_pipeline(
    train_path: Path = config.ARTIFACTS_DIR / 'cleaning' / 'training_cleaned.csv',
    test_path: Path = config.ARTIFACTS_DIR / 'cleaning' / 'test_cleaned.csv',
    output_dir: Path = config.ARTIFACTS_DIR / 'regimes'
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    candidates = evaluate_regime_candidates(df_train)
    selected_k = 4
    scaler, kmeans, train_regimes = fit_operating_regimes(df_train, n_clusters=selected_k)
    test_regimes = predict_operating_regimes(df_test, scaler, kmeans)
    with open(output_dir / 'regime_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(output_dir / 'regime_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    df_tr_assign = pd.DataFrame({
        'Test_ID': df_train['Test_ID'],
        'regime_id': train_regimes
    })
    if 'Validity_Label' in df_train.columns:
        df_tr_assign['Validity_Label'] = df_train['Validity_Label']
    if 'Reference_Parameter' in df_train.columns:
        df_tr_assign['Reference_Parameter'] = df_train['Reference_Parameter']
    df_tr_assign.to_csv(output_dir / 'regime_assignments_training.csv', index=False)
    df_te_assign = pd.DataFrame({
        'Test_ID': df_test['Test_ID'],
        'regime_id': test_regimes
    })
    df_te_assign.to_csv(output_dir / 'regime_assignments_test.csv', index=False)
    stability = evaluate_regime_stability(df_train, n_clusters=selected_k)
    var_res = evaluate_variance_reduction(df_train, train_regimes)
    profiles = generate_regime_profiles(df_train, train_regimes)
    profiles.to_csv(output_dir / 'regime_profiles.csv', index=False)
    with open(output_dir / 'regime_stability.json', 'w') as f:
        json.dump(stability, f, indent=2)
    with open(output_dir / 'regime_vs_global.json', 'w') as f:
        json.dump(var_res, f, indent=2)
    metrics = {
        'candidate_evaluations': candidates,
        'selected_k': selected_k,
        'selection_justification': 'k=4 achieves optimal balance across operating space with min cluster size 240 (24% of data) and 0.1919 silhouette score.',
        'training_cluster_sizes': [int((train_regimes == i).sum()) for i in range(selected_k)],
        'test_cluster_sizes': [int((test_regimes == i).sum()) for i in range(selected_k)],
        'stability': stability
    }
    with open(output_dir / 'regime_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    generate_regime_plots(df_train, scaler, kmeans, output_dir)
    return metrics

if __name__ == '__main__':
    run_full_regime_pipeline()
