import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src import config

def compute_file_hash(filepath: Path) -> Dict[str, Any]:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    df = pd.read_csv(filepath)
    return {
        "filepath": str(filepath),
        "sha256": hasher.hexdigest(),
        "file_size_bytes": filepath.stat().st_size,
        "row_count": len(df),
        "column_count": len(df.columns)
    }

def validate_schema(df: pd.DataFrame, expected_cols: List[str]) -> Dict[str, Any]:
    actual_cols = list(df.columns)
    missing_cols = [c for c in expected_cols if c not in actual_cols]
    unexpected_cols = [c for c in actual_cols if c not in expected_cols]
    duplicate_col_names = [c for c in set(actual_cols) if actual_cols.count(c) > 1]
    is_order_matching = (actual_cols == expected_cols)
    return {
        "is_valid": len(missing_cols) == 0 and len(unexpected_cols) == 0 and len(duplicate_col_names) == 0,
        "missing_columns": missing_cols,
        "unexpected_columns": unexpected_cols,
        "duplicate_column_names": duplicate_col_names,
        "is_order_matching": is_order_matching,
        "actual_columns": actual_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

def audit_numeric_coercion(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    failures = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s_coerced = pd.to_numeric(df[col], errors="coerce")
        mask_failed = df[col].notna() & s_coerced.isna()
        failed_rows = df[mask_failed]
        for idx, row in failed_rows.iterrows():
            test_id = row["Test_ID"] if "Test_ID" in df.columns else None
            failures.append({
                "row_index": int(idx),
                "Test_ID": test_id,
                "column": col,
                "original_value": str(row[col]),
                "coercion_result": "NaN"
            })
    failures_df = pd.DataFrame(failures)
    if failures_df.empty:
        failures_df = pd.DataFrame(columns=["row_index", "Test_ID", "column", "original_value", "coercion_result"])
    summary = {
        "total_coercion_failures": len(failures),
        "failed_columns": list(failures_df["column"].unique()) if len(failures) > 0 else []
    }
    return failures_df, summary

def audit_missingness_and_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    total_rows = len(df)
    results = {}
    for col in df.columns:
        s = df[col]
        missing_cnt = int(s.isna().sum())
        missing_pct = float((missing_cnt / total_rows) * 100)
        unique_cnt = int(s.nunique(dropna=True))
        col_info = {
            "dtype": str(s.dtype),
            "row_count": total_rows,
            "missing_count": missing_cnt,
            "missing_percent": round(missing_pct, 4),
            "unique_count": unique_cnt
        }
        if col in numeric_cols and missing_cnt < total_rows:
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            col_info.update({
                "min": float(s_num.min()),
                "max": float(s_num.max()),
                "mean": float(s_num.mean()),
                "median": float(s_num.median()),
                "std": float(s_num.std())
            })
        results[col] = col_info
    return results

def audit_negative_and_zero(df: pd.DataFrame, check_cols: List[str]) -> Dict[str, Any]:
    results = {}
    total_rows = len(df)
    for col in check_cols:
        if col not in df.columns:
            continue
        s_num = pd.to_numeric(df[col], errors="coerce").dropna()
        neg_cnt = int((s_num < 0).sum())
        zero_cnt = int((s_num == 0).sum())
        results[col] = {
            "negative_count": neg_cnt,
            "negative_percent": round(float((neg_cnt / total_rows) * 100), 4),
            "zero_count": zero_cnt,
            "zero_percent": round(float((zero_cnt / total_rows) * 100), 4)
        }
    return results

def audit_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dup_mask = df.duplicated(keep=False)
    dup_rows = df[dup_mask].copy()
    test_id_dup_cnt = 0
    if "Test_ID" in df.columns:
        test_id_dup_cnt = int(df["Test_ID"].duplicated().sum())
    summary = {
        "total_duplicate_rows": int(dup_mask.sum()),
        "duplicate_groups_count": int(df.duplicated().sum()),
        "duplicate_test_ids_count": test_id_dup_cnt
    }
    return dup_rows, summary

def audit_near_duplicates(df: pd.DataFrame, feature_cols: List[str], tol_rel: float = 1e-4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    avail_cols = [c for c in feature_cols if c in df.columns]
    df_num = df[avail_cols].copy()
    means = df_num.mean()
    stds = df_num.std().replace(0, 1)
    df_norm = (df_num - means) / stds
    conflicts = []
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(df_norm.fillna(0).values, metric="euclidean"))
    threshold = np.sqrt(len(avail_cols)) * tol_rel * 10
    n_rows = len(df)
    visited = set()
    group_id = 0
    for i in range(n_rows):
        if i in visited:
            continue
        neighbors = np.where(dist_matrix[i] <= threshold)[0]
        if len(neighbors) > 1:
            group_id += 1
            visited.update(neighbors)
            group_rows = df.iloc[neighbors].copy()
            has_conflict = False
            if "Validity_Label" in df.columns:
                unique_labels = group_rows["Validity_Label"].nunique()
                if unique_labels > 1:
                    has_conflict = True
            for idx, r in group_rows.iterrows():
                row_dict = r.to_dict()
                row_dict["row_index"] = int(idx)
                row_dict["near_dup_group_id"] = group_id
                row_dict["near_duplicate_conflict"] = 1 if has_conflict else 0
                if has_conflict:
                    conflicts.append(row_dict)
    conflicts_df = pd.DataFrame(conflicts)
    if conflicts_df.empty:
        cols = list(df.columns) + ["row_index", "near_dup_group_id", "near_duplicate_conflict"]
        conflicts_df = pd.DataFrame(columns=cols)
    summary = {
        "near_duplicate_criterion": f"Euclidean distance on standardized features <= {threshold:.6f}",
        "total_near_dup_groups": group_id,
        "conflicting_near_dup_rows": len(conflicts)
    }
    return conflicts_df, summary

def audit_suspicious_sentinels(df: pd.DataFrame, sentinel_candidates: List[float] = [18.0, 25.0, 55.0], sensor_cols: List[str] = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]) -> Dict[str, Any]:
    results = {}
    total_rows = len(df)
    for val in sentinel_candidates:
        val_key = str(val)
        results[val_key] = {}
        for sensor in sensor_cols:
            if sensor not in df.columns:
                continue
            s_val = df[sensor]
            mask = np.isclose(s_val.dropna(), val, atol=1e-3)
            matching_rows = df.loc[s_val.dropna()[mask].index]
            cnt = len(matching_rows)
            pct = round(float((cnt / total_rows) * 100), 4)
            uniq_ids = int(matching_rows["Test_ID"].nunique()) if "Test_ID" in df.columns else 0
            valid_cnt = int((matching_rows["Validity_Label"] == "Valid").sum()) if "Validity_Label" in df.columns else None
            invalid_cnt = int((matching_rows["Validity_Label"] == "Invalid").sum()) if "Validity_Label" in df.columns else None
            results[val_key][sensor] = {
                "count": cnt,
                "percentage": pct,
                "unique_test_ids": uniq_ids,
                "valid_count": valid_cnt,
                "invalid_count": invalid_cnt,
                "avg_applied_voltage": float(matching_rows["Applied_Voltage_kV"].mean()) if cnt > 0 else None,
                "avg_load_current": float(matching_rows["Load_Current_A"].mean()) if cnt > 0 else None,
                "avg_test_duration": float(matching_rows["Test_Duration_min"].mean()) if cnt > 0 else None
            }
    return results

def audit_distribution_and_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    results = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s_num = pd.to_numeric(df[col], errors="coerce").dropna()
        if s_num.empty:
            continue
        q1 = float(s_num.quantile(0.25))
        q3 = float(s_num.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        below_cnt = int((s_num < lower_bound).sum())
        above_cnt = int((s_num > upper_bound).sum())
        results[col] = {
            "min": float(s_num.min()),
            "max": float(s_num.max()),
            "mean": float(s_num.mean()),
            "median": float(s_num.median()),
            "std": float(s_num.std()),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_iqr_bound": lower_bound,
            "upper_iqr_bound": upper_bound,
            "count_below_lower_bound": below_cnt,
            "count_above_upper_bound": above_cnt
        }
    return results

def audit_labels_and_targets(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    label_audit = {}
    if "Validity_Label" in df.columns:
        s_lab = df["Validity_Label"]
        val_counts = s_lab.value_counts(dropna=False).to_dict()
        label_audit = {
            "unique_labels": list(s_lab.dropna().unique()),
            "counts": {str(k): int(v) for k, v in val_counts.items()},
            "missing_count": int(s_lab.isna().sum()),
            "missing_percent": round(float((s_lab.isna().sum() / len(df)) * 100), 4)
        }
    ref_audit = {}
    if "Reference_Parameter" in df.columns:
        s_ref = pd.to_numeric(df["Reference_Parameter"], errors="coerce")
        ref_audit = {
            "missing_count": int(s_ref.isna().sum()),
            "missing_percent": round(float((s_ref.isna().sum() / len(df)) * 100), 4),
            "min": float(s_ref.min()) if s_ref.notna().any() else None,
            "max": float(s_ref.max()) if s_ref.notna().any() else None,
            "mean": float(s_ref.mean()) if s_ref.notna().any() else None,
            "median": float(s_ref.median()) if s_ref.notna().any() else None,
            "std": float(s_ref.std()) if s_ref.notna().any() else None,
            "q25": float(s_ref.quantile(0.25)) if s_ref.notna().any() else None,
            "q75": float(s_ref.quantile(0.75)) if s_ref.notna().any() else None
        }
    return label_audit, ref_audit

def run_full_quality_audit(
    train_path: Path = config.TRAIN_DATA_PATH,
    test_path: Path = config.TEST_DATA_PATH,
    output_dir: Path = config.ARTIFACTS_DIR / "data_quality"
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_hash = compute_file_hash(train_path)
    test_hash = compute_file_hash(test_path)
    hashes_data = {"training": train_hash, "test": test_hash}
    with open(output_dir / "input_hashes.json", "w") as f:
        json.dump(hashes_data, f, indent=2)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    train_schema = validate_schema(df_train, ["Test_ID"] + config.INPUT_FEATURES + [config.TARGET_REFERENCE, config.TARGET_VALIDITY])
    test_schema = validate_schema(df_test, ["Test_ID"] + config.INPUT_FEATURES)
    train_failures, train_coercion_summary = audit_numeric_coercion(df_train, config.INPUT_FEATURES + [config.TARGET_REFERENCE])
    test_failures, test_coercion_summary = audit_numeric_coercion(df_test, config.INPUT_FEATURES)
    all_failures = pd.concat([train_failures.assign(dataset="training"), test_failures.assign(dataset="test")], ignore_index=True)
    all_failures.to_csv(output_dir / "numeric_coercion_failures.csv", index=False)
    train_missing_stats = audit_missingness_and_stats(df_train, config.INPUT_FEATURES + [config.TARGET_REFERENCE])
    test_missing_stats = audit_missingness_and_stats(df_test, config.INPUT_FEATURES)
    train_neg_zero = audit_negative_and_zero(df_train, config.INPUT_FEATURES)
    test_neg_zero = audit_negative_and_zero(df_test, config.INPUT_FEATURES)
    train_exact_dup_df, train_exact_dup_summary = audit_exact_duplicates(df_train)
    test_exact_dup_df, test_exact_dup_summary = audit_exact_duplicates(df_test)
    all_exact_dups = pd.concat([train_exact_dup_df.assign(dataset="training"), test_exact_dup_df.assign(dataset="test")], ignore_index=True)
    all_exact_dups.to_csv(output_dir / "exact_duplicates.csv", index=False)
    train_conflicts_df, train_near_dup_summary = audit_near_duplicates(df_train, config.INPUT_FEATURES)
    train_conflicts_df.to_csv(output_dir / "near_duplicate_conflicts.csv", index=False)
    train_sentinels = audit_suspicious_sentinels(df_train, config.SENTINEL_CANDIDATES, ["Sensor_S1", "Sensor_S2", "Sensor_S3"])
    test_sentinels = audit_suspicious_sentinels(df_test, config.SENTINEL_CANDIDATES, ["Sensor_S1", "Sensor_S2", "Sensor_S3"])
    sentinel_report = {"training": train_sentinels, "test": test_sentinels}
    with open(output_dir / "suspicious_values.json", "w") as f:
        json.dump(sentinel_report, f, indent=2)
    train_iqr = audit_distribution_and_iqr(df_train, config.INPUT_FEATURES)
    test_iqr = audit_distribution_and_iqr(df_test, config.INPUT_FEATURES)
    label_audit, ref_audit = audit_labels_and_targets(df_train)
    label_report = {"validity_label_audit": label_audit, "reference_parameter_audit": ref_audit}
    with open(output_dir / "label_audit.json", "w") as f:
        json.dump(label_report, f, indent=2)
    train_ids = set(df_train["Test_ID"].dropna()) if "Test_ID" in df_train.columns else set()
    test_ids = set(df_test["Test_ID"].dropna()) if "Test_ID" in df_test.columns else set()
    id_overlap = list(train_ids.intersection(test_ids))
    training_quality = {
        "dataset": "training",
        "file_hash": train_hash,
        "schema_validation": train_schema,
        "coercion_summary": train_coercion_summary,
        "missingness_stats": train_missing_stats,
        "negatives_and_zeros": train_neg_zero,
        "exact_duplicates": train_exact_dup_summary,
        "near_duplicates": train_near_dup_summary,
        "distributions_and_iqr": train_iqr,
        "validity_label_audit": label_audit,
        "reference_parameter_audit": ref_audit
    }
    with open(output_dir / "training_quality.json", "w") as f:
        json.dump(training_quality, f, indent=2)
    test_quality = {
        "dataset": "test",
        "file_hash": test_hash,
        "schema_validation": test_schema,
        "coercion_summary": test_coercion_summary,
        "missingness_stats": test_missing_stats,
        "negatives_and_zeros": test_neg_zero,
        "exact_duplicates": test_exact_dup_summary,
        "distributions_and_iqr": test_iqr,
        "test_id_overlap_with_train_count": len(id_overlap),
        "test_id_overlap_list": id_overlap
    }
    with open(output_dir / "test_quality.json", "w") as f:
        json.dump(test_quality, f, indent=2)
    return {
        "training_quality": training_quality,
        "test_quality": test_quality,
        "input_hashes": hashes_data
    }

if __name__ == "__main__":
    run_full_quality_audit()