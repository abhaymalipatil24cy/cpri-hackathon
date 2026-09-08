import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

def load_analysis_data(data_path: Path = config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv") -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    assert len(df) == 1000, f"Expected 1000 training rows, got {len(df)}"
    assert "Validity_Label" in df.columns, "Validity_Label target missing"
    assert "Reference_Parameter" in df.columns, "Reference_Parameter target missing"
    return df

def summarize_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    num_cols = config.INPUT_FEATURES + [config.TARGET_REFERENCE]
    summary = {"overall": {}, "valid": {}, "invalid": {}}
    df_valid = df[df["Validity_Label"] == "Valid"]
    df_invalid = df[df["Validity_Label"] == "Invalid"]
    def calc_stats(sub_df: pd.DataFrame) -> Dict[str, Any]:
        res = {}
        for col in num_cols:
            if col not in sub_df.columns:
                continue
            s = pd.to_numeric(sub_df[col], errors="coerce")
            q1 = float(s.quantile(0.25)) if s.notna().any() else None
            q3 = float(s.quantile(0.75)) if s.notna().any() else None
            iqr = q3 - q1 if (q1 is not None and q3 is not None) else None
            res[col] = {
                "count": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "median": round(float(s.median()), 4) if s.notna().any() else None,
                "std": round(float(s.std()), 4) if s.notna().any() else None,
                "min": round(float(s.min()), 4) if s.notna().any() else None,
                "Q1": round(q1, 4) if q1 is not None else None,
                "Q3": round(q3, 4) if q3 is not None else None,
                "max": round(float(s.max()), 4) if s.notna().any() else None,
                "IQR": round(iqr, 4) if iqr is not None else None
            }
        return res
    summary["overall"] = calc_stats(df)
    summary["valid"] = calc_stats(df_valid)
    summary["invalid"] = calc_stats(df_invalid)
    return summary

def compare_valid_invalid(df: pd.DataFrame) -> Dict[str, Any]:
    df_valid = df[df["Validity_Label"] == "Valid"]
    df_invalid = df[df["Validity_Label"] == "Invalid"]
    num_cols = config.INPUT_FEATURES + [config.TARGET_REFERENCE]
    comparison = {
        "class_counts": {
            "Valid": len(df_valid),
            "Invalid": len(df_invalid),
            "total": len(df)
        },
        "feature_comparisons": {}
    }
    for col in num_cols:
        s_v = df_valid[col].dropna()
        s_i = df_invalid[col].dropna()
        v_mean, i_mean = s_v.mean(), s_i.mean()
        v_std, i_std = s_v.std(), s_i.std()
        pooled_std = np.sqrt(((len(s_v) - 1) * v_std**2 + (len(s_i) - 1) * i_std**2) / (len(s_v) + len(s_i) - 2))
        cohens_d = (i_mean - v_mean) / pooled_std if pooled_std > 0 else 0.0
        comparison["feature_comparisons"][col] = {
            "valid_mean": round(float(v_mean), 4),
            "invalid_mean": round(float(i_mean), 4),
            "valid_median": round(float(s_v.median()), 4),
            "invalid_median": round(float(s_i.median()), 4),
            "valid_std": round(float(v_std), 4),
            "invalid_std": round(float(i_std), 4),
            "valid_range": [round(float(s_v.min()), 4), round(float(s_v.max()), 4)],
            "invalid_range": [round(float(s_i.min()), 4), round(float(s_i.max()), 4)],
            "cohens_d": round(float(cohens_d), 4)
        }
    return comparison

def analyze_sensor_relationships(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    num_cols = config.INPUT_FEATURES + [config.TARGET_REFERENCE]
    df_num = df[num_cols].dropna()
    pearson_corr = df_num.corr(method="pearson").round(4)
    spearman_corr = df_num.corr(method="spearman").round(4)
    rel_summary = {
        "sensor_pair_correlations": {
            "S1_S2": float(pearson_corr.loc["Sensor_S1", "Sensor_S2"]),
            "S1_S3": float(pearson_corr.loc["Sensor_S1", "Sensor_S3"]),
            "S1_S4": float(pearson_corr.loc["Sensor_S1", "Sensor_S4"]),
            "S2_S3": float(pearson_corr.loc["Sensor_S2", "Sensor_S3"]),
            "S2_S4": float(pearson_corr.loc["Sensor_S2", "Sensor_S4"]),
            "S3_S4": float(pearson_corr.loc["Sensor_S3", "Sensor_S4"])
        },
        "spearman_sensor_pairs": {
            "S1_S2": float(spearman_corr.loc["Sensor_S1", "Sensor_S2"]),
            "S1_S3": float(spearman_corr.loc["Sensor_S1", "Sensor_S3"]),
            "S1_S4": float(spearman_corr.loc["Sensor_S1", "Sensor_S4"]),
            "S2_S3": float(spearman_corr.loc["Sensor_S2", "Sensor_S3"]),
            "S2_S4": float(spearman_corr.loc["Sensor_S2", "Sensor_S4"]),
            "S3_S4": float(spearman_corr.loc["Sensor_S3", "Sensor_S4"])
        }
    }
    return pearson_corr, rel_summary

def analyze_suspicious_values(df: pd.DataFrame) -> Dict[str, Any]:
    sentinel_details = []
    for col in ["Sensor_S1", "Sensor_S2", "Sensor_S3"]:
        mask = (df[col].sub(25.0).abs() < 1e-3).fillna(False)
        rows = df[mask]
        for idx, r in rows.iterrows():
            sentinel_details.append({
                "Test_ID": r["Test_ID"],
                "sensor": col,
                "value": float(r[col]),
                "Validity_Label": r["Validity_Label"],
                "Reference_Parameter": float(r["Reference_Parameter"]),
                "Voltage": float(r["Applied_Voltage_kV"]),
                "Current": float(r["Load_Current_A"]),
                "Temp": float(r["Ambient_Temperature_C"]),
                "Duration": float(r["Test_Duration_min"]),
                "S1": float(r["Sensor_S1"]) if pd.notna(r["Sensor_S1"]) else None,
                "S2": float(r["Sensor_S2"]) if pd.notna(r["Sensor_S2"]) else None,
                "S3": float(r["Sensor_S3"]) if pd.notna(r["Sensor_S3"]) else None,
                "S4": float(r["Sensor_S4"]) if pd.notna(r["Sensor_S4"]) else None
            })
    neg_rows = df[df["Sensor_S2"] < 0]
    neg_details = []
    for idx, r in neg_rows.iterrows():
        neg_details.append({
            "Test_ID": r["Test_ID"],
            "sensor": "Sensor_S2",
            "value": float(r["Sensor_S2"]),
            "Validity_Label": r["Validity_Label"],
            "Reference_Parameter": float(r["Reference_Parameter"]),
            "Voltage": float(r["Applied_Voltage_kV"]),
            "Current": float(r["Load_Current_A"]),
            "Temp": float(r["Ambient_Temperature_C"]),
            "Duration": float(r["Test_Duration_min"])
        })
    zero_details = []
    for col in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
        mask = (df[col] == 0).fillna(False)
        for idx, r in df[mask].iterrows():
            zero_details.append({
                "Test_ID": r["Test_ID"],
                "sensor": col,
                "value": 0.0,
                "Validity_Label": r["Validity_Label"],
                "Reference_Parameter": float(r["Reference_Parameter"]),
                "Voltage": float(r["Applied_Voltage_kV"]),
                "Current": float(r["Load_Current_A"])
            })
    return {
        "sentinel_25_occurrences": sentinel_details,
        "sentinel_25_count": len(sentinel_details),
        "sentinel_25_p_invalid": 1.0 if len(sentinel_details) > 0 and all(x["Validity_Label"] == "Invalid" for x in sentinel_details) else None,
        "negative_occurrences": neg_details,
        "zero_occurrences": zero_details
    }

def evaluate_candidate_rules(df: pd.DataFrame) -> Dict[str, Any]:
    total_valid = (df["Validity_Label"] == "Valid").sum()
    total_invalid = (df["Validity_Label"] == "Invalid").sum()
    rules = {}
    rule_a_mask = (df["negative_sensor_flag"] == 1)
    rule_b_mask = (df["sentinel_candidate_flag"] == 1)
    rule_c_mask = rule_b_mask.copy()
    df_raw = pd.read_csv(config.TRAIN_DATA_PATH)
    rule_d_mask = df_raw["Sensor_S1"].isna() & df_raw["Sensor_S2"].isna() & df_raw["Sensor_S3"].isna()
    rule_e_mask = (df["Sensor_S3"] < (df["Sensor_S1"] - 5.0))
    curr_q75 = df["Load_Current_A"].quantile(0.75)
    rule_f_mask = rule_e_mask & (df["Load_Current_A"] > curr_q75)
    rule_masks = {
        "Rule_A (Sensor < 0)": rule_a_mask,
        "Rule_B (Sensor == 25)": rule_b_mask,
        "Rule_C (Sensor in {18,25,55})": rule_c_mask,
        "Rule_D (All S1-S3 missing)": rule_d_mask,
        "Rule_E (S3 < S1 - 5)": rule_e_mask,
        "Rule_F (S3 < S1 - 5 & Current > Q75)": rule_f_mask
    }
    for name, mask in rule_masks.items():
        flagged_cnt = int(mask.sum())
        valid_flagged = int((mask & (df["Validity_Label"] == "Valid")).sum())
        invalid_flagged = int((mask & (df["Validity_Label"] == "Invalid")).sum())
        precision = round(float(invalid_flagged / flagged_cnt), 4) if flagged_cnt > 0 else 0.0
        recall = round(float(invalid_flagged / total_invalid), 4)
        f1 = round(float(2 * precision * recall / (precision + recall)), 4) if (precision + recall) > 0 else 0.0
        fpr = round(float(valid_flagged / total_valid), 4)
        rules[name] = {
            "number_flagged": flagged_cnt,
            "valid_flagged": valid_flagged,
            "invalid_flagged": invalid_flagged,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "valid_false_positive_rate": fpr
        }
    return rules

def analyze_duplicate_targets(df: pd.DataFrame) -> Dict[str, Any]:
    dup_df = df[df["observable_duplicate_group"] > 0].copy()
    groups_summary = []
    for g_id, group in dup_df.groupby("observable_duplicate_group"):
        refs = group["Reference_Parameter"].values
        groups_summary.append({
            "group_id": int(g_id),
            "row_count": len(group),
            "Test_IDs": list(group["Test_ID"]),
            "ref_values": [round(float(x), 4) for x in refs],
            "ref_min": round(float(np.min(refs)), 4),
            "ref_max": round(float(np.max(refs)), 4),
            "ref_range": round(float(np.ptp(refs)), 4),
            "ref_std": round(float(np.std(refs)), 4),
            "validity_labels": list(group["Validity_Label"].unique())
        })
    return {
        "total_duplicate_groups": len(groups_summary),
        "total_duplicate_rows": len(dup_df),
        "all_groups_invalid": all(g["validity_labels"] == ["Invalid"] for g in groups_summary),
        "avg_group_ref_range": round(float(np.mean([g["ref_range"] for g in groups_summary])), 4),
        "max_group_ref_range": round(float(np.max([g["ref_range"] for g in groups_summary])), 4),
        "groups": groups_summary
    }

def analyze_s4(df: pd.DataFrame) -> Dict[str, Any]:
    s4_obs = df[df["Sensor_S4"].notna()]
    s4_miss = df[df["Sensor_S4"].isna()]
    corrs = {}
    for col in config.INPUT_FEATURES + [config.TARGET_REFERENCE]:
        if col != "Sensor_S4" and col in df.columns:
            s_col = df[col].dropna()
            idx_common = df["Sensor_S4"].notna() & df[col].notna()
            r_p = df.loc[idx_common, "Sensor_S4"].corr(df.loc[idx_common, col], method="pearson")
            r_s = df.loc[idx_common, "Sensor_S4"].corr(df.loc[idx_common, col], method="spearman")
            corrs[col] = {"pearson": round(float(r_p), 4), "spearman": round(float(r_s), 4)}
    return {
        "missing_count": len(s4_miss),
        "missing_percent": round(float((len(s4_miss) / len(df)) * 100), 4),
        "validity_distribution_of_missing": {
            "Valid": int((s4_miss["Validity_Label"] == "Valid").sum()),
            "Invalid": int((s4_miss["Validity_Label"] == "Invalid").sum())
        },
        "correlations_with_other_variables": corrs
    }

def generate_eda_plots(df: pd.DataFrame, output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    raw_df = pd.read_csv(config.TRAIN_DATA_PATH)
    missing = raw_df.isna().sum()
    missing = missing[missing > 0]
    sns.barplot(x=missing.index, y=missing.values, ax=ax, palette="viridis")
    ax.set_title("Raw Training Data Missing Value Counts")
    ax.set_ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "missingness_summary.png", dpi=150)
    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sensors = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
    for idx, s in enumerate(sensors):
        ax = axes[idx // 2, idx % 2]
        sns.kdeplot(data=df, x=s, hue="Validity_Label", ax=ax, common_norm=False)
        ax.set_title(f"{s} Distribution by Validity Label")
    plt.tight_layout()
    plt.savefig(plots_dir / "sensor_distributions_by_validity.png", dpi=150)
    plt.close()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="Sensor_S1", y="Sensor_S3", hue="Validity_Label", style="Validity_Label", ax=ax, alpha=0.8)
    ax.set_title("Sensor_S1 vs Sensor_S3")
    plt.tight_layout()
    plt.savefig(plots_dir / "s1_vs_s3_validity.png", dpi=150)
    plt.close()

def generate_eda_artifacts(
    data_path: Path = config.ARTIFACTS_DIR / "cleaning" / "training_cleaned.csv",
    output_dir: Path = config.ARTIFACTS_DIR / "experiments" / "eda"
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_analysis_data(data_path)
    dist_summary = summarize_distributions(df)
    with open(output_dir / "distribution_summary.json", "w") as f:
        json.dump(dist_summary, f, indent=2)
    val_inv_comp = compare_valid_invalid(df)
    with open(output_dir / "valid_invalid_comparison.json", "w") as f:
        json.dump(val_inv_comp, f, indent=2)
    corr_matrix, corr_summary = analyze_sensor_relationships(df)
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
    susp_analysis = analyze_suspicious_values(df)
    with open(output_dir / "suspicious_value_analysis.json", "w") as f:
        json.dump(susp_analysis, f, indent=2)
    candidate_rules = evaluate_candidate_rules(df)
    with open(output_dir / "candidate_rule_evaluation.json", "w") as f:
        json.dump(candidate_rules, f, indent=2)
    dup_analysis = analyze_duplicate_targets(df)
    with open(output_dir / "duplicate_target_analysis.json", "w") as f:
        json.dump(dup_analysis, f, indent=2)
    s4_res = analyze_s4(df)
    with open(output_dir / "s4_analysis.json", "w") as f:
        json.dump(s4_res, f, indent=2)
    df["power_proxy"] = df["Load_Current_A"]**2 * df["Test_Duration_min"]
    phys_res = {
        "power_proxy_summary": {
            "mean": round(float(df["power_proxy"].mean()), 4),
            "median": round(float(df["power_proxy"].median()), 4),
            "valid_mean": round(float(df[df["Validity_Label"]=="Valid"]["power_proxy"].mean()), 4),
            "invalid_mean": round(float(df[df["Validity_Label"]=="Invalid"]["power_proxy"].mean()), 4)
        },
        "s3_minus_s1_summary": {
            "mean": round(float((df["Sensor_S3"] - df["Sensor_S1"]).mean()), 4),
            "valid_mean": round(float((df[df["Validity_Label"]=="Valid"]["Sensor_S3"] - df[df["Validity_Label"]=="Valid"]["Sensor_S1"]).mean()), 4),
            "invalid_mean": round(float((df[df["Validity_Label"]=="Invalid"]["Sensor_S3"] - df[df["Validity_Label"]=="Invalid"]["Sensor_S1"]).mean()), 4)
        }
    }
    with open(output_dir / "physical_relationships.json", "w") as f:
        json.dump(phys_res, f, indent=2)
    generate_eda_plots(df, output_dir)
    return {
        "valid_invalid_counts": val_inv_comp["class_counts"],
        "candidate_rules": candidate_rules,
        "duplicate_groups": dup_analysis["total_duplicate_groups"]
    }

if __name__ == "__main__":
    generate_eda_artifacts()