"""Drift Intelligence: data drift, prediction drift, and data-quality drift.

Compares a reference distribution (training data / historical predictions)
against a current one (production traffic / live predictions).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import ks_2samp, wasserstein_distance, entropy


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Classic PSI. <0.1 = stable, 0.1-0.25 = moderate shift, >0.25 = significant shift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    exp_counts = np.where(exp_counts == 0, 1e-6, exp_counts)
    act_counts = np.where(act_counts == 0, 1e-6, act_counts)
    return float(np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts)))


def jensen_shannon_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
    lo, hi = min(expected.min(), actual.min()), max(expected.max(), actual.max())
    e_hist, edges = np.histogram(expected, bins=bins, range=(lo, hi), density=True)
    a_hist, _ = np.histogram(actual, bins=edges, density=True)
    e_hist = e_hist / (e_hist.sum() + 1e-12)
    a_hist = a_hist / (a_hist.sum() + 1e-12)
    m = 0.5 * (e_hist + a_hist)
    js = 0.5 * entropy(e_hist + 1e-12, m + 1e-12) + 0.5 * entropy(a_hist + 1e-12, m + 1e-12)
    return float(js)


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    cats = set(expected.unique()) | set(actual.unique())
    e_freq = expected.value_counts(normalize=True).reindex(cats, fill_value=1e-6)
    a_freq = actual.value_counts(normalize=True).reindex(cats, fill_value=1e-6)
    return float(np.sum((a_freq - e_freq) * np.log(a_freq / e_freq)))


@dataclass
class FeatureDriftResult:
    feature: str
    dtype: str
    psi: float
    ks_stat: float | None = None
    ks_pvalue: float | None = None
    js_divergence: float | None = None
    wasserstein: float | None = None
    severity: str = "stable"


@dataclass
class DriftReport:
    feature_drift: list = field(default_factory=list)
    n_drifted_features: int = 0
    overall_drift_pct: float = 0.0
    prediction_drift_psi: float | None = None
    schema_changes: list = field(default_factory=list)
    data_quality_drift: dict = field(default_factory=dict)


def _severity(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def detect_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> list[FeatureDriftResult]:
    results = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref, cur = reference_df[col].dropna(), current_df[col].dropna()
        if pd.api.types.is_numeric_dtype(ref):
            psi = population_stability_index(ref.values, cur.values)
            ks_stat, ks_p = ks_2samp(ref.values, cur.values)
            js = jensen_shannon_divergence(ref.values, cur.values)
            wd = wasserstein_distance(ref.values, cur.values)
            results.append(FeatureDriftResult(col, "numeric", round(psi, 4), round(ks_stat, 4), round(ks_p, 4), round(js, 4), round(wd, 4), _severity(psi)))
        else:
            psi = categorical_psi(ref, cur)
            results.append(FeatureDriftResult(col, "categorical", round(psi, 4), severity=_severity(psi)))
    return results


def detect_schema_changes(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> list[str]:
    changes = []
    ref_cols, cur_cols = set(reference_df.columns), set(current_df.columns)
    for c in cur_cols - ref_cols:
        changes.append(f"new column in production: {c}")
    for c in ref_cols - cur_cols:
        changes.append(f"missing column in production: {c}")
    for col in ref_cols & cur_cols:
        if str(reference_df[col].dtype) != str(current_df[col].dtype):
            changes.append(f"dtype changed for {col}: {reference_df[col].dtype} -> {current_df[col].dtype}")
    return changes


def detect_data_quality_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    return {
        "null_pct_ref": round(reference_df.isna().mean().mean() * 100, 2),
        "null_pct_current": round(current_df.isna().mean().mean() * 100, 2),
        "duplicate_pct_ref": round(reference_df.duplicated().mean() * 100, 2),
        "duplicate_pct_current": round(current_df.duplicated().mean() * 100, 2),
        "row_count_ref": len(reference_df),
        "row_count_current": len(current_df),
    }


def detect_prediction_drift(train_predictions: np.ndarray, prod_predictions: np.ndarray) -> float:
    """PSI between the model's predicted-probability distribution at train time vs. in production."""
    return round(population_stability_index(train_predictions, prod_predictions), 4)


def full_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame,
    feature_cols: list[str],
    train_predictions: np.ndarray | None = None,
    prod_predictions: np.ndarray | None = None,
) -> DriftReport:
    feat_drift = detect_feature_drift(reference_df, current_df, feature_cols)
    n_drifted = sum(1 for f in feat_drift if f.severity != "stable")
    report = DriftReport(
        feature_drift=[f.__dict__ for f in feat_drift],
        n_drifted_features=n_drifted,
        overall_drift_pct=round(100 * n_drifted / max(len(feat_drift), 1), 1),
        schema_changes=detect_schema_changes(reference_df, current_df),
        data_quality_drift=detect_data_quality_drift(reference_df, current_df),
    )
    if train_predictions is not None and prod_predictions is not None:
        report.prediction_drift_psi = detect_prediction_drift(train_predictions, prod_predictions)
    return report


if __name__ == "__main__":
    train = pd.read_csv("data/train_raw.csv")
    prod = pd.read_csv("data/production_traffic.csv")
    shared_cols = [c for c in train.columns if c in prod.columns and c not in ("customer_id", "email", "churned")]
    report = full_drift_report(train, prod, shared_cols)
    import json
    print(json.dumps(report.__dict__, indent=2, default=str))
