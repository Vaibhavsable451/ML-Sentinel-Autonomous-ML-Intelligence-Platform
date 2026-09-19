"""Feature engineering: derived features + statistical feature selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain-driven derived features. Safe to call before the preprocessing pipeline."""
    df = df.copy()

    if {"outstanding_debt", "annual_income"}.issubset(df.columns):
        df["debt_to_income"] = df["outstanding_debt"] / df["annual_income"].replace(0, np.nan)

    if {"monthly_charges", "tenure_months"}.issubset(df.columns):
        df["lifetime_value_est"] = df["monthly_charges"] * df["tenure_months"]

    if "monthly_charges" in df.columns:
        df["log_monthly_charges"] = np.log1p(df["monthly_charges"].clip(lower=0))

    if "annual_income" in df.columns:
        df["log_annual_income"] = np.log1p(df["annual_income"].clip(lower=0))

    if "tenure_months" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[-0.1, 6, 12, 24, 48, np.inf],
            labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
        ).astype(str)

    if {"support_tickets_90d", "tenure_months"}.issubset(df.columns):
        df["tickets_per_tenure_month"] = df["support_tickets_90d"] / (df["tenure_months"].replace(0, np.nan) + 1)

    if "credit_score" in df.columns:
        df["credit_band"] = pd.cut(
            df["credit_score"],
            bins=[0, 580, 670, 740, 800, 900],
            labels=["poor", "fair", "good", "very_good", "excellent"],
        ).astype(str)

    return df


def correlation_report(df: pd.DataFrame, target: str, numeric_cols: list[str]) -> pd.Series:
    corrs = df[numeric_cols + [target]].corr(numeric_only=True)[target].drop(target)
    return corrs.sort_values(key=np.abs, ascending=False)


def mutual_information_report(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.Series:
    mi = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi, index=feature_names).sort_values(ascending=False)


def recursive_feature_elimination(X: np.ndarray, y: np.ndarray, feature_names: list[str], n_features: int = 15) -> list[str]:
    n_features = min(n_features, X.shape[1])
    estimator = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    return [f for f, keep in zip(feature_names, rfe.support_) if keep]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ml.preprocessing.pipeline import run_pipeline

    raw = pd.read_csv("data/train_raw.csv")
    enriched = add_engineered_features(raw)
    print("New columns added:", [c for c in enriched.columns if c not in raw.columns])

    result = run_pipeline(enriched, target="churned")
    mi = mutual_information_report(result.X_train, result.y_train, result.feature_names)
    print("\nTop 10 features by mutual information:")
    print(mi.head(10).round(4))

    rfe_selected = recursive_feature_elimination(result.X_train, result.y_train, result.feature_names, n_features=12)
    print("\nRFE-selected features:", rfe_selected)
