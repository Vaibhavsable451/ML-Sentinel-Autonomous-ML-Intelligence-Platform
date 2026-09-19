"""Preprocessing pipeline that turns raw, messy data into model-ready arrays."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ml.preprocessing.data_quality import profile, DataQualityReport, _is_categorical


@dataclass
class PreprocessResult:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    transformer: ColumnTransformer
    quality_report: DataQualityReport
    dropped_columns: list[str]


DEFAULT_DROP_HINTS = ("customer_id",)  # always non-predictive identifiers


def build_preprocessor(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def run_pipeline(
    df: pd.DataFrame,
    target: str,
    id_like_cols: list[str] | None = None,
    rare_category_threshold: float = 0.01,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    drop_leakage: bool = True,
) -> PreprocessResult:
    id_like_cols = id_like_cols or list(DEFAULT_DROP_HINTS)
    df = df.copy()

    # 1. data-quality profile BEFORE any cleaning (so the report reflects raw reality)
    report = profile(df, target=target, id_like_cols=id_like_cols)

    # 2. drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. drop PII + id-like + (optionally) leakage columns
    dropped = list(dict.fromkeys(id_like_cols + report.pii_suspects))
    if drop_leakage:
        dropped += [c for c in report.leakage_suspects if c not in dropped]
    dropped = [c for c in dropped if c in df.columns]
    df = df.drop(columns=dropped)

    # 4. cap extreme outliers (numeric) at 1st/99th percentile instead of deleting rows
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    for col in numeric_cols:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # 5. collapse rare categories (< threshold share) into "other"
    categorical_cols = [c for c in df.columns if _is_categorical(df[c]) and c != target]
    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < rare_category_threshold].index
        if len(rare):
            df[col] = df[col].where(~df[col].isin(rare), other="other")

    y = df[target].astype(int).values
    X_df = df.drop(columns=[target])

    # 6. split: train / val / test (stratified — class imbalance aware)
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    rel_test = test_size / (test_size + val_size)
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=rel_test, stratify=y_temp, random_state=random_state
    )

    # 7. fit transformer on TRAIN ONLY (avoid leakage into val/test), then transform all
    transformer = build_preprocessor(X_train_df, numeric_cols, categorical_cols)
    X_train = transformer.fit_transform(X_train_df)
    X_val = transformer.transform(X_val_df)
    X_test = transformer.transform(X_test_df)

    feature_names = list(transformer.get_feature_names_out())

    return PreprocessResult(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names,
        transformer=transformer,
        quality_report=report,
        dropped_columns=dropped,
    )


if __name__ == "__main__":
    df = pd.read_csv("data/train_raw.csv")
    result = run_pipeline(df, target="churned")
    print("Dropped columns:", result.dropped_columns)
    print("Train shape:", result.X_train.shape, "Val:", result.X_val.shape, "Test:", result.X_test.shape)
    print("Class balance (train):", np.bincount(result.y_train) / len(result.y_train))
    print("N features after encoding:", len(result.feature_names))
