"""Data-quality profiling: the checks ML Sentinel runs before it trusts a dataset."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    missing_pct: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    outlier_cols: dict = field(default_factory=dict)
    dtype_issues: list = field(default_factory=list)
    imbalance: dict | None = None
    leakage_suspects: list = field(default_factory=list)
    pii_suspects: list = field(default_factory=list)
    quality_score: float = 0.0

    def to_dict(self):
        return self.__dict__


PII_NAME_HINTS = ("email", "ssn", "phone", "address", "name", "passport", "credit_card")


def _detect_outliers_iqr(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 10 or not np.issubdtype(s.dtype, np.number):
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    return int(((s < lower) | (s > upper)).sum())


def _is_categorical(series: pd.Series) -> bool:
    """True for object dtype AND pandas' newer native string dtype (pandas>=2.x/3.x)."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _leakage_check(df: pd.DataFrame, target: str, id_like_cols: list[str]) -> list[str]:
    """Flag columns whose single-value groups are near-perfectly separated by target."""
    suspects = []
    for col in df.columns:
        if col in (target, *id_like_cols):
            continue
        if _is_categorical(df[col]) and df[col].nunique() <= 20:
            purity = df.groupby(col)[target].mean()
            # if any category predicts the target with >90% purity and has decent support
            support = df[col].value_counts()
            for cat, p in purity.items():
                if support.get(cat, 0) >= 20 and (p >= 0.93 or p <= 0.07):
                    suspects.append(col)
                    break
    return suspects


def profile(df: pd.DataFrame, target: str | None = None, id_like_cols: list[str] | None = None) -> DataQualityReport:
    id_like_cols = id_like_cols or []
    n_rows, n_cols = df.shape

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    duplicate_rows = int(df.duplicated().sum())

    outlier_cols = {}
    for col in df.select_dtypes(include=np.number).columns:
        n_out = _detect_outliers_iqr(df[col])
        if n_out > 0:
            outlier_cols[col] = n_out

    dtype_issues = []
    for col in df.columns:
        if _is_categorical(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.9 and df[col].notna().mean() > 0:
                dtype_issues.append(f"{col}: looks numeric but stored as text")

    imbalance = None
    if target and target in df.columns:
        counts = df[target].value_counts(normalize=True)
        imbalance = {"classes": counts.round(3).to_dict(), "minority_pct": round(counts.min() * 100, 2)}

    leakage_suspects = _leakage_check(df, target, id_like_cols) if target else []

    pii_suspects = [c for c in df.columns if any(h in c.lower() for h in PII_NAME_HINTS)]

    # composite 0-100 quality score (higher is better)
    penalty = 0.0
    penalty += min(30, sum(missing_pct.values()) / max(n_cols, 1))
    penalty += min(20, duplicate_rows / max(n_rows, 1) * 100)
    penalty += min(15, sum(outlier_cols.values()) / max(n_rows, 1) * 100)
    penalty += 10 * len(dtype_issues)
    penalty += 15 * len(leakage_suspects)
    quality_score = round(max(0.0, 100.0 - penalty), 1)

    return DataQualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        missing_pct={k: v for k, v in missing_pct.items() if v > 0},
        duplicate_rows=duplicate_rows,
        outlier_cols=outlier_cols,
        dtype_issues=dtype_issues,
        imbalance=imbalance,
        leakage_suspects=leakage_suspects,
        pii_suspects=pii_suspects,
        quality_score=quality_score,
    )


if __name__ == "__main__":
    df = pd.read_csv("data/train_raw.csv")
    report = profile(df, target="churned", id_like_cols=["customer_id", "email"])
    import json
    print(json.dumps(report.to_dict(), indent=2, default=str))
