"""Fairness metrics across a sensitive/protected group column."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class GroupFairnessReport:
    per_group_selection_rate: dict
    per_group_tpr: dict
    per_group_fpr: dict
    per_group_accuracy: dict
    demographic_parity_gap: float
    equal_opportunity_gap: float
    equalized_odds_gap: float


def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> GroupFairnessReport:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.values})

    selection_rate, tpr, fpr, acc = {}, {}, {}, {}
    for g, gdf in df.groupby("group"):
        selection_rate[g] = round(gdf["y_pred"].mean(), 4)
        positives = gdf[gdf.y_true == 1]
        negatives = gdf[gdf.y_true == 0]
        tpr[g] = round((positives.y_pred == 1).mean(), 4) if len(positives) else None
        fpr[g] = round((negatives.y_pred == 1).mean(), 4) if len(negatives) else None
        acc[g] = round((gdf.y_true == gdf.y_pred).mean(), 4)

    def _gap(d: dict) -> float:
        vals = [v for v in d.values() if v is not None]
        return round(max(vals) - min(vals), 4) if len(vals) >= 2 else 0.0

    dp_gap = _gap(selection_rate)
    eo_gap = _gap(tpr)
    eodds_gap = round(max(_gap(tpr), _gap(fpr)), 4)

    return GroupFairnessReport(
        per_group_selection_rate=selection_rate,
        per_group_tpr=tpr,
        per_group_fpr=fpr,
        per_group_accuracy=acc,
        demographic_parity_gap=dp_gap,
        equal_opportunity_gap=eo_gap,
        equalized_odds_gap=eodds_gap,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ml.preprocessing.pipeline import run_pipeline
    from ml.models.benchmark import run_benchmark

    df = pd.read_csv("data/train_raw.csv")
    result = run_pipeline(df, target="churned")
    bench = run_benchmark(result.X_train, result.y_train, cv_folds=3)

    # recover the 'region' column aligned to the test split for a fairness slice
    df_clean = df.drop_duplicates().reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    _, X_temp, _, y_temp, _, region_temp = train_test_split(
        df_clean.drop(columns=["churned"]), df_clean["churned"], df_clean["region"],
        test_size=0.3, stratify=df_clean["churned"], random_state=42,
    )
    _, region_test = train_test_split(region_temp, test_size=0.5, stratify=y_temp, random_state=42)

    y_pred = bench.champion_model.predict(result.X_test)
    report = evaluate_fairness(result.y_test, y_pred, region_test.reset_index(drop=True))
    print(report)
