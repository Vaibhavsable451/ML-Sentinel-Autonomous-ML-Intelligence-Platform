"""Evaluation: classification metrics + calibration quality."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve


@dataclass
class EvalReport:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    brier_score: float
    confusion_matrix: list
    calibration_error: float  # mean |predicted_prob - observed_freq| across bins


def evaluate(model, X, y, n_calibration_bins: int = 10) -> EvalReport:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    frac_pos, mean_pred = calibration_curve(y, y_proba, n_bins=n_calibration_bins, strategy="quantile")
    calibration_error = float(np.mean(np.abs(frac_pos - mean_pred)))

    return EvalReport(
        accuracy=round(accuracy_score(y, y_pred), 4),
        f1=round(f1_score(y, y_pred), 4),
        precision=round(precision_score(y, y_pred), 4),
        recall=round(recall_score(y, y_pred), 4),
        roc_auc=round(roc_auc_score(y, y_proba), 4),
        brier_score=round(brier_score_loss(y, y_proba), 4),
        confusion_matrix=confusion_matrix(y, y_pred).tolist(),
        calibration_error=round(calibration_error, 4),
    )


if __name__ == "__main__":
    import sys, pandas as pd
    sys.path.insert(0, ".")
    from ml.preprocessing.pipeline import run_pipeline
    from ml.models.benchmark import run_benchmark

    df = pd.read_csv("data/train_raw.csv")
    result = run_pipeline(df, target="churned")
    bench = run_benchmark(result.X_train, result.y_train, cv_folds=4)
    report = evaluate(bench.champion_model, result.X_test, result.y_test)
    print(report)
