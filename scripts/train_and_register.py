"""End-to-end training script: run this to (re)produce the champion model artifact
that the FastAPI service and Streamlit dashboard read from.

    PYTHONPATH=. python3 scripts/train_and_register.py
"""
import json
import sys

sys.path.insert(0, ".")
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from ml.drift.drift_detection import full_drift_report
from ml.evaluation.metrics import evaluate
from ml.features.engineering import add_engineered_features
from ml.models.benchmark import run_benchmark
from ml.models.registry import (
    ModelCardMetrics,
    evaluate_promotion,
    promote,
    save_to_local_registry,
)
from ml.preprocessing.pipeline import run_pipeline
from ml.risk.risk_engine import build_risk_report
from ml.risk.security import scan_dataframe_for_pii

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv("data/train_raw.csv")
    prod = pd.read_csv("data/production_traffic.csv")

    enriched = add_engineered_features(df)
    result = run_pipeline(enriched, target="churned")

    print("Benchmarking models...")
    bench = run_benchmark(result.X_train, result.y_train, cv_folds=4)
    print(bench.leaderboard.to_string(index=False))

    eval_report = evaluate(bench.champion_model, result.X_test, result.y_test)
    metrics = ModelCardMetrics(
        accuracy=eval_report.accuracy, f1=eval_report.f1, precision=eval_report.precision,
        recall=eval_report.recall, roc_auc=eval_report.roc_auc, calibration_error=eval_report.calibration_error,
    )

    shared_cols = [c for c in df.columns if c in prod.columns and c not in ("customer_id", "email", "churned")]
    drift = full_drift_report(df, prod, shared_cols)

    pii_findings = scan_dataframe_for_pii(df)

    risk = build_risk_report(
        data_quality_score=result.quality_report.quality_score,
        drift_pct=drift.overall_drift_pct,
        current_f1=eval_report.f1, baseline_f1=eval_report.f1,
        calibration_error=eval_report.calibration_error,
        fairness_gap=0.05,
        security_findings=(len(pii_findings), 0),
    )

    entry = save_to_local_registry(bench.champion_model, bench.champion_name, metrics, risk.total_score)
    decision = evaluate_promotion(entry)
    if decision.promote:
        promote(asdict(entry), decision.reason)

    # persist everything the API/UI need, without re-running training
    joblib.dump(bench.champion_model, ARTIFACT_DIR / "champion_model.joblib")
    joblib.dump(result.transformer, ARTIFACT_DIR / "transformer.joblib")
    joblib.dump(result.X_train, ARTIFACT_DIR / "X_train_background.joblib")
    (ARTIFACT_DIR / "feature_names.json").write_text(json.dumps(result.feature_names))
    (ARTIFACT_DIR / "leaderboard.json").write_text(bench.leaderboard.to_json(orient="records"))
    (ARTIFACT_DIR / "quality_report.json").write_text(json.dumps(result.quality_report.to_dict(), default=str))
    (ARTIFACT_DIR / "drift_report.json").write_text(json.dumps(drift.__dict__, default=str))
    (ARTIFACT_DIR / "risk_report.json").write_text(json.dumps(risk.__dict__, default=str))
    (ARTIFACT_DIR / "eval_report.json").write_text(json.dumps(eval_report.__dict__, default=str))
    (ARTIFACT_DIR / "champion_name.json").write_text(json.dumps({"name": bench.champion_name}))

    print(f"\nChampion: {bench.champion_name} | F1={eval_report.f1} | ROC-AUC={eval_report.roc_auc}")
    print(f"Data quality score: {result.quality_report.quality_score}/100")
    print(f"Drift: {drift.overall_drift_pct}% of features drifted")
    print(f"Risk score: {risk.total_score}/100 ({risk.risk_level})")
    print(f"Artifacts written to {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
