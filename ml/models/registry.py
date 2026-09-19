"""Model registry: MLflow-backed experiment tracking + champion/challenger promotion logic.

Uses a local MLflow tracking store (file-based) so this runs anywhere;
point MLFLOW_TRACKING_URI at a remote MLflow tracking server in production.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

REGISTRY_DIR = Path("model_registry")
MLFLOW_TRACKING_DIR = Path("mlruns")


def _init_mlflow(experiment_name: str = "ml-sentinel-churn"):
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR.resolve()}")
    mlflow.set_experiment(experiment_name)


@dataclass
class ModelCardMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    calibration_error: float


@dataclass
class RegisteredModel:
    name: str
    version: str
    status: str  # "candidate" | "champion" | "archived"
    metrics: dict
    risk_score: float | None
    created_at: str
    artifact_path: str
    sha256: str


def log_training_run(model, model_name: str, metrics: ModelCardMetrics, params: dict, X_sample) -> str:
    """Logs a training run to MLflow and returns the run_id."""
    _init_mlflow()
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(asdict(metrics))
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_sample[:5])
        return run.info.run_id


def save_to_local_registry(
    model, name: str, metrics: ModelCardMetrics, risk_score: float | None = None,
    status: str = "candidate",
) -> RegisteredModel:
    """A simple, file-based registry (JSON index + joblib artifacts) that mirrors what
    an MLflow Model Registry would store, so this works offline too."""
    from ml.risk.security import compute_artifact_checksum

    REGISTRY_DIR.mkdir(exist_ok=True)
    index_path = REGISTRY_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    version = f"v{len(index) + 1}"
    artifact_path = REGISTRY_DIR / f"{name}_{version}.joblib"
    joblib.dump(model, artifact_path)
    sha = compute_artifact_checksum(artifact_path)

    entry = RegisteredModel(
        name=name, version=version, status=status,
        metrics=asdict(metrics), risk_score=risk_score,
        created_at=datetime.now(timezone.utc).isoformat(),
        artifact_path=str(artifact_path), sha256=sha,
    )
    index.append(asdict(entry))
    index_path.write_text(json.dumps(index, indent=2))
    return entry


def get_champion() -> dict | None:
    index_path = REGISTRY_DIR / "index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    champions = [e for e in index if e["status"] == "champion"]
    return champions[-1] if champions else None


def promote(version_entry: dict, reason: str) -> None:
    index_path = REGISTRY_DIR / "index.json"
    index = json.loads(index_path.read_text())
    for e in index:
        if e["status"] == "champion":
            e["status"] = "archived"
        if e["name"] == version_entry["name"] and e["version"] == version_entry["version"]:
            e["status"] = "champion"
            e["promotion_reason"] = reason
    index_path.write_text(json.dumps(index, indent=2))


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    challenger_metrics: dict
    champion_metrics: dict | None


def evaluate_promotion(
    challenger: RegisteredModel,
    min_f1_improvement: float = 0.0,
    max_risk_score: float = 60.0,
    latency_budget_ok: bool = True,
) -> PromotionDecision:
    """Champion/challenger gate: challenger must beat (or tie) champion on F1, not exceed
    the max acceptable risk score, and respect latency budget — otherwise it stays a candidate."""
    champion = get_champion()
    champ_metrics = champion["metrics"] if champion else None

    reasons = []
    ok = True

    if champion is not None:
        if challenger.metrics["f1"] < champ_metrics["f1"] + min_f1_improvement:
            ok = False
            reasons.append(f"F1 {challenger.metrics['f1']} does not beat champion {champ_metrics['f1']}")
    else:
        reasons.append("no existing champion — first model becomes champion by default")

    if challenger.risk_score is not None and challenger.risk_score > max_risk_score:
        ok = False
        reasons.append(f"risk score {challenger.risk_score} exceeds threshold {max_risk_score}")

    if not latency_budget_ok:
        ok = False
        reasons.append("latency budget exceeded")

    if ok and not reasons:
        reasons.append("challenger meets or exceeds champion on all governance checks")

    return PromotionDecision(
        promote=ok, reason="; ".join(reasons),
        challenger_metrics=challenger.metrics, champion_metrics=champ_metrics,
    )


if __name__ == "__main__":
    import sys

    import pandas as pd
    sys.path.insert(0, ".")
    from ml.evaluation.metrics import evaluate
    from ml.models.benchmark import run_benchmark
    from ml.preprocessing.pipeline import run_pipeline
    from ml.risk.risk_engine import build_risk_report

    df = pd.read_csv("data/train_raw.csv")
    result = run_pipeline(df, target="churned")
    bench = run_benchmark(result.X_train, result.y_train, cv_folds=3)
    eval_report = evaluate(bench.champion_model, result.X_test, result.y_test)

    metrics = ModelCardMetrics(
        accuracy=eval_report.accuracy, f1=eval_report.f1, precision=eval_report.precision,
        recall=eval_report.recall, roc_auc=eval_report.roc_auc, calibration_error=eval_report.calibration_error,
    )
    risk = build_risk_report(
        data_quality_score=result.quality_report.quality_score, drift_pct=0,
        current_f1=eval_report.f1, baseline_f1=eval_report.f1,
        calibration_error=eval_report.calibration_error, fairness_gap=0.05,
    )
    entry = save_to_local_registry(bench.champion_model, bench.champion_name, metrics, risk.total_score)
    print("Registered:", entry)

    decision = evaluate_promotion(entry)
    print("\nPromotion decision:", decision)
    if decision.promote:
        promote(asdict(entry), decision.reason)
        print("Promoted to champion.")
