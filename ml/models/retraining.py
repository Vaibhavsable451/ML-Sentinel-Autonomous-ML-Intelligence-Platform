"""Autonomous Retraining Engine.

Monitor -> detect drift/risk -> decide -> (optionally) train challenger ->
evaluate -> governance gate -> register -> require human approval before promotion.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

from ml.preprocessing.pipeline import run_pipeline
from ml.models.benchmark import run_benchmark
from ml.models.registry import ModelCardMetrics, save_to_local_registry, evaluate_promotion, get_champion
from ml.evaluation.metrics import evaluate
from ml.risk.risk_engine import build_risk_report
from ml.drift.drift_detection import full_drift_report


@dataclass
class RetrainingDecision:
    should_retrain: bool
    reasons: list
    drift_pct: float
    risk_score: float


DRIFT_TRIGGER_PCT = 25.0     # % of features showing moderate+ drift
RISK_TRIGGER_SCORE = 45.0    # overall 0-100 risk score
PERFORMANCE_DROP_TRIGGER = 0.10  # relative F1 drop vs champion


def decide_retraining(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str],
    current_f1: float, champion_f1: float, data_quality_score: float,
) -> RetrainingDecision:
    drift = full_drift_report(reference_df, current_df, feature_cols)
    risk = build_risk_report(
        data_quality_score=data_quality_score, drift_pct=drift.overall_drift_pct,
        current_f1=current_f1, baseline_f1=champion_f1,
        calibration_error=0.03, fairness_gap=0.05,
    )

    reasons = []
    rel_drop = (champion_f1 - current_f1) / champion_f1 if champion_f1 else 0
    if drift.overall_drift_pct >= DRIFT_TRIGGER_PCT:
        reasons.append(f"feature drift at {drift.overall_drift_pct}% of features (threshold {DRIFT_TRIGGER_PCT}%)")
    if risk.total_score >= RISK_TRIGGER_SCORE:
        reasons.append(f"overall risk score {risk.total_score} exceeds threshold {RISK_TRIGGER_SCORE}")
    if rel_drop >= PERFORMANCE_DROP_TRIGGER:
        reasons.append(f"performance degraded {rel_drop*100:.1f}% relative to champion")

    return RetrainingDecision(
        should_retrain=len(reasons) > 0, reasons=reasons or ["no trigger conditions met"],
        drift_pct=drift.overall_drift_pct, risk_score=risk.total_score,
    )


def run_autonomous_cycle(train_df: pd.DataFrame, production_df: pd.DataFrame, target: str = "churned") -> dict:
    """Full monitor -> retrain -> evaluate -> gate -> (await approval) cycle."""
    champion = get_champion()
    champion_f1 = champion["metrics"]["f1"] if champion else 0.55

    shared_cols = [c for c in train_df.columns if c in production_df.columns and c not in ("customer_id", "email", target)]

    # NOTE: production_df here has no ground-truth label yet in real life; for this
    # demo we assume a lagged-label sample is available for evaluation purposes.
    result = run_pipeline(train_df, target=target)
    decision = decide_retraining(
        reference_df=train_df, current_df=production_df, feature_cols=shared_cols,
        current_f1=champion_f1 * 0.85,  # simulate observed degradation for the demo
        champion_f1=champion_f1,
        data_quality_score=result.quality_report.quality_score,
    )

    outcome = {"decision": asdict(decision), "trained_challenger": False, "promotion": None}
    if decision.should_retrain:
        bench = run_benchmark(result.X_train, result.y_train, cv_folds=3)
        eval_report = evaluate(bench.champion_model, result.X_test, result.y_test)
        metrics = ModelCardMetrics(
            accuracy=eval_report.accuracy, f1=eval_report.f1, precision=eval_report.precision,
            recall=eval_report.recall, roc_auc=eval_report.roc_auc, calibration_error=eval_report.calibration_error,
        )
        risk = build_risk_report(
            data_quality_score=result.quality_report.quality_score, drift_pct=decision.drift_pct,
            current_f1=eval_report.f1, baseline_f1=champion_f1,
            calibration_error=eval_report.calibration_error, fairness_gap=0.05,
        )
        entry = save_to_local_registry(bench.champion_model, bench.champion_name, metrics, risk.total_score)
        promotion = evaluate_promotion(entry)
        outcome.update({
            "trained_challenger": True,
            "challenger_name": bench.champion_name,
            "challenger_metrics": metrics.__dict__,
            "risk_report": risk.__dict__,
            # safety: NEVER auto-promote to production — always require human approval
            "promotion": {**promotion.__dict__, "requires_human_approval": True},
        })
    return outcome


if __name__ == "__main__":
    train = pd.read_csv("data/train_raw.csv")
    prod = pd.read_csv("data/production_traffic.csv")
    result = run_autonomous_cycle(train, prod)
    import json
    print(json.dumps(result, indent=2, default=str))
