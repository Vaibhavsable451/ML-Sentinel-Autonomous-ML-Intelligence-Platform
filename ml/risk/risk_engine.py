"""ML System Risk Engine — rolls up every signal into one 0-100 risk score.

Higher score = higher risk (0 = perfectly healthy, 100 = critical).
Each sub-score is 0-100 risk as well, then weighted-averaged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time


DEFAULT_WEIGHTS = {
    "data_quality": 0.18,
    "feature_drift": 0.20,
    "performance": 0.20,
    "fairness": 0.12,
    "calibration": 0.10,
    "security": 0.08,
    "reliability": 0.12,
}


@dataclass
class RiskBreakdown:
    data_quality: float
    feature_drift: float
    performance: float
    fairness: float
    calibration: float
    security: float
    reliability: float

    def weighted_total(self, weights: dict = DEFAULT_WEIGHTS) -> float:
        total = sum(getattr(self, k) * w for k, w in weights.items())
        return round(min(100.0, max(0.0, total)), 1)


@dataclass
class RiskReport:
    total_score: float
    breakdown: dict
    risk_level: str
    top_drivers: list
    generated_at: float = field(default_factory=time.time)


def _risk_level(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "high"
    return "critical"


def score_data_quality(quality_score_0_100_good: float) -> float:
    """Convert a 'higher is better' quality score into a 'higher is worse' risk score."""
    return round(100 - quality_score_0_100_good, 1)


def score_feature_drift(overall_drift_pct: float, prediction_drift_psi: float | None = None) -> float:
    risk = overall_drift_pct  # already 0-100
    if prediction_drift_psi is not None:
        risk = 0.7 * risk + 0.3 * min(100, prediction_drift_psi * 200)
    return round(min(100, risk), 1)


def score_performance(current_f1: float, baseline_f1: float) -> float:
    if baseline_f1 <= 0:
        return 0.0
    drop_pct = max(0.0, (baseline_f1 - current_f1) / baseline_f1) * 100
    return round(min(100, drop_pct * 4), 1)  # a 25% relative F1 drop => risk 100


def score_calibration(calibration_error: float) -> float:
    # calibration_error is mean abs gap between predicted & observed probs (0=perfect)
    return round(min(100, calibration_error * 400), 1)


def score_fairness(max_group_gap: float) -> float:
    # max_group_gap: largest absolute difference in selection rate / TPR between groups (0-1)
    return round(min(100, max_group_gap * 200), 1)


def score_security(open_findings: int, critical_findings: int) -> float:
    return round(min(100, open_findings * 8 + critical_findings * 25), 1)


def score_reliability(error_rate_pct: float, p99_latency_ms: float, latency_slo_ms: float = 300) -> float:
    latency_penalty = max(0.0, (p99_latency_ms - latency_slo_ms) / latency_slo_ms) * 50
    return round(min(100, error_rate_pct * 10 + latency_penalty), 1)


def build_risk_report(
    data_quality_score: float,
    drift_pct: float,
    current_f1: float,
    baseline_f1: float,
    calibration_error: float,
    fairness_gap: float,
    security_findings: tuple[int, int] = (0, 0),
    error_rate_pct: float = 0.2,
    p99_latency_ms: float = 120,
    prediction_drift_psi: float | None = None,
) -> RiskReport:
    breakdown = RiskBreakdown(
        data_quality=score_data_quality(data_quality_score),
        feature_drift=score_feature_drift(drift_pct, prediction_drift_psi),
        performance=score_performance(current_f1, baseline_f1),
        fairness=score_fairness(fairness_gap),
        calibration=score_calibration(calibration_error),
        security=score_security(*security_findings),
        reliability=score_reliability(error_rate_pct, p99_latency_ms),
    )
    total = breakdown.weighted_total()
    drivers = sorted(breakdown.__dict__.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return RiskReport(
        total_score=total,
        breakdown=breakdown.__dict__,
        risk_level=_risk_level(total),
        top_drivers=[{"component": k, "score": v} for k, v in drivers],
    )


if __name__ == "__main__":
    report = build_risk_report(
        data_quality_score=82.8,
        drift_pct=14.3,
        current_f1=0.5824,
        baseline_f1=0.61,
        calibration_error=0.0267,
        fairness_gap=0.06,
        security_findings=(1, 0),
        error_rate_pct=0.4,
        p99_latency_ms=180,
        prediction_drift_psi=0.05,
    )
    import json
    print(json.dumps(report.__dict__, indent=2, default=str))
