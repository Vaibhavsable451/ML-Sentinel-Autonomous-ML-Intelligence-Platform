import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.data_quality import profile
from ml.preprocessing.pipeline import run_pipeline
from ml.features.engineering import add_engineered_features
from ml.drift.drift_detection import population_stability_index, full_drift_report
from ml.risk.risk_engine import build_risk_report, score_performance
from ml.risk.security import validate_input_schema, SchemaValidationError, mask_value


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv("data/train_raw.csv")


@pytest.fixture(scope="module")
def prod_df():
    return pd.read_csv("data/production_traffic.csv")


def test_data_quality_flags_known_issues(raw_df):
    report = profile(raw_df, target="churned", id_like_cols=["customer_id", "email"])
    assert report.duplicate_rows > 0
    assert "email" in report.pii_suspects
    assert "last_invoice_status" in report.leakage_suspects
    assert 0 <= report.quality_score <= 100


def test_preprocessing_pipeline_shapes(raw_df):
    result = run_pipeline(raw_df, target="churned")
    assert result.X_train.shape[0] > 0
    assert result.X_train.shape[1] == len(result.feature_names)
    assert set(result.y_train).issubset({0, 1})
    # leakage column and PII must never reach the model
    assert not any("last_invoice_status" in f or "email" in f for f in result.feature_names)


def test_feature_engineering_adds_expected_columns(raw_df):
    enriched = add_engineered_features(raw_df)
    for col in ["debt_to_income", "tenure_bucket", "credit_band"]:
        assert col in enriched.columns


def test_psi_is_near_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    assert population_stability_index(x, x) < 1e-6


def test_drift_report_flags_the_known_shifted_feature(raw_df, prod_df):
    shared = [c for c in raw_df.columns if c in prod_df.columns and c not in ("customer_id", "email", "churned")]
    report = full_drift_report(raw_df, prod_df, shared)
    plan_result = next(f for f in report.feature_drift if f["feature"] == "plan")
    assert plan_result["severity"] in ("moderate", "significant")


def test_risk_score_bounded_0_to_100():
    report = build_risk_report(
        data_quality_score=50, drift_pct=80, current_f1=0.3, baseline_f1=0.9,
        calibration_error=0.3, fairness_gap=0.4, security_findings=(5, 2),
    )
    assert 0 <= report.total_score <= 100
    assert report.risk_level in ("low", "moderate", "high", "critical")


def test_performance_risk_zero_when_no_degradation():
    assert score_performance(current_f1=0.7, baseline_f1=0.7) == 0.0


def test_schema_validation_rejects_malicious_payload():
    with pytest.raises(SchemaValidationError):
        validate_input_schema({"name": "<script>alert(1)</script>"}, {"name": str})


def test_schema_validation_accepts_valid_payload():
    validate_input_schema({"age": 30}, {"age": (int, float)})  # should not raise


def test_pii_masking_hides_email():
    masked = mask_value("contact me at john@example.com")
    assert "john@example.com" not in masked
