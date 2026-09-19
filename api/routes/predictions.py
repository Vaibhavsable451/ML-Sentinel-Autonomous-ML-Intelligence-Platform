from __future__ import annotations
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, HTTPException

from api.schemas import CustomerFeatures, PredictionResponse, ExplainResponse
from ml.risk.security import validate_input_schema, SchemaValidationError
from ml.explainability.explain import explain_instance, find_counterfactual
from ml.features.engineering import add_engineered_features

router = APIRouter()

EXPECTED_SCHEMA = {
    "age": (int, float), "tenure_months": (int, float), "monthly_charges": (int, float),
    "annual_income": (int, float), "outstanding_debt": (int, float), "credit_score": (int, float),
    "support_tickets_90d": int, "plan": str, "contract_type": str, "payment_method": str, "region": str,
}


def _risk_label(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.6:
        return "medium"
    return "high"


def _transform_one(request: Request, features: CustomerFeatures) -> np.ndarray:
    state = request.app.state.STATE
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="no trained model artifact found — run scripts/train_and_register.py")

    try:
        validate_input_schema(features.model_dump(), EXPECTED_SCHEMA)
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    row_df = pd.DataFrame([features.model_dump()])
    row_df = add_engineered_features(row_df)
    X = state["transformer"].transform(row_df)
    return X


@router.post("/single", response_model=PredictionResponse)
def predict_single(features: CustomerFeatures, request: Request):
    state = request.app.state.STATE
    X = _transform_one(request, features)
    prob = float(state["model"].predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    return PredictionResponse(churn_probability=round(prob, 4), prediction=pred, risk_label=_risk_label(prob))


@router.post("/batch")
def predict_batch(rows: list[CustomerFeatures], request: Request):
    if len(rows) > 5000:
        raise HTTPException(status_code=400, detail="batch too large (max 5000 rows)")
    state = request.app.state.STATE
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="no trained model artifact found")
    df = pd.DataFrame([r.model_dump() for r in rows])
    df = add_engineered_features(df)
    X = state["transformer"].transform(df)
    probs = state["model"].predict_proba(X)[:, 1]
    return [
        {"churn_probability": round(float(p), 4), "prediction": int(p >= 0.5), "risk_label": _risk_label(p)}
        for p in probs
    ]


@router.post("/explain", response_model=ExplainResponse)
def predict_explain(features: CustomerFeatures, request: Request):
    state = request.app.state.STATE
    X = _transform_one(request, features)
    row = X[0]
    prob = float(state["model"].predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    top_factors = explain_instance(state["model"], row, state["X_background"], state["feature_names"], top_k=5)
    cf = None
    if pred == 1:
        cf = find_counterfactual(state["model"], row, state["feature_names"])

    return ExplainResponse(
        prediction=PredictionResponse(churn_probability=round(prob, 4), prediction=pred, risk_label=_risk_label(prob)),
        top_factors=top_factors,
        counterfactual=cf,
    )
