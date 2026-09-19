
from typing import ClassVar

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    age: float = Field(..., ge=18, le=100)
    tenure_months: float = Field(..., ge=0, le=600)
    monthly_charges: float = Field(..., ge=0, le=1000)
    annual_income: float = Field(..., ge=0, le=2_000_000)
    outstanding_debt: float = Field(..., ge=0, le=2_000_000)
    credit_score: float = Field(..., ge=300, le=850)
    support_tickets_90d: int = Field(..., ge=0, le=100)
    plan: str
    contract_type: str
    payment_method: str
    region: str

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "age": 34, "tenure_months": 8, "monthly_charges": 95.5,
                "annual_income": 48000, "outstanding_debt": 22000, "credit_score": 640,
                "support_tickets_90d": 3, "plan": "premium", "contract_type": "month-to-month",
                "payment_method": "e_check", "region": "west",
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    risk_label: str


class ExplanationItem(BaseModel):
    feature: str
    contribution: float
    value: float


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    top_factors: list[ExplanationItem]
    counterfactual: dict | None = None
