from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _reports(request: Request) -> dict:
    state = request.app.state.STATE
    if "reports" not in state:
        raise HTTPException(status_code=503, detail="no governance reports found — run scripts/train_and_register.py")
    return state["reports"]


@router.get("/risk")
def get_risk(request: Request):
    return _reports(request)["risk_report"]


@router.get("/drift")
def get_drift(request: Request):
    return _reports(request)["drift_report"]


@router.get("/quality")
def get_quality(request: Request):
    return _reports(request)["quality_report"]


@router.get("/leaderboard")
def get_leaderboard(request: Request):
    return _reports(request)["leaderboard"]


@router.get("/champion")
def get_champion(request: Request):
    return {**_reports(request)["champion_name"], **_reports(request)["eval_report"]}
