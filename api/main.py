"""ML Sentinel API — FastAPI service exposing prediction + governance endpoints.

Run with:  PYTHONPATH=. uvicorn api.main:app --reload --port 8000
Requires artifacts/ to exist (run scripts/train_and_register.py first).
"""
import sys

sys.path.insert(0, ".")
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.auth.security import RateLimiter
from api.routes import governance, predictions

ARTIFACT_DIR = Path("artifacts")
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if (ARTIFACT_DIR / "champion_model.joblib").exists():
        STATE["model"] = joblib.load(ARTIFACT_DIR / "champion_model.joblib")
        STATE["transformer"] = joblib.load(ARTIFACT_DIR / "transformer.joblib")
        STATE["X_background"] = joblib.load(ARTIFACT_DIR / "X_train_background.joblib")
        STATE["feature_names"] = json.loads((ARTIFACT_DIR / "feature_names.json").read_text())
        STATE["reports"] = {
            name: json.loads((ARTIFACT_DIR / f"{name}.json").read_text())
            for name in ["quality_report", "drift_report", "risk_report", "eval_report", "leaderboard", "champion_name"]
        }
    else:
        STATE["model"] = None
    yield
    STATE.clear()


app = FastAPI(title="ML Sentinel API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


@app.middleware("http")
async def rate_limit_and_audit(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    # audit log line — in production this ships to CloudWatch / a SIEM
    print(f'[AUDIT] {request.method} {request.url.path} ip={client_ip} status={response.status_code} took_ms={duration_ms}')
    return response


app.state.STATE = STATE
app.include_router(predictions.router, prefix="/predict", tags=["predictions"])
app.include_router(governance.router, prefix="/governance", tags=["governance"])


@app.get("/health")
def health():
    return {"status": "ok" if STATE.get("model") is not None else "no_model_loaded"}


@app.get("/")
def root():
    return {
        "service": "ML Sentinel API",
        "endpoints": ["/predict/single", "/predict/batch", "/predict/explain",
                      "/governance/risk", "/governance/drift", "/governance/quality",
                      "/governance/leaderboard", "/governance/champion", "/health"],
    }
