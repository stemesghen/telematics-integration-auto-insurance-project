import os, json, logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

#  Paths & knobs 
DATA_INGEST   = Path("data_ingest")
DRIVER_PERIOD = Path("data/driver_period.csv")
MODEL_PATH    = Path("models/behavior_model.joblib")

BASELINE   = float(os.getenv("BASELINE_PREV", "0.30"))
SLOPE_K    = float(os.getenv("PRICING_SLOPE", "0.5"))
FACTOR_MIN = float(os.getenv("FACTOR_MIN", "0.90"))
FACTOR_MAX = float(os.getenv("FACTOR_MAX", "1.10"))

#  Simple API key auth (POC) 
API_KEY = os.getenv("API_KEY", "dev-secret")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def require_api_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

#  Logging 
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

#  Schema
class TelemetryPoint(BaseModel):
    ts: datetime
    lat: float
    lon: float
    speed_mps: Optional[float] = None
    ax_mps2: Optional[float] = None
    ay_mps2: Optional[float] = None
    az_mps2: Optional[float] = None

class TelemetryBatch(BaseModel):
    policy_id: str
    trip_id: str
    points: List[TelemetryPoint]

# 6 features (matching w/ the model)
class ScoreRequest(BaseModel):
    exposure_miles: float = 0.0
    trip_ct: float = 0.0
    harsh_brake_per_100mi: float = 0.0
    duration_s: float = 0.0
    overspeed_ratio: float = 0.0
    night_ratio: float = 0.0

#  Model reload helper 
def _ensure_model_loaded():
    """Load/reload the model artifact if it's not currently in memory."""
    global model, FEATURE_NAMES
    if model is None and MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, FEATURE_NAMES = loaded
        else:
            model = loaded
            # fallback names if artifact didn't include them
            FEATURE_NAMES = [
                "exposure_miles","trip_ct","harsh_brake_per_100mi","duration_s",
                "avg_overspeed_ratio","night_miles_ratio","phone_usage_per_hr",
                "mean_speed_mps","miles_per_trip","speed_var_across_trips"
            ]


#  Other Helpers 
def _predict_proba_safe(M, X_df: pd.DataFrame) -> np.ndarray:
    """Return P(y=1) regardless of whether the model is a pipeline, calibrated model, etc."""
    if hasattr(M, "predict_proba"):
        p = M.predict_proba(X_df)
        p = p[0] if isinstance(p, list) else p
        return np.asarray(p)[:, 1]
    if hasattr(M, "decision_function"):
        z = np.asarray(M.decision_function(X_df)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))
    yhat = np.asarray(M.predict(X_df)).reshape(-1)
    return yhat.astype(float)

def _latest_dp_for_policy(policy_id: str) -> dict:
    if not DRIVER_PERIOD.exists():
        raise HTTPException(503, "driver_period.csv not found—run the builder first.")
    df = pd.read_csv(DRIVER_PERIOD, parse_dates=["period_start"])
    rows = df[df["policy_id"] == policy_id].sort_values("period_start")
    if rows.empty:
        raise HTTPException(404, f"No driver-period row for policy_id={policy_id}")
    r = rows.iloc[-1]
    feats = [
        "exposure_miles","trip_ct","harsh_brake_per_100mi",
        "duration_s","overspeed_ratio","night_ratio"
    ]
    return {k: float(r.get(k, 0.0)) for k in feats}

# The App 
app = FastAPI(title="Telematics Ingest & Scoring API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
FEATURE_NAMES = None

@app.on_event("startup")
def _load_model():
    global model, FEATURE_NAMES
    loaded = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, FEATURE_NAMES = loaded
    else:
        model = loaded
        FEATURE_NAMES = [
            "exposure_miles","trip_ct","harsh_brake_per_100mi",
            "duration_s","overspeed_ratio","night_ratio"
        ]
    if model is None:
        logger.warning("Model not loaded; %s not found", MODEL_PATH)
    else:
        logger.info("Loaded model with features: %s", FEATURE_NAMES)

@app.get("/healthz")
def healthz():
    _ensure_model_loaded()  # make /healthz reflect current artifact
    return {"ok": True, "model_loaded": MODEL_PATH.exists(), "features": FEATURE_NAMES}

# Accept single batch or list of batches
@app.post("/ingest/telemetry", dependencies=[Depends(require_api_key)])
def ingest(payload: Union[TelemetryBatch, List[TelemetryBatch]] = Body(...)):
    DATA_INGEST.mkdir(exist_ok=True)
    batches = payload if isinstance(payload, list) else [payload]
    total = 0
    last_file = None
    for batch in batches:
        out = DATA_INGEST / f"{batch.policy_id}__{batch.trip_id}.jsonl"
        with out.open("a") as f:
            for p in batch.points:
                f.write(json.dumps({
                    "policy_id": batch.policy_id,
                    "trip_id": batch.trip_id,
                    **p.model_dump(mode="json")
                }) + "\n")
                total += 1
        last_file = str(out)
    return {"status": "ok", "received": total, "file": last_file}


@app.post("/score", dependencies=[Depends(require_api_key)])
def score(req: ScoreRequest):
    _ensure_model_loaded()  
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # build row robustly against missing fields
    req_dict = req.model_dump()
    row = {name: float(req_dict.get(name, 0.0)) for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)

    prob = float(_predict_proba_safe(model, X)[0])
    return {"risk_p": prob, "features": row}
@app.get("/pricing/{policy_id}", dependencies=[Depends(require_api_key)])
def pricing(policy_id: str):
    _ensure_model_loaded()  
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feats = _latest_dp_for_policy(policy_id)
    feats = {name: float(feats.get(name, 0.0)) for name in FEATURE_NAMES}
    X = pd.DataFrame([feats], columns=FEATURE_NAMES)

    p = float(_predict_proba_safe(model, X)[0])
    factor = float(np.clip(1 + SLOPE_K * (p - BASELINE), FACTOR_MIN, FACTOR_MAX))
    return {
        "policy_id": policy_id,
        "risk_p": p,
        "telematics_factor": factor,
        "features": feats,
        "baseline": BASELINE,
        "slope_k": SLOPE_K,
        "cap": [FACTOR_MIN, FACTOR_MAX],
    }


@app.post("/score", dependencies=[Depends(require_api_key)])
def score(req: ScoreRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    req_dict = req.model_dump()
    row = {name: float(req_dict.get(name, 0.0)) for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    prob = float(_predict_proba_safe(model, X)[0])
    return {"risk_p": prob, "features": row}

@app.get("/pricing/{policy_id}", dependencies=[Depends(require_api_key)])
def pricing(policy_id: str):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    feats = _latest_dp_for_policy(policy_id)
    feats = {name: float(feats.get(name, 0.0)) for name in FEATURE_NAMES}
    X = pd.DataFrame([feats], columns=FEATURE_NAMES)
    p = float(_predict_proba_safe(model, X)[0])
    factor = float(np.clip(1 + SLOPE_K * (p - BASELINE), FACTOR_MIN, FACTOR_MAX))
    return {
        "policy_id": policy_id,
        "risk_p": p,
        "telematics_factor": factor,
        "features": feats,
        "baseline": BASELINE,
        "slope_k": SLOPE_K,
        "cap": [FACTOR_MIN, FACTOR_MAX],
    }

