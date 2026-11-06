"""FastAPI prediction service.

Run:  uvicorn src.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import FEATURES, METADATA_PATH
from src.inference import StressPredictor

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = StressPredictor()
    except FileNotFoundError as e:  # start anyway; /health reports the problem
        state["error"] = str(e)
    yield
    state.clear()


app = FastAPI(
    title="Driver Stress Monitoring API",
    description=(
        "Binary stress classification (stress vs. no_stress) from 30 HRV features, "
        "using a Random Forest trained on the WESAD dataset. Research prototype — "
        "not a medical or automotive safety device."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _example_features() -> dict[str, float]:
    try:
        return json.loads(METADATA_PATH.read_text())["example_input"]
    except (FileNotFoundError, KeyError, ValueError):
        return {f: 0.0 for f in FEATURES}


class PredictRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"features": _example_features()}]}}

    features: dict[str, float] = Field(
        ...,
        description=f"All {len(FEATURES)} HRV features by name: {', '.join(FEATURES)}",
    )


class PredictResponse(BaseModel):
    label: str
    label_id: int
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float


def get_predictor() -> StressPredictor:
    if "predictor" not in state:
        raise HTTPException(status_code=503, detail=state.get("error", "Model not loaded"))
    return state["predictor"]


@app.get("/health")
def health():
    loaded = "predictor" in state
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
            **({} if loaded else {"error": state.get("error")})}


@app.get("/model-info")
def model_info():
    md = get_predictor().metadata
    return {k: md[k] for k in (
        "model_name", "task", "label_names", "threshold", "features", "dataset",
        "split", "cv_loso", "test_metrics", "feature_importances", "example_input",
        "trained_at", "versions",
    )} | {"cv_loso": {"summary": md["cv_loso"]["summary"]}}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor = get_predictor()
    try:
        return predictor.predict(req.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
