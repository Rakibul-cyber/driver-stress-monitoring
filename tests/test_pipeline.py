import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.preprocessing import (
    FEATURES,
    HOLDOUT_PARQUET,
    MODEL_PATH,
    RAW_CSV,
    clean,
    features_to_frame,
    validate_schema,
)

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and HOLDOUT_PARQUET.exists()),
    reason="run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def holdout() -> pd.DataFrame:
    return pd.read_parquet(HOLDOUT_PARQUET)


@pytest.fixture(scope="module")
def sample_features(holdout) -> dict[str, float]:
    return {f: float(holdout.iloc[0][f]) for f in FEATURES}


@pytest.fixture(scope="module")
def client():
    from src.api import app

    with TestClient(app) as c:
        yield c


# ---------- dataset / schema ----------

@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not downloaded")
def test_raw_dataset_schema_and_labels():
    df = pd.read_csv(RAW_CSV, nrows=5000)
    validate_schema(df)  # raises on mismatch
    cleaned = clean(df)
    assert set(cleaned["label"].unique()) <= {0, 1}
    assert list(cleaned.columns[4:]) == FEATURES


def test_validate_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"HR": [70.0]}))


@needs_model
def test_holdout_is_subject_independent(holdout):
    import json
    from src.preprocessing import METADATA_PATH

    meta = json.loads(METADATA_PATH.read_text())
    assert set(holdout.subject_id.unique()) == set(meta["split"]["test_subjects"])
    assert not set(meta["split"]["test_subjects"]) & set(meta["split"]["train_subjects"])
    assert holdout[FEATURES].notna().all().all()


def test_features_to_frame_validation():
    with pytest.raises(ValueError, match="missing features"):
        features_to_frame({"HR": 70.0})
    good = {f: 1.0 for f in FEATURES}
    assert features_to_frame(good).shape == (1, len(FEATURES))
    with pytest.raises(ValueError, match="non-finite"):
        features_to_frame({**good, "HR": float("nan")})


# ---------- model ----------

@needs_model
def test_model_prediction_output(sample_features):
    from src.inference import StressPredictor

    res = StressPredictor().predict(sample_features)
    assert res["label"] in {"stress", "no_stress"}
    p = res["probabilities"]
    assert math.isclose(p["stress"] + p["no_stress"], 1.0, abs_tol=1e-3)
    assert 0.5 <= res["confidence"] <= 1.0
    assert res["latency_ms"] > 0


@needs_model
def test_replay_yields_predictions(holdout):
    from src.inference import StressPredictor, replay

    out = list(replay(StressPredictor(), holdout, interval_s=0, step=100, limit=5))
    assert len(out) == 5
    assert all(r["label"] in {"stress", "no_stress"} for r in out)


# ---------- API ----------

@needs_model
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


@needs_model
def test_predict_endpoint(client, sample_features):
    r = client.post("/predict", json={"features": sample_features})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"stress", "no_stress"}
    assert set(body["probabilities"]) == {"stress", "no_stress"}


@needs_model
def test_predict_rejects_invalid_input(client, sample_features):
    r = client.post("/predict", json={"features": {"HR": 80.0}})
    assert r.status_code == 422
    assert "missing features" in r.json()["detail"]
    r = client.post("/predict", json={"features": {**sample_features, "HR": "fast"}})
    assert r.status_code == 422


@needs_model
def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["features"] == FEATURES
    assert "accuracy" in body["test_metrics"]
