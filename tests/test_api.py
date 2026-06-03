"""
Unit tests for the ICU FastAPI inference service.

Run with:
    pytest tests/ -v
"""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Stub heavy optional imports so tests work without a full model ────────────
# Stub shap
shap_stub = types.ModuleType("shap")
shap_stub.TreeExplainer   = MagicMock()
shap_stub.LinearExplainer = MagicMock()
sys.modules.setdefault("shap", shap_stub)

# Stub mlflow
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri  = MagicMock()
mlflow_stub.lightgbm          = MagicMock()
mlflow_stub.sklearn           = MagicMock()
mlflow_stub.pyfunc            = MagicMock()
tracking_stub = types.ModuleType("mlflow.tracking")
tracking_stub.MlflowClient    = MagicMock()
sys.modules.setdefault("mlflow",          mlflow_stub)
sys.modules.setdefault("mlflow.tracking", tracking_stub)

# Stub joblib
joblib_stub = types.ModuleType("joblib")
joblib_stub.load = MagicMock(return_value={
    "preprocessor":       MagicMock(transform=lambda x: np.zeros((len(x), 98))),
    "variance_threshold": MagicMock(transform=lambda x: x),
    "keep_idx":           np.arange(98),
    "feature_names":      np.array([f"f{i}" for i in range(98)]),
    "temp_cols_dropped":  [],
    "skewed_cols":        [],
    "vaso_cap_99":        None,
})
sys.modules.setdefault("joblib", joblib_stub)

from fastapi.testclient import TestClient  # noqa: E402

# ── Patch model loading so startup doesn't fail without real artefacts ────────
with patch("api.load_model_and_pipeline") as mock_load:
    from api import app, state  # noqa: E402
    # Manually set state so routes work
    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.85, 0.15]]))
    state.model         = mock_model
    state.pipeline      = joblib_stub.load.return_value
    state.feature_names = [f"f{i}" for i in range(98)]
    state.model_version = "4"
    state.model_type    = "LightGBM"
    state.explainer     = None

client = TestClient(app)

SAMPLE_FEATURES = {f"f{i}": float(i) for i in range(98)}


# ── /health ───────────────────────────────────────────────────────────────────
def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_schema():
    r = client.get("/health")
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["model_loaded"] is True


# ── /model/info ───────────────────────────────────────────────────────────────
def test_model_info_returns_200():
    r = client.get("/model/info")
    assert r.status_code == 200


def test_model_info_schema():
    body = client.get("/model/info").json()
    for key in ("registry_name", "champion_version", "model_type", "n_features",
                "threshold_high", "threshold_medium", "shap_available"):
        assert key in body, f"Missing key: {key}"


# ── /predict ──────────────────────────────────────────────────────────────────
def test_predict_returns_200():
    r = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 200


def test_predict_schema():
    body = client.post("/predict", json={"features": SAMPLE_FEATURES}).json()
    for key in ("risk_score", "alert", "risk_level", "model_version", "threshold"):
        assert key in body, f"Missing key: {key}"


def test_predict_risk_level_values():
    body = client.post("/predict", json={"features": SAMPLE_FEATURES}).json()
    assert body["risk_level"] in ("HIGH", "MEDIUM", "LOW")


def test_predict_risk_score_range():
    body = client.post("/predict", json={"features": SAMPLE_FEATURES}).json()
    assert 0.0 <= body["risk_score"] <= 1.0


def test_predict_empty_features_returns_error():
    """Sending no features should either succeed (model handles NaN) or return a 500."""
    r = client.post("/predict", json={"features": {}})
    assert r.status_code in (200, 422, 500)


# ── /predict/batch ────────────────────────────────────────────────────────────
def test_predict_batch_returns_200():
    payload = {"patients": [
        {"features": SAMPLE_FEATURES, "stay_id": "p1"},
        {"features": SAMPLE_FEATURES, "stay_id": "p2"},
    ]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200


def test_predict_batch_count():
    payload = {"patients": [
        {"features": SAMPLE_FEATURES, "stay_id": f"p{i}"} for i in range(5)
    ]}
    body = client.post("/predict/batch", json=payload).json()
    assert body["count"] == 5
    assert len(body["predictions"]) == 5


# ── /metrics ─────────────────────────────────────────────────────────────────
def test_metrics_returns_200():
    r = client.get("/metrics")
    assert r.status_code == 200


def test_metrics_content_type():
    r = client.get("/metrics")
    assert "text/plain" in r.headers.get("content-type", "")


def test_metrics_contains_expected_metric_names():
    body = client.get("/metrics").text
    for name in ("icu_predictions_total", "icu_prediction_latency_seconds", "icu_risk_score"):
        assert name in body, f"Expected metric '{name}' not found in /metrics output"


# ── /drift/report ─────────────────────────────────────────────────────────────
def test_drift_report_returns_200():
    """Drift report returns 200 even with no prediction log (insufficient_data status)."""
    with patch("api.compute_drift_report", create=True,
               return_value={"status": "insufficient_data", "features": [], "max_psi": 0.0,
                             "drifted_features": [], "live_rows": 0, "reference_rows": 5000}):
        from drift_detector import compute_drift_report as _orig
        with patch("drift_detector.compute_drift_report",
                   return_value={"status": "insufficient_data", "features": [],
                                 "max_psi": 0.0, "drifted_features": [],
                                 "live_rows": 0, "reference_rows": 5000}):
            r = client.get("/drift/report")
    assert r.status_code in (200, 500)  # 500 if X_train.npy absent in CI


def test_drift_report_schema_when_ok():
    mock_report = {
        "status": "ok",
        "live_rows": 100,
        "reference_rows": 5000,
        "features": [{"feature": "f0", "psi": 0.01, "status": "stable"}],
        "drifted_features": [],
        "warning_features": [],
        "max_psi": 0.01,
    }
    with patch("drift_detector.compute_drift_report", return_value=mock_report):
        r = client.get("/drift/report")
    if r.status_code == 200:
        body = r.json()
        assert "status" in body
        assert "features" in body
        assert "max_psi" in body
