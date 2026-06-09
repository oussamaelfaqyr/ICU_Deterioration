"""
ICU Deterioration — FastAPI Prediction Service
================================================
Loads the champion model from the MLflow Registry and serves real-time predictions.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /predict          — score a single patient
    POST /predict/batch    — score many patients
    GET  /health           — liveness probe
    GET  /model/info       — current champion metadata
    GET  /metrics          — Prometheus metrics scrape endpoint
    GET  /drift/report     — PSI-based feature drift report
"""

import os
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import time
import threading

from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("icu_api")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("shap").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "mimic_processed"
REGISTRY_NAME = "ICU_Deterioration_Model"
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
PRED_LOG      = BASE_DIR / "prediction_log.jsonl"
PRED_LOG_MAX  = 10_000   # rolling window — older rows are trimmed

# Scoring thresholds
THRESHOLD_HIGH   = 0.20   # risk_level = HIGH
THRESHOLD_MEDIUM = 0.10   # risk_level = MEDIUM

# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────────────────────
_prom_predictions = Counter(
    "icu_predictions_total",
    "Total predictions served",
    ["risk_level", "model_version"],
)
_prom_latency = Histogram(
    "icu_prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
_prom_score = Histogram(
    "icu_risk_score",
    "Distribution of raw risk scores",
    buckets=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0],
)
_prom_model_version = Gauge(
    "icu_model_version",
    "Currently loaded champion model version (numeric)",
)
_prom_drift_psi = Gauge(
    "icu_feature_psi",
    "Population Stability Index per feature",
    ["feature"],
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    model         = None
    pipeline      = None
    feature_names = None
    explainer     = None
    model_version = None
    model_type    = None
    threshold     = THRESHOLD_HIGH

state = AppState()

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS (mirror icu_preprocessing_pipeline.py logic)
# ─────────────────────────────────────────────────────────────────────────────
GCS_COLS = [
    "gcs_motor_mean","gcs_verb_mean","gcs_eye_mean",
    "gcs_motor_delta","gcs_eye_delta",
    "gcs_motor_min","gcs_verb_min","gcs_eye_min",
    "gcs_motor_max","gcs_verb_max","gcs_eye_max",
]

def _aggregate_gcs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    motor = df.get("gcs_motor_mean", pd.Series(np.nan, index=df.index))
    verb  = df.get("gcs_verb_mean",  pd.Series(np.nan, index=df.index))
    eye   = df.get("gcs_eye_mean",   pd.Series(np.nan, index=df.index))
    df["gcs_total_mean"]  = motor + verb + eye
    df["gcs_motor_delta"] = df.get("gcs_motor_delta", pd.Series(np.nan, index=df.index))
    df["gcs_eye_delta"]   = df.get("gcs_eye_delta",   pd.Series(np.nan, index=df.index))
    drop_these = [c for c in GCS_COLS if c in df.columns]
    df = df.drop(columns=drop_these, errors="ignore")
    return df


def preprocess(raw: pd.DataFrame) -> np.ndarray:
    """Apply the same transformations as icu_preprocessing_pipeline.py."""
    pipe = state.pipeline

    df = raw.copy()

    # 1. Temporal features from intime
    if "intime" in df.columns:
        ts = pd.to_datetime(df["intime"], errors="coerce")
        df["admit_hour"]    = ts.dt.hour
        df["admit_weekday"] = ts.dt.weekday
        df["admit_month"]   = ts.dt.month
        df.drop(columns=["intime"], inplace=True, errors="ignore")

    # 2. Drop temp_c_* columns
    temp_cols = pipe.get("temp_cols_dropped", [])
    df.drop(columns=temp_cols, inplace=True, errors="ignore")

    # 3. Drop stay_id / label if present
    df.drop(columns=["stay_id", "label"], inplace=True, errors="ignore")

    # 4. Winsorize vaso_max_rate
    cap_99 = pipe.get("vaso_cap_99")
    if cap_99 is not None and "vaso_max_rate" in df.columns:
        df["vaso_max_rate"] = df["vaso_max_rate"].clip(upper=cap_99)

    # 5. log1p transform on skewed features
    for col in pipe.get("skewed_cols", []):
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # 6. GCS aggregation
    df = _aggregate_gcs(df)

    # 7. sklearn preprocessor (impute + scale)
    preprocessor = pipe["preprocessor"]
    X = preprocessor.transform(df)

    # 8. Variance threshold
    vt = pipe["variance_threshold"]
    X = vt.transform(X)

    # 9. Correlation filter
    keep_idx = pipe["keep_idx"]
    X = X[:, keep_idx]

    return X.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _get_base_model(model):
    """Extract the raw XGBoost/LGBM estimator from a CalibratedClassifierCV wrapper."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    if hasattr(model, "estimator"):
        return model.estimator
    return model


def load_model_and_pipeline():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Load preprocessing pipeline
    pipeline_path = PROCESSED_DIR / "preprocessing_pipeline.joblib"
    if not pipeline_path.exists():
        raise RuntimeError(f"Preprocessing pipeline not found at {pipeline_path}")
    state.pipeline      = joblib.load(pipeline_path)
    state.feature_names = state.pipeline["feature_names"].tolist()
    log.info(f"Loaded preprocessing pipeline ({len(state.feature_names)} features)")

    # Load champion model — resolve directly from disk to avoid path issues
    # when mlflow.db was created on Windows but the container runs on Linux.
    try:
        mv = client.get_model_version_by_alias(REGISTRY_NAME, "champion")
        state.model_version = mv.version
        state.model_type    = mv.tags.get("model_type", "Unknown")
    except Exception:
        log.warning("No 'champion' alias found, loading latest model version...")
        versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
        if not versions:
            raise RuntimeError(f"No model versions found for '{REGISTRY_NAME}'")
        mv = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        state.model_version = mv.version
        state.model_type    = mv.tags.get("model_type", "Unknown")

    log.info(f"Loading champion model: {REGISTRY_NAME} v{state.model_version} ({state.model_type})")

    # mv.source is a models:/m-<uuid> URI — find the physical folder.
    # Actual files live at: mlruns/1/models/<uuid>/artifacts/
    import re as _re
    model_dir = None

    m = _re.search(r"models:/(.+)$", mv.source or "")
    if m:
        candidate = BASE_DIR / "mlruns" / "1" / "models" / m.group(1) / "artifacts"
        if candidate.exists():
            model_dir = candidate
            log.info(f"Resolved model dir: {model_dir}")

    # Fallback: pick newest mlruns/1/models/*/artifacts that has MLmodel or model.pkl
    if model_dir is None:
        models_root = BASE_DIR / "mlruns" / "1" / "models"
        if models_root.exists():
            candidates = sorted(
                [d / "artifacts" for d in models_root.iterdir()
                 if (d / "artifacts" / "MLmodel").exists()],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if candidates:
                model_dir = candidates[0]
                log.warning(f"Fallback: using newest model dir by mtime: {model_dir}")

    if model_dir is None:
        raise RuntimeError(
            "Cannot locate model artifacts on disk. "
            "Ensure mlruns/ is included in the Docker image."
        )

    model_uri = model_dir.as_uri()
    log.info(f"Loading model from URI: {model_uri}")

    # Try native flavors (predict_proba + SHAP), fall back to pyfunc
    try:
        state.model = mlflow.lightgbm.load_model(model_uri)
        log.info("Loaded as LightGBM model")
    except Exception:
        try:
            state.model = mlflow.sklearn.load_model(model_uri)
            log.info("Loaded as sklearn model")
        except Exception:
            log.warning("Native flavor not available — falling back to pyfunc")
            state.model = mlflow.pyfunc.load_model(model_uri)


    # Build SHAP explainer on a small background set from test data
    try:
        X_test_path = PROCESSED_DIR / "X_test.npy"
        if X_test_path.exists():
            X_bg = np.load(X_test_path)[:100]
            base = _get_base_model(state.model)
            # Choose explainer by model type
            model_name = type(base).__name__
            if "Logistic" in model_name or "Linear" in model_name:
                state.explainer = shap.LinearExplainer(base, X_bg)
            else:
                state.explainer = shap.TreeExplainer(base)
            log.info(f"SHAP explainer ready ({model_name})")
    except Exception as e:
        log.warning(f"SHAP explainer not available: {e}")
        state.explainer = None


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting ICU Prediction API…")
    load_model_and_pipeline()
    log.info("API ready.")
    yield
    log.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ICU Deterioration Prediction API",
    description="Real-time ICU deterioration risk scoring with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    features: dict[str, Any]
    stay_id: Optional[Any] = None


class BatchRequest(BaseModel):
    patients: list[PatientFeatures]


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    stay_id: Optional[Any]
    risk_score: float
    alert: bool
    risk_level: str
    top_features: list[FeatureContribution]
    model_version: str
    model_type: str
    threshold: float


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _risk_level(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    if score >= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _explain(X_pp: np.ndarray) -> list[FeatureContribution]:
    """Return top-3 SHAP-driven feature contributions."""
    if state.explainer is None:
        return []
    try:
        vals = state.explainer.shap_values(X_pp)
        # For binary classifiers, vals may be [neg, pos] or just pos
        if isinstance(vals, list):
            vals = vals[1]
        sv = vals[0]                  # single sample → 1-D
        fn = state.feature_names
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]
        result = []
        for i in top3_idx:
            v = float(sv[i])
            result.append(FeatureContribution(
                feature=fn[i],
                shap_value=round(v, 4),
                direction="↑ risk" if v > 0 else "↓ risk",
            ))
        return result
    except Exception as e:
        log.warning(f"SHAP failed: {e}")
        return []


def _score_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (preprocessed array, probabilities)."""
    X_pp = preprocess(df)
    if hasattr(state.model, "predict_proba"):
        probs = state.model.predict_proba(X_pp)[:, 1]
    else:
        probs = state.model.predict(X_pp)
    return X_pp, probs


_pred_log_lock = threading.Lock()

def _log_prediction(X_pp: np.ndarray, score: float, risk_level: str) -> None:
    """Append a single prediction's feature vector to the rolling JSONL log."""
    try:
        entry = json.dumps({
            "ts": time.time(),
            "risk_level": risk_level,
            "risk_score": round(score, 5),
            "feature_vector": X_pp[0].tolist(),
        })
        with _pred_log_lock:
            # Append new entry
            with open(PRED_LOG, "a") as f:
                f.write(entry + "\n")
            # Trim to rolling window (read-rewrite only when over limit)
            try:
                with open(PRED_LOG, "r") as f:
                    lines = f.readlines()
                if len(lines) > PRED_LOG_MAX:
                    with open(PRED_LOG, "w") as f:
                        f.writelines(lines[-PRED_LOG_MAX:])
            except Exception:
                pass
    except Exception as e:
        log.debug(f"Prediction log write failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "model_version": state.model_version,
        "model_type": state.model_type,
    }


@app.get("/model/info")
def model_info():
    return {
        "registry_name": REGISTRY_NAME,
        "champion_version": state.model_version,
        "model_type": state.model_type,
        "n_features": len(state.feature_names) if state.feature_names else None,
        "feature_names": state.feature_names,
        "threshold_high": THRESHOLD_HIGH,
        "threshold_medium": THRESHOLD_MEDIUM,
        "shap_available": state.explainer is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PatientFeatures):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([req.features])
        X_pp, probs = _score_df(df)
        score = float(probs[0])
        level = _risk_level(score)
        contribs = _explain(X_pp[:1])

        # Prometheus instrumentation
        elapsed = time.perf_counter() - t0
        _prom_latency.observe(elapsed)
        _prom_score.observe(score)
        _prom_predictions.labels(
            risk_level=level,
            model_version=str(state.model_version),
        ).inc()

        # Log feature vector for drift detection
        _log_prediction(X_pp, score, level)

        return PredictionResponse(
            stay_id=req.stay_id,
            risk_score=round(score, 4),
            alert=score >= state.threshold,
            risk_level=level,
            top_features=contribs,
            model_version=str(state.model_version),
            model_type=state.model_type,
            threshold=state.threshold,
        )
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(500, str(e))


class PreprocessedRequest(BaseModel):
    features_array: list[float]
    stay_id: Optional[Any] = None

@app.post("/predict/preprocessed", response_model=PredictionResponse)
def predict_preprocessed(req: PreprocessedRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    t0 = time.perf_counter()
    try:
        X_pp = np.array([req.features_array], dtype=np.float32)
        if hasattr(state.model, "predict_proba"):
            probs = state.model.predict_proba(X_pp)[:, 1]
        else:
            probs = state.model.predict(X_pp)
            
        score = float(probs[0])
        level = _risk_level(score)
        contribs = _explain(X_pp[:1])

        elapsed = time.perf_counter() - t0
        _prom_latency.observe(elapsed)
        _prom_score.observe(score)
        _prom_predictions.labels(
            risk_level=level,
            model_version=str(state.model_version),
        ).inc()
        _log_prediction(X_pp, score, level)

        return PredictionResponse(
            stay_id=req.stay_id,
            risk_score=round(score, 4),
            alert=score >= state.threshold,
            risk_level=level,
            top_features=contribs,
            model_version=str(state.model_version),
            model_type=state.model_type,
            threshold=state.threshold,
        )
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(500, str(e))
        raise HTTPException(500, str(e))


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    results = []
    for patient in req.patients:
        t0 = time.perf_counter()
        try:
            df = pd.DataFrame([patient.features])
            X_pp, probs = _score_df(df)
            score = float(probs[0])
            level = _risk_level(score)
            contribs = _explain(X_pp[:1])

            elapsed = time.perf_counter() - t0
            _prom_latency.observe(elapsed)
            _prom_score.observe(score)
            _prom_predictions.labels(
                risk_level=level,
                model_version=str(state.model_version),
            ).inc()
            _log_prediction(X_pp, score, level)

            results.append({
                "stay_id": patient.stay_id,
                "risk_score": round(score, 4),
                "alert": score >= state.threshold,
                "risk_level": level,
                "top_features": [c.dict() for c in contribs],
                "model_version": str(state.model_version),
                "model_type": state.model_type,
                "threshold": state.threshold,
            })
        except Exception as e:
            results.append({"stay_id": patient.stay_id, "error": str(e)})
    return {"predictions": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrape endpoint — returns all registered metrics."""
    return PlainTextResponse(
        content=generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/drift/report")
def drift_report():
    """
    PSI-based feature drift report.
    Compares live prediction traffic (prediction_log.jsonl) against
    the training distribution (mimic_processed/X_train.npy).
    """
    try:
        from drift_detector import compute_drift_report
        report = compute_drift_report()
        # Update Prometheus gauges for each feature
        for feat in report.get("features", []):
            _prom_drift_psi.labels(feature=feat["feature"]).set(feat["psi"])
        return report
    except Exception as e:
        log.exception("Drift report failed")
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD  (single-page HTML served from /  )
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>ICU Deterioration Risk Dashboard</title>
<meta name="description" content="Real-time ICU patient deterioration risk scoring dashboard with AI-powered SHAP explainability"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e1a;--surface:#111827;--surface2:#1a2235;--border:#1e2d45;
  --accent:#3b82f6;--accent2:#06b6d4;--green:#10b981;--yellow:#f59e0b;--red:#ef4444;
  --text:#e2e8f0;--muted:#64748b;--font:'Inter',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}
/* ── Header ── */
header{
  background:linear-gradient(135deg,#0d1b2e 0%,#0a1628 50%,#0d1b2e 100%);
  border-bottom:1px solid var(--border);padding:1.25rem 2rem;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:50;backdrop-filter:blur(12px);
}
.logo{display:flex;align-items:center;gap:.75rem}
.logo-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:10px;display:grid;place-items:center;font-size:1.25rem}
.logo-text h1{font-size:1.1rem;font-weight:700;color:var(--text)}
.logo-text p{font-size:.72rem;color:var(--muted);font-weight:400}
.header-right{display:flex;align-items:center;gap:1rem}
.model-badge{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:.4rem .8rem;font-size:.75rem;color:var(--muted)}
.model-badge span{color:var(--accent);font-weight:600}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
/* ── Layout ── */
.container{max-width:1400px;margin:0 auto;padding:1.5rem 2rem}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem}
.grid-2{display:grid;grid-template-columns:2fr 1fr;gap:1.5rem}
/* ── Cards ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.25rem}
.card-title{font-size:.75rem;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.75rem}
/* ── Stat cards ── */
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:1.25rem;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.stat-card.blue::before{background:linear-gradient(90deg,var(--accent),var(--accent2))}
.stat-card.green::before{background:linear-gradient(90deg,var(--green),#34d399)}
.stat-card.yellow::before{background:linear-gradient(90deg,var(--yellow),#fbbf24)}
.stat-card.red::before{background:linear-gradient(90deg,var(--red),#f87171)}
.stat-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.stat-value{font-size:2rem;font-weight:800;margin:.25rem 0;line-height:1}
.stat-sub{font-size:.75rem;color:var(--muted)}
/* ── Table ── */
.table-wrap{overflow-x:auto;border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead th{padding:.65rem 1rem;text-align:left;font-size:.7rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid var(--border);
  background:var(--surface2)}
tbody tr{border-bottom:1px solid var(--border);transition:background .15s}
tbody tr:hover{background:var(--surface2)}
tbody td{padding:.7rem 1rem;vertical-align:middle}
.no-data{text-align:center;color:var(--muted);padding:2.5rem;font-size:.85rem}
/* ── Risk badges ── */
.risk-badge{display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .65rem;
  border-radius:6px;font-size:.72rem;font-weight:600;letter-spacing:.04em}
.risk-HIGH{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.risk-MEDIUM{background:rgba(245,158,11,.15);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.risk-LOW{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.3)}
.alert-dot{width:7px;height:7px;border-radius:50%}
.risk-HIGH .alert-dot{background:var(--red);box-shadow:0 0 6px var(--red)}
.risk-MEDIUM .alert-dot{background:var(--yellow)}
.risk-LOW .alert-dot{background:var(--green)}
/* ── Score bar ── */
.score-bar{height:6px;border-radius:3px;background:var(--border);overflow:hidden;min-width:80px}
.score-fill{height:100%;border-radius:3px;transition:width .5s ease}
.score-fill.high{background:linear-gradient(90deg,var(--yellow),var(--red))}
.score-fill.medium{background:linear-gradient(90deg,var(--accent),var(--yellow))}
.score-fill.low{background:linear-gradient(90deg,var(--green),var(--accent))}
/* ── SHAP bars ── */
.shap-row{display:flex;align-items:center;gap:.75rem;margin-bottom:.65rem}
.shap-name{font-size:.75rem;color:var(--text);min-width:140px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.shap-bar-wrap{flex:1;height:8px;border-radius:4px;background:var(--border);overflow:hidden;position:relative}
.shap-bar{height:100%;border-radius:4px;transition:width .4s ease}
.shap-bar.pos{background:linear-gradient(90deg,#ef4444,#f87171)}
.shap-bar.neg{background:linear-gradient(90deg,#10b981,#34d399)}
.shap-val{font-size:.72rem;color:var(--muted);min-width:52px;text-align:right}
/* ── Detail panel ── */
.detail-panel{display:none;background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:1rem;margin-top:.5rem}
.detail-panel.open{display:block;animation:fadeIn .2s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
/* ── Search & filter ── */
.toolbar{display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;flex-wrap:wrap}
.search-box{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:.5rem .9rem;color:var(--text);font-size:.82rem;outline:none;flex:1;min-width:200px}
.search-box:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(59,130,246,.2)}
.filter-btn{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:.5rem .9rem;color:var(--muted);font-size:.78rem;cursor:pointer;transition:all .15s;
  font-family:var(--font)}
.filter-btn.active{background:rgba(59,130,246,.15);border-color:var(--accent);color:var(--accent)}
.filter-btn:hover:not(.active){border-color:var(--accent);color:var(--text)}
/* ── Predict form ── */
.form-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.75rem}
.form-group label{display:block;font-size:.72rem;color:var(--muted);margin-bottom:.3rem;text-transform:uppercase;letter-spacing:.06em}
.form-group input,.form-group select{width:100%;background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;padding:.5rem .75rem;color:var(--text);font-size:.82rem;outline:none;font-family:var(--font)}
.form-group input:focus,.form-group select:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(59,130,246,.2)}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.6rem 1.25rem;border:none;border-radius:8px;
  font-family:var(--font);font-size:.82rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.btn-primary:hover{opacity:.9;transform:translateY(-1px);box-shadow:0 4px 15px rgba(59,130,246,.4)}
.btn-primary:disabled{opacity:.5;transform:none;cursor:not-allowed}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--muted)}
.btn-outline:hover{border-color:var(--accent);color:var(--text)}
/* ── Prediction result card ── */
.result-card{background:linear-gradient(135deg,var(--surface),var(--surface2));
  border:1px solid var(--border);border-radius:14px;padding:1.5rem;margin-top:1rem;display:none}
.result-card.show{display:block;animation:fadeIn .25s ease}
.result-score{font-size:3rem;font-weight:800;line-height:1}
.result-score.high{color:var(--red)}
.result-score.medium{color:var(--yellow)}
.result-score.low{color:var(--green)}
/* ── Gauge ── */
.gauge-wrap{display:flex;justify-content:center;margin:1rem 0}
.gauge{position:relative;width:140px;height:75px;overflow:hidden}
.gauge svg{width:140px;height:75px}
.gauge-label{position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  font-size:.7rem;color:var(--muted);white-space:nowrap}
/* ── Spinner ── */
.spinner{width:18px;height:18px;border:2px solid rgba(255,255,255,.3);
  border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
/* ── Responsive ── */
@media(max-width:900px){
  .grid-3{grid-template-columns:1fr 1fr}
  .grid-2{grid-template-columns:1fr}
}
@media(max-width:600px){
  .grid-3{grid-template-columns:1fr}
  header{flex-direction:column;gap:.75rem;align-items:flex-start}
  .container{padding:1rem}
}
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem}
.section-title{font-size:.95rem;font-weight:600}
.refresh-btn{background:none;border:1px solid var(--border);border-radius:7px;padding:.35rem .7rem;
  color:var(--muted);font-size:.75rem;cursor:pointer;font-family:var(--font);transition:all .15s}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}
hr{border:none;border-top:1px solid var(--border);margin:1rem 0}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🫀</div>
    <div class="logo-text">
      <h1>ICU Risk Dashboard</h1>
      <p>AI-Powered Deterioration Prediction</p>
    </div>
  </div>
  <div class="header-right">
    <div class="model-badge" id="model-badge">Model: <span id="hdr-model">loading…</span></div>
    <div class="status-dot" title="API Online"></div>
  </div>
</header>

<div class="container">

  <!-- ── Stats row ── -->
  <div class="grid-3" id="stats-row">
    <div class="stat-card blue">
      <div class="stat-label">Total Scored</div>
      <div class="stat-value" id="stat-total">–</div>
      <div class="stat-sub">predictions this session</div>
    </div>
    <div class="stat-card red">
      <div class="stat-label">High Risk Alerts</div>
      <div class="stat-value" id="stat-high">–</div>
      <div class="stat-sub" id="stat-high-pct">–% of total</div>
    </div>
    <div class="stat-card green">
      <div class="stat-label">Mean Risk Score</div>
      <div class="stat-value" id="stat-mean">–</div>
      <div class="stat-sub">calibrated probability</div>
    </div>
  </div>

  <div class="grid-2">

    <!-- ── LEFT: Patient table ── -->
    <div>
      <div class="card" style="padding-bottom:0">
        <div class="section-header">
          <span class="section-title">Patient Predictions</span>
          <button class="refresh-btn" onclick="refreshSample()">↻ Load Sample</button>
        </div>
        <div class="toolbar">
          <input class="search-box" id="search" placeholder="Search stay ID…" oninput="filterTable()"/>
          <button class="filter-btn active" id="f-all"    onclick="setFilter('ALL')">All</button>
          <button class="filter-btn" id="f-HIGH"   onclick="setFilter('HIGH')">High</button>
          <button class="filter-btn" id="f-MEDIUM" onclick="setFilter('MEDIUM')">Medium</button>
          <button class="filter-btn" id="f-LOW"    onclick="setFilter('LOW')">Low</button>
        </div>
        <div class="table-wrap">
          <table id="pred-table">
            <thead>
              <tr>
                <th>Stay ID</th>
                <th>Risk Score</th>
                <th>Level</th>
                <th>Top Driver</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody id="pred-tbody">
              <tr><td colspan="5" class="no-data">Click "Load Sample" to score patients →</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── RIGHT: Predict form + result ── -->
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div class="card">
        <div class="card-title">Score a Patient</div>
        <form id="predict-form" onsubmit="submitPredict(event)">
          <div class="form-grid" id="feature-fields">
            <!-- populated by JS -->
          </div>
          <div style="margin-top:1rem;display:flex;gap:.75rem;align-items:center">
            <button type="submit" class="btn btn-primary" id="submit-btn">
              <div class="spinner" id="submit-spinner"></div>
              <span id="submit-label">Predict Risk</span>
            </button>
            <button type="button" class="btn btn-outline" onclick="fillSample()">Fill Sample</button>
          </div>
        </form>

        <div class="result-card" id="result-card">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem">
            <div>
              <div style="font-size:.72rem;color:var(--muted);margin-bottom:.25rem;text-transform:uppercase;letter-spacing:.08em">Risk Score</div>
              <div class="result-score" id="r-score">–</div>
            </div>
            <div id="r-badge"></div>
          </div>
          <div class="gauge-wrap">
            <div class="gauge">
              <svg viewBox="0 0 140 75">
                <path d="M10 70 A60 60 0 0 1 130 70" fill="none" stroke="#1e2d45" stroke-width="12" stroke-linecap="round"/>
                <path id="gauge-arc" d="M10 70 A60 60 0 0 1 130 70" fill="none" stroke="url(#ggrad)" stroke-width="12" stroke-linecap="round"
                      stroke-dasharray="188" stroke-dashoffset="188"/>
                <defs>
                  <linearGradient id="ggrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#10b981"/>
                    <stop offset="50%" stop-color="#f59e0b"/>
                    <stop offset="100%" stop-color="#ef4444"/>
                  </linearGradient>
                </defs>
              </svg>
              <div class="gauge-label" id="gauge-label">0%</div>
            </div>
          </div>
          <hr/>
          <div style="font-size:.75rem;color:var(--muted);margin-bottom:.6rem;text-transform:uppercase;letter-spacing:.06em">Top Driving Features (SHAP)</div>
          <div id="r-shap"></div>
          <div style="margin-top:.75rem;font-size:.7rem;color:var(--muted)">
            Model: <span id="r-model" style="color:var(--accent)"></span> &nbsp;·&nbsp;
            Threshold: <span id="r-threshold"></span>
          </div>
        </div>
      </div>

      <!-- Model info card -->
      <div class="card" id="model-info-card">
        <div class="card-title">Champion Model</div>
        <div id="model-info-body" style="font-size:.8rem;color:var(--muted)">Loading…</div>
      </div>
    </div>

  </div><!-- /grid-2 -->
</div><!-- /container -->

<script>
const API = '';   // same origin

// ── State ──────────────────────────────────────────────────────────────────
let allRows = [];
let activeFilter = 'ALL';
let modelInfo = {};

// Key features to show in the quick-predict form
const QUICK_FEATURES = [
  {key:'age',        label:'Age',         default:65,   type:'number'},
  {key:'pre_icu_hours', label:'Pre-ICU hrs', default:4, type:'number'},
  {key:'los_days',   label:'LOS (days)',  default:2,    type:'number'},
  {key:'sbp_min',    label:'SBP min',     default:90,   type:'number'},
  {key:'dbp_min',    label:'DBP min',     default:55,   type:'number'},
  {key:'hr_max',     label:'HR max',      default:100,  type:'number'},
  {key:'spo2_min',   label:'SpO2 min',    default:92,   type:'number'},
  {key:'resp_max',   label:'RR max',      default:22,   type:'number'},
  {key:'temp_mean',  label:'Temp mean',   default:37.2, type:'number'},
  {key:'lab_lactate',label:'Lactate',     default:1.2,  type:'number'},
  {key:'gcs_motor_mean', label:'GCS motor', default:5, type:'number'},
  {key:'gcs_verb_mean',  label:'GCS verbal',default:4, type:'number'},
  {key:'gcs_eye_mean',   label:'GCS eye',   default:3, type:'number'},
  {key:'vaso_flag',  label:'Vasopressors',default:0,    type:'number'},
  {key:'urine_min',  label:'Urine min',   default:30,   type:'number'},
  {key:'creatinine_max', label:'Creatinine', default:1.1, type:'number'},
];

// ── Boot ────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  buildForm();
  await loadModelInfo();
  await refreshSample();
});

// ── Load model info ──────────────────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const r = await fetch(`${API}/model/info`);
    modelInfo = await r.json();
    document.getElementById('hdr-model').textContent =
      `${modelInfo.model_type || 'Unknown'} v${modelInfo.champion_version || '?'}`;
    document.getElementById('model-info-body').innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
        <div><span style="color:var(--muted)">Type</span><br/><strong>${modelInfo.model_type || '–'}</strong></div>
        <div><span style="color:var(--muted)">Version</span><br/><strong>v${modelInfo.champion_version || '–'}</strong></div>
        <div><span style="color:var(--muted)">Features</span><br/><strong>${modelInfo.n_features || '–'}</strong></div>
        <div><span style="color:var(--muted)">SHAP</span><br/><strong>${modelInfo.shap_available ? '✓ Available' : '✗ Unavailable'}</strong></div>
        <div><span style="color:var(--muted)">High threshold</span><br/><strong>≥ ${((modelInfo.threshold_high || 0.2)*100).toFixed(0)}%</strong></div>
        <div><span style="color:var(--muted)">Medium threshold</span><br/><strong>≥ ${((modelInfo.threshold_medium || 0.1)*100).toFixed(0)}%</strong></div>
      </div>`;
  } catch(e) {
    console.error('model info:', e);
    document.getElementById('hdr-model').textContent = 'Error';
  }
}

// ── Build quick-predict form ─────────────────────────────────────────────────
function buildForm() {
  const container = document.getElementById('feature-fields');
  QUICK_FEATURES.forEach(f => {
    const div = document.createElement('div');
    div.className = 'form-group';
    div.innerHTML = `<label for="f_${f.key}">${f.label}</label>
      <input id="f_${f.key}" name="${f.key}" type="${f.type}" step="any" placeholder="${f.default}" value=""/>`;
    container.appendChild(div);
  });
}

// ── Fill sample values ───────────────────────────────────────────────────────
function fillSample() {
  QUICK_FEATURES.forEach(f => {
    document.getElementById(`f_${f.key}`).value = f.default;
  });
}

// ── Submit single-patient prediction ─────────────────────────────────────────
async function submitPredict(e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  const spinner = document.getElementById('submit-spinner');
  const label = document.getElementById('submit-label');
  btn.disabled = true; spinner.style.display='block'; label.textContent='Scoring…';

  const features = {};
  QUICK_FEATURES.forEach(f => {
    const v = document.getElementById(`f_${f.key}`).value;
    if(v !== '') features[f.key] = parseFloat(v);
  });

  try {
    const res = await fetch(`${API}/predict`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({features, stay_id: 'custom'})
    });
    const data = await res.json();
    renderResult(data);
  } catch(err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    btn.disabled=false; spinner.style.display='none'; label.textContent='Predict Risk';
  }
}

// ── Render single result ─────────────────────────────────────────────────────
function renderResult(d) {
  const pct = Math.round(d.risk_score * 100);
  const lvl = d.risk_level;
  const cls = lvl === 'HIGH' ? 'high' : lvl === 'MEDIUM' ? 'medium' : 'low';

  document.getElementById('r-score').textContent = pct + '%';
  document.getElementById('r-score').className = `result-score ${cls}`;
  document.getElementById('r-badge').innerHTML = riskBadge(lvl);
  document.getElementById('gauge-label').textContent = pct + '%';
  document.getElementById('r-model').textContent = `${d.model_type} v${d.model_version}`;
  document.getElementById('r-threshold').textContent = Math.round(d.threshold*100) + '%';

  // Gauge arc
  const arc = document.getElementById('gauge-arc');
  const fill = Math.min(d.risk_score, 1) * 188;
  arc.style.strokeDashoffset = 188 - fill;

  // SHAP
  const shapEl = document.getElementById('r-shap');
  if(d.top_features && d.top_features.length) {
    const maxV = Math.max(...d.top_features.map(f => Math.abs(f.shap_value)));
    shapEl.innerHTML = d.top_features.map(f => {
      const width = maxV > 0 ? Math.round(Math.abs(f.shap_value)/maxV*100) : 0;
      const cls2 = f.shap_value > 0 ? 'pos' : 'neg';
      return `<div class="shap-row">
        <div class="shap-name" title="${f.feature}">${f.feature}</div>
        <div class="shap-bar-wrap"><div class="shap-bar ${cls2}" style="width:${width}%"></div></div>
        <div class="shap-val">${f.direction}</div>
      </div>`;
    }).join('');
  } else {
    shapEl.innerHTML = '<p style="font-size:.75rem;color:var(--muted)">SHAP not available for this model.</p>';
  }

  const rc = document.getElementById('result-card');
  rc.className = 'result-card show';
}

// ── Sample batch prediction ───────────────────────────────────────────────────
async function refreshSample() {
  // Generate synthetic sample patients
  const patients = Array.from({length:20}, (_,i) => ({
    stay_id: 3000000 + i,
    features: generateSamplePatient(i)
  }));

  try {
    const res = await fetch(`${API}/predict/batch`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({patients})
    });
    const data = await res.json();
    allRows = data.predictions || [];
    renderTable();
    updateStats();
  } catch(err) {
    console.error('Batch predict error:', err);
    document.getElementById('pred-tbody').innerHTML =
      '<tr><td colspan="5" class="no-data">API error — is the server running?</td></tr>';
  }
}

// ── Synthetic sample data generator ──────────────────────────────────────────
function generateSamplePatient(seed) {
  const rng = (min, max) => min + ((seed * 7919 + min * 31 + max * 17) % 100) / 100 * (max-min);
  return {
    age: Math.round(rng(25, 90)),
    pre_icu_hours: rng(0.5, 72),
    los_days: rng(0.5, 30),
    sbp_min: Math.round(rng(60, 140)),
    dbp_min: Math.round(rng(35, 90)),
    hr_max: Math.round(rng(55, 160)),
    spo2_min: rng(80, 100),
    resp_max: Math.round(rng(10, 40)),
    temp_mean: rng(36.0, 39.5),
    lab_lactate: rng(0.5, 8.0),
    gcs_motor_mean: rng(2, 6),
    gcs_verb_mean: rng(1, 5),
    gcs_eye_mean: rng(1, 4),
    vaso_flag: seed % 3 === 0 ? 1 : 0,
    urine_min: rng(5, 80),
    creatinine_max: rng(0.6, 5.0),
  };
}

// ── Render table ──────────────────────────────────────────────────────────────
function renderTable() {
  const search = document.getElementById('search').value.toLowerCase();
  const tbody = document.getElementById('pred-tbody');

  const rows = allRows.filter(r => {
    const matchSearch = !search || String(r.stay_id).toLowerCase().includes(search);
    const matchFilter = activeFilter === 'ALL' || r.risk_level === activeFilter;
    return matchSearch && matchFilter;
  });

  if(!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="no-data">No patients match filters.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    if(r.error) return `<tr><td>${r.stay_id}</td><td colspan="4" style="color:var(--red)">${r.error}</td></tr>`;
    const pct = Math.round(r.risk_score * 100);
    const lvl = r.risk_level;
    const cls2 = lvl === 'HIGH' ? 'high' : lvl === 'MEDIUM' ? 'medium' : 'low';
    const top = r.top_features && r.top_features[0] ? r.top_features[0].feature : '–';
    const dir = r.top_features && r.top_features[0] ? r.top_features[0].direction : '';
    return `<tr onclick="toggleDetail('${r.stay_id}')">
      <td style="font-weight:600">${r.stay_id}</td>
      <td>
        <div style="display:flex;align-items:center;gap:.6rem">
          <span style="font-weight:700;min-width:34px">${pct}%</span>
          <div class="score-bar"><div class="score-fill ${cls2}" style="width:${pct}%"></div></div>
        </div>
      </td>
      <td>${riskBadge(lvl)}</td>
      <td style="font-size:.78rem;color:var(--muted)">${top} <span style="color:${lvl==='HIGH'?'var(--red)':'var(--green)'}">${dir}</span></td>
      <td><button class="filter-btn" style="font-size:.7rem;padding:.2rem .6rem" onclick="event.stopPropagation();toggleDetail('${r.stay_id}')">▼</button></td>
    </tr>
    <tr id="detail-${r.stay_id}" style="display:none">
      <td colspan="5">
        <div class="detail-panel open">
          ${shapHtml(r)}
        </div>
      </td>
    </tr>`;
  }).join('');
}

function shapHtml(r) {
  if(!r.top_features || !r.top_features.length) return '<p style="font-size:.75rem;color:var(--muted)">No SHAP data.</p>';
  const maxV = Math.max(...r.top_features.map(f => Math.abs(f.shap_value)));
  return `<div style="font-size:.72rem;color:var(--muted);margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.06em">SHAP Feature Contributions</div>` +
    r.top_features.map(f => {
      const w = maxV > 0 ? Math.round(Math.abs(f.shap_value)/maxV*100) : 10;
      const cls2 = f.shap_value > 0 ? 'pos' : 'neg';
      return `<div class="shap-row">
        <div class="shap-name">${f.feature}</div>
        <div class="shap-bar-wrap"><div class="shap-bar ${cls2}" style="width:${w}%"></div></div>
        <div class="shap-val">${f.direction}</div>
      </div>`;
    }).join('');
}

function toggleDetail(id) {
  const row = document.getElementById(`detail-${id}`);
  if(!row) return;
  row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
}

function riskBadge(lvl) {
  return `<span class="risk-badge risk-${lvl}"><span class="alert-dot"></span>${lvl}</span>`;
}

function filterTable() { renderTable(); }
function setFilter(f) {
  activeFilter = f;
  ['ALL','HIGH','MEDIUM','LOW'].forEach(k => {
    const el = document.getElementById(`f-${k}`);
    if(el) el.className = 'filter-btn' + (k===f?' active':'');
  });
  renderTable();
}

function updateStats() {
  const total = allRows.length;
  const high  = allRows.filter(r => r.risk_level === 'HIGH').length;
  const mean  = total ? (allRows.reduce((s,r)=>s+(r.risk_score||0),0)/total) : 0;
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-high').textContent  = high;
  document.getElementById('stat-high-pct').textContent = total ? `${Math.round(high/total*100)}% of total` : '–';
  document.getElementById('stat-mean').textContent  = (mean*100).toFixed(1) + '%';
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
