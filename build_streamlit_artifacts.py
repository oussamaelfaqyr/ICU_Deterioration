"""
build_streamlit_artifacts.py
============================
Builds the `streamlit_artifacts/` folder directly from local project files,
bypassing MLflow entirely.  Run this whenever you need to refresh the
Streamlit dashboard without re-training.

Sources used (all already on disk):
  - data preparation/mimic_processed/model_artifacts/model_bundle.json
      → logistic-regression weights + standardizer + feature names
  - mimic_processed/preprocessing_pipeline.joblib
      → sklearn ColumnTransformer (for the fallback meta dict)
  - data preparation/mimic_processed/model_artifacts/test_predictions.parquet
  - data preparation/mimic_processed/model_artifacts/metrics.json

Outputs written to streamlit_artifacts/:
  - inference_pipeline.joblib   (sklearn Pipeline: preprocessor + model)
  - preprocessing_pipeline.joblib (copy of meta dict, for feature-name fallback)
  - test_predictions.parquet
  - metrics.json
"""

import json
import shutil
import warnings
import logging
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

warnings.filterwarnings("ignore")  # suppress sklearn version warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_artifacts")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
ARTIFACTS_SRC = ROOT / "data preparation" / "mimic_processed" / "model_artifacts"
PROC_PIPE_SRC = ROOT / "mimic_processed" / "preprocessing_pipeline.joblib"
BUNDLE_JSON   = ARTIFACTS_SRC / "model_bundle.json"
TEST_PREDS    = ARTIFACTS_SRC / "test_predictions.parquet"
METRICS_SRC   = ARTIFACTS_SRC / "metrics.json"

OUT_DIR = ROOT / "streamlit_artifacts"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load model_bundle.json ─────────────────────────────────────────────────
log.info(f"Loading model bundle from {BUNDLE_JSON}")
with open(BUNDLE_JSON, encoding="utf-8") as f:
    bundle = json.load(f)

feature_names = bundle["feature_names"]
weights       = np.array(bundle["model"]["weights"], dtype=np.float64)
bias          = float(bundle["model"]["bias"])
mean_         = np.array(bundle["standardizer"]["mean"],  dtype=np.float64)
scale_        = np.array(bundle["standardizer"]["scale"], dtype=np.float64)

log.info(f"Model: logistic regression · {len(feature_names)} features")

# ── 2. Build a proper sklearn-compatible LogisticRegression wrapper ───────────
#    We create a minimal sklearn Pipeline:
#      StandardScaler (pre-fitted)  →  LogisticRegression (pre-fitted)
#
#    Both estimators are instantiated normally and then their internals are
#    overwritten so they behave as if they were fit on the training data.

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- StandardScaler (already fitted) -----------------------------------------
scaler = StandardScaler()
scaler.mean_  = mean_
scaler.scale_ = scale_
scaler.var_   = scale_ ** 2
scaler.n_features_in_ = len(feature_names)
scaler.n_samples_seen_ = 1          # dummy — required attribute
scaler.feature_names_in_ = np.array(feature_names)

# --- LogisticRegression (already fitted) -------------------------------------
clf = LogisticRegression()
clf.classes_        = np.array([0, 1])
clf.coef_           = weights.reshape(1, -1)
clf.intercept_      = np.array([bias])
clf.n_features_in_  = len(feature_names)
clf.feature_names_in_ = np.array(feature_names)

# Mark as fitted (sklearn >= 1.0 checks __sklearn_is_fitted__)
clf._sklearn_version = "1.5.0"

# --- Assemble pipeline --------------------------------------------------------
inference_pipeline = Pipeline([
    ("scaler", scaler),
    ("model",  clf),
])

# Quick sanity check
dummy = pd.DataFrame([{f: 0.0 for f in feature_names}])
prob  = inference_pipeline.predict_proba(dummy)[0, 1]
log.info(f"Sanity-check predict_proba on zeros → {prob:.4f}  (expect ~sigmoid(bias))")

# Save
pipe_out = OUT_DIR / "inference_pipeline.joblib"
joblib.dump(inference_pipeline, pipe_out)
log.info(f"Saved {pipe_out}")

# ── 3. Copy / save preprocessing_pipeline.joblib (meta dict) ─────────────────
#    streamlit_app.py uses this for feature-name fallback; we build a minimal
#    version guaranteed to work on any sklearn version.

meta_dict = {
    "feature_names": feature_names,
    "standardizer_mean":  mean_.tolist(),
    "standardizer_scale": scale_.tolist(),
}

# Try to include the real ColumnTransformer if it loads cleanly
if PROC_PIPE_SRC.exists():
    try:
        real_meta = joblib.load(PROC_PIPE_SRC)
        # Merge the real dict's extra keys (preprocessor, variance_threshold …)
        # but keep our guaranteed feature_names
        real_meta["feature_names"] = feature_names
        joblib.dump(real_meta, OUT_DIR / "preprocessing_pipeline.joblib")
        log.info(f"Copied (and patched) {PROC_PIPE_SRC} → streamlit_artifacts/")
    except Exception as e:
        log.warning(f"Could not load real preprocessing_pipeline.joblib ({e}); saving minimal meta dict instead.")
        joblib.dump(meta_dict, OUT_DIR / "preprocessing_pipeline.joblib")
else:
    joblib.dump(meta_dict, OUT_DIR / "preprocessing_pipeline.joblib")
    log.info("Saved minimal preprocessing_pipeline.joblib (no sklearn ColumnTransformer)")

# ── 4. Copy test_predictions.parquet (normalise column names) ─────────────────
# streamlit_app.py looks for columns: pred_proba, label
# The source file may use different names — we rename them here.
if TEST_PREDS.exists():
    df_tmp = pd.read_parquet(TEST_PREDS)
    # Map known source column names → canonical names expected by the app
    col_map = {
        "test_risk_score":    "pred_proba",
        "score":              "pred_proba",
        "risk_score":         "pred_proba",
        "y_pred_proba":       "pred_proba",
        "test_predicted_label": "pred_label",   # extra info, keep it
    }
    df_tmp = df_tmp.rename(columns={k: v for k, v in col_map.items() if k in df_tmp.columns})
    df_tmp.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    log.info(f"Saved test_predictions.parquet  ({len(df_tmp):,} rows)  cols={list(df_tmp.columns)}")
else:
    log.warning(f"test_predictions.parquet not found at {TEST_PREDS}")

# ── 5. Copy metrics.json ──────────────────────────────────────────────────────
if METRICS_SRC.exists():
    shutil.copy(METRICS_SRC, OUT_DIR / "metrics.json")
    log.info("Copied metrics.json")
else:
    log.warning(f"metrics.json not found at {METRICS_SRC}")

# ── Done ─────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("streamlit_artifacts/ is ready.  Run:  streamlit run streamlit_app.py")
log.info("=" * 60)
