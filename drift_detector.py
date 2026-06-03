"""
Drift Detector — Population Stability Index (PSI) based feature drift.

Uses X_train.npy as the reference distribution.
Compares against the last N rows of prediction_log.jsonl (live traffic).

PSI interpretation:
  < 0.10  → stable
  0.10-0.25 → warning
  > 0.25  → drift (trigger retraining)
"""
import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("drift_detector")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
PROCESSED_DIR  = BASE_DIR / "mimic_processed"
PRED_LOG       = BASE_DIR / "prediction_log.jsonl"

PSI_STABLE     = 0.10
PSI_WARNING    = 0.25
N_BINS         = 10
MIN_LIVE_ROWS  = 50    # need at least this many live predictions to report
REFERENCE_ROWS = 5000  # rows of X_train used as reference

_reference_data: np.ndarray | None = None
_feature_names:  list[str] | None  = None


# ── PSI calculation ──────────────────────────────────────────────────────────
def _psi_score(reference: np.ndarray, live: np.ndarray, n_bins: int = N_BINS) -> float:
    """Compute PSI for a single feature vector."""
    # Build bin edges from reference, add small eps to avoid edge issues
    eps = 1e-8
    bins = np.linspace(reference.min() - eps, reference.max() + eps, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    live_counts, _ = np.histogram(live, bins=bins)

    # Convert to proportions, replace zeros with tiny value
    ref_pct  = np.where(ref_counts == 0, eps, ref_counts / len(reference))
    live_pct = np.where(live_counts == 0, eps, live_counts / len(live))

    psi = np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
    return float(psi)


def _psi_status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_WARNING:
        return "warning"
    return "drift"


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_reference() -> tuple[np.ndarray, list[str]]:
    """Load and cache reference distribution from X_train.npy."""
    global _reference_data, _feature_names
    if _reference_data is not None:
        return _reference_data, _feature_names

    ref_path = PROCESSED_DIR / "X_train.npy"
    fn_path  = PROCESSED_DIR / "feature_names_final.csv"

    if not ref_path.exists():
        raise RuntimeError(f"Reference data not found: {ref_path}")

    data = np.load(ref_path)
    _reference_data = data[:REFERENCE_ROWS]

    if fn_path.exists():
        import csv
        with open(fn_path) as f:
            _feature_names = [row[0] for row in csv.reader(f) if row]
    else:
        _feature_names = [f"feature_{i}" for i in range(_reference_data.shape[1])]

    log.info(f"Reference loaded: {_reference_data.shape[0]} rows x {_reference_data.shape[1]} features")
    return _reference_data, _feature_names


def load_live_vectors(n_recent: int = 2000) -> np.ndarray | None:
    """Read the most recent feature vectors from prediction_log.jsonl."""
    if not PRED_LOG.exists():
        return None

    rows = []
    with open(PRED_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Take the most recent n_recent
    rows = rows[-n_recent:]
    if len(rows) < MIN_LIVE_ROWS:
        return None

    try:
        return np.array([r["feature_vector"] for r in rows], dtype=np.float32)
    except (KeyError, ValueError):
        return None


# ── Main report ──────────────────────────────────────────────────────────────
def compute_drift_report() -> dict:
    """
    Returns a drift report dict:
    {
      "status": "ok" | "warning" | "drift" | "insufficient_data",
      "live_rows": int,
      "reference_rows": int,
      "features": [
        {"feature": str, "psi": float, "status": "stable"|"warning"|"drift"},
        ...
      ],
      "drifted_features": [str, ...],
      "max_psi": float,
    }
    """
    try:
        ref, names = load_reference()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

    live = load_live_vectors()
    if live is None:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_LIVE_ROWS} live predictions to report drift.",
            "live_rows": 0,
            "reference_rows": len(ref),
            "features": [],
            "drifted_features": [],
            "max_psi": 0.0,
        }

    n_features = min(ref.shape[1], live.shape[1])
    feature_reports = []
    for i in range(n_features):
        psi = _psi_score(ref[:, i], live[:, i])
        feature_reports.append({
            "feature": names[i] if i < len(names) else f"feature_{i}",
            "psi": round(psi, 5),
            "status": _psi_status(psi),
        })

    # Sort by PSI descending so worst offenders are first
    feature_reports.sort(key=lambda x: x["psi"], reverse=True)

    drifted    = [f["feature"] for f in feature_reports if f["status"] == "drift"]
    warning    = [f["feature"] for f in feature_reports if f["status"] == "warning"]
    max_psi    = feature_reports[0]["psi"] if feature_reports else 0.0

    if drifted:
        overall = "drift"
    elif warning:
        overall = "warning"
    else:
        overall = "ok"

    return {
        "status": overall,
        "live_rows": len(live),
        "reference_rows": len(ref),
        "features": feature_reports,
        "drifted_features": drifted,
        "warning_features": warning,
        "max_psi": round(max_psi, 5),
    }
