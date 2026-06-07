import os
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("export")

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()
    registry_name = "ICU_Deterioration_Model"
    out_dir = Path("streamlit_artifacts")
    out_dir.mkdir(exist_ok=True)

    # Load champion model
    try:
        mv = client.get_model_version_by_alias(registry_name, "champion")
        model_uri = f"models:/{registry_name}@champion"
    except Exception:
        log.warning("No 'champion' alias found, loading latest model version...")
        versions = client.search_model_versions(f"name='{registry_name}'")
        if not versions:
            raise RuntimeError(f"No model versions found for '{registry_name}'")
        mv = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_uri = f"models:/{registry_name}/{mv.version}"

    log.info(f"Loading champion model: {registry_name} v{mv.version}")
    
    try:
        model = mlflow.lightgbm.load_model(model_uri)
    except Exception:
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(model_uri)

    # Create inference_pipeline.joblib
    proc_pipe_path = Path("mimic_processed/preprocessing_pipeline.joblib")
    if proc_pipe_path.exists():
        pipe_dict = joblib.load(proc_pipe_path)
        from preprocessor_helper import FeatureSelector
        inference_pipeline = Pipeline([
            ("preprocessor", pipe_dict["preprocessor"]),
            ("variance_threshold", pipe_dict["variance_threshold"]),
            ("selector", FeatureSelector(pipe_dict["keep_idx"])),
            ("model", model)
        ])
        joblib.dump(inference_pipeline, out_dir / "inference_pipeline.joblib")
        # Save preprocessor dict for fallback mapping
        shutil.copy(proc_pipe_path, out_dir / "preprocessing_pipeline.joblib")
    else:
        # Just dump the model if no preprocessor
        joblib.dump(model, out_dir / "inference_pipeline.joblib")
    
    # Copy metrics
    metrics_path = Path("data preparation/mimic_processed/model_artifacts/metrics.json")
    if metrics_path.exists():
        shutil.copy(metrics_path, out_dir / "metrics.json")
        
    # Test predictions
    X_test_path = Path("mimic_processed/X_test.npy")
    y_test_path = Path("mimic_processed/y_test.npy")
    if X_test_path.exists() and y_test_path.exists():
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)
            
        df_preds = pd.DataFrame({
            "label": y_test,
            "pred_proba": probs
        })
        df_preds.to_parquet(out_dir / "test_predictions.parquet", index=False)
        log.info(f"Saved test_predictions.parquet ({len(df_preds)} rows)")

    log.info("Export complete.")

if __name__ == "__main__":
    main()
