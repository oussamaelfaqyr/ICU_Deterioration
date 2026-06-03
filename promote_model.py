"""
Automated Model Promotion Script
Evaluates the newly registered model against the current champion (alias-based)
and promotes it if it performs better on PR-AUC.

Uses MLflow aliases instead of deprecated stages (compatible with MLflow >= 2.9.0).
  - champion  →  the live / Production model
  - challenger →  latest candidate waiting for evaluation
"""
import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("promote_model")

REGISTRY_NAME = "ICU_Deterioration_Model"
METRIC_KEY    = "pr_auc"
MARGIN        = 0.0          # require candidate to beat champion by at least this


def get_pr_auc(version) -> float | None:
    """Extract pr_auc from model version tags, falling back to the run metrics."""
    tag_val = version.tags.get(METRIC_KEY)
    if tag_val is not None:
        return float(tag_val)
    # Fallback: read from the MLflow run directly
    try:
        client = MlflowClient()
        run = client.get_run(version.run_id)
        return run.data.metrics.get(METRIC_KEY)
    except Exception:
        return None


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()

    # ── Fetch all model versions ───────────────────────────────────────────────
    try:
        versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    except Exception as e:
        log.error(f"Failed to query model registry: {e}")
        return

    if not versions:
        log.warning(f"No model versions found for '{REGISTRY_NAME}'")
        return

    # Sort newest → oldest
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    # ── Identify champion and candidate ───────────────────────────────────────
    # Champion = model with alias "champion" (new API equivalent of "Production")
    # Bootstrap: if no alias exists yet, check for legacy "Production" stage versions
    try:
        champion_mv = client.get_model_version_by_alias(REGISTRY_NAME, "champion")
        champion_version = champion_mv.version
        log.info(f"Found existing champion alias -> v{champion_version}")
    except Exception:
        champion_mv = None
        champion_version = None
        # Legacy fallback: promote any version in the old 'Production' stage to alias
        prod_versions = [v for v in versions if getattr(v, 'current_stage', '') == "Production"]
        if prod_versions:
            legacy = max(prod_versions, key=lambda v: int(v.version))
            log.info(f"No champion alias found. Bootstrapping from legacy Production stage v{legacy.version}")
            client.set_registered_model_alias(REGISTRY_NAME, "champion", legacy.version)
            champion_mv = legacy
            champion_version = legacy.version

    # Candidate = always the newest registered version
    candidate = versions[0]  # sorted newest -> oldest above

    # Exit gracefully if newest version is already the champion
    if candidate.version == champion_version:
        log.info(f"Newest version (v{candidate.version}) is already the champion. Nothing to do.")
        return

    cand_pr_auc = get_pr_auc(candidate)
    if cand_pr_auc is None:
        log.warning(f"Candidate v{candidate.version} has no '{METRIC_KEY}' metric. Cannot evaluate.")
        return

    log.info(f"Candidate  (v{candidate.version}) PR-AUC: {cand_pr_auc:.4f}")

    # ── Compare and promote ───────────────────────────────────────────────────
    if champion_mv is not None:
        champ_pr_auc = get_pr_auc(champion_mv)
        champ_pr_auc = champ_pr_auc if champ_pr_auc is not None else 0.0
        log.info(f"Champion   (v{champion_version})  PR-AUC: {champ_pr_auc:.4f}")

        if cand_pr_auc > champ_pr_auc + MARGIN:
            log.info(f"Candidate beats champion by {cand_pr_auc - champ_pr_auc:.4f}. Promoting...")
            # Remove old champion alias, set new one
            client.delete_registered_model_alias(REGISTRY_NAME, "champion")
            client.set_registered_model_alias(REGISTRY_NAME, "champion", candidate.version)
            log.info(f"[OK] v{candidate.version} is now the champion.")
        else:
            log.info("Candidate does not outperform champion. Keeping current champion.")
    else:
        log.info("No champion found. Promoting candidate as first champion...")
        client.set_registered_model_alias(REGISTRY_NAME, "champion", candidate.version)
        log.info(f"[OK] v{candidate.version} is now the champion.")


if __name__ == "__main__":
    main()
