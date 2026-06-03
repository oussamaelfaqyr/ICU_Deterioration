"""
Airflow DAG: ICU Daily Retraining Pipeline
==========================================
Runs daily at 02:00 UTC.

Flow:
  check_drift --> maybe_retrain --> [dvc_repro | skip_retrain] --> promote --> notify

Requires:
  - ICU_PROJECT_DIR env var pointing to the project root
  - MLFLOW_TRACKING_URI env var (e.g. http://mlflow:5000)
  - The API service running at ICU_API_URL (default: http://api:8000)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

log = logging.getLogger("icu_retrain_dag")

# ── Config ────────────────────────────────────────────────────────────────────
ICU_PROJECT_DIR = os.getenv("ICU_PROJECT_DIR", "/opt/icu_project")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ICU_API_URL     = os.getenv("ICU_API_URL", "http://api:8000")
PSI_DRIFT_THRESHOLD = 0.25   # any feature above this triggers retraining


# ── Tasks ─────────────────────────────────────────────────────────────────────
def _check_drift(**context) -> str:
    """
    Calls /drift/report on the live API.
    Returns the branch task_id based on drift status.
    """
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(f"{ICU_API_URL}/drift/report", timeout=30) as resp:
            report = json.loads(resp.read())
    except urllib.error.URLError as e:
        log.warning(f"Could not reach API for drift report: {e}. Proceeding with retrain as precaution.")
        return "dvc_repro"

    status   = report.get("status", "unknown")
    max_psi  = report.get("max_psi", 0.0)
    drifted  = report.get("drifted_features", [])

    log.info(f"Drift report: status={status}, max_psi={max_psi:.4f}, drifted={drifted}")

    # Push drift info to XCom for the notify task
    context["ti"].xcom_push(key="drift_report", value=report)

    if status == "insufficient_data":
        log.info("Insufficient live data for drift detection. Skipping retrain.")
        return "skip_retrain"

    if status in ("drift",) or max_psi > PSI_DRIFT_THRESHOLD:
        log.info(f"Drift detected ({len(drifted)} features). Triggering retraining.")
        return "dvc_repro"

    log.info("No significant drift detected. Skipping retrain.")
    return "skip_retrain"


def _notify(**context) -> None:
    """Log a summary of the pipeline run."""
    ti = context["ti"]
    drift_report = ti.xcom_pull(key="drift_report", task_ids="check_drift") or {}

    status  = drift_report.get("status", "unknown")
    max_psi = drift_report.get("max_psi", 0.0)
    drifted = drift_report.get("drifted_features", [])

    summary = (
        f"\n{'='*60}\n"
        f"ICU Retrain DAG run complete\n"
        f"  Logical date : {context['logical_date']}\n"
        f"  Drift status : {status}\n"
        f"  Max PSI      : {max_psi:.4f}\n"
        f"  Drifted feats: {', '.join(drifted) if drifted else 'none'}\n"
        f"{'='*60}"
    )
    log.info(summary)
    # Extend here: send to Slack / email / PagerDuty
    # e.g. requests.post(SLACK_WEBHOOK, json={"text": summary})


# ── DAG definition ────────────────────────────────────────────────────────────
default_args = {
    "owner":            "mlops",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

with DAG(
    dag_id="icu_daily_retrain",
    default_args=default_args,
    description="Daily drift check and conditional DVC retraining for ICU model",
    schedule_interval="0 2 * * *",    # 02:00 UTC every day
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["icu", "mlops", "retraining"],
) as dag:

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=_check_drift,
        provide_context=True,
    )

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command=(
            f"cd {ICU_PROJECT_DIR} && "
            f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
            "dvc repro train promote"
        ),
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "MLFLOW_TRACKING_URI": MLFLOW_URI,
        },
        append_env=True,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    # Promote runs inside dvc_repro (via dvc.yaml promote stage),
    # but we expose it as a standalone task too for manual triggers.
    promote = BashOperator(
        task_id="promote",
        bash_command=(
            f"cd {ICU_PROJECT_DIR} && "
            f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
            "python promote_model.py"
        ),
        trigger_rule="none_failed_min_one_success",
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        provide_context=True,
        trigger_rule="all_done",   # always runs
    )

    # ── Wire up ──────────────────────────────────────────────────────────────
    check_drift >> [dvc_repro, skip_retrain]
    dvc_repro   >> promote
    skip_retrain >> notify
    promote      >> notify
