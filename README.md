# ICU Deterioration Prediction

An end-to-end MLOps pipeline for predicting patient deterioration in the Intensive Care Unit (ICU).

## Architecture

This project is built using a modern MLOps stack:
* **DVC**: Data versioning and ML pipeline orchestration.
* **MLflow**: Experiment tracking and Model Registry.
* **FastAPI**: Real-time inference API with an integrated dashboard.
* **SHAP**: Real-time feature explainability.
* **Docker & Docker Compose**: Full containerization of all services.
* **Prometheus & Grafana**: System and model monitoring.
* **Airflow**: Automated drift-based daily retraining.
* **GitHub Actions**: Automated CI/CD pipelines.

## Getting Started

1. **Clone the repository.**
2. **Pull the data (requires DVC access):**
   ```bash
   dvc pull
   ```
3. **Start the services:**
   ```bash
   docker compose up -d
   ```

## Services

Once the stack is running, you can access the following services:
* **Inference API / Dashboard**: http://localhost:8000
* **Prometheus**: http://localhost:9090
* **Grafana**: http://localhost:3000
* **MLflow**: http://localhost:5000
* **Airflow**: http://localhost:8080

## CI/CD & Retraining

* Continuous Integration runs `pytest` and verifies Docker builds on every push to `main`.
* Continuous Deployment deploys tagged releases (`v1.x`) from the GitHub Container Registry.
* An Airflow DAG (`icu_daily_retrain`) runs daily to check for data drift (Population Stability Index) and triggers `dvc repro` if the drift exceeds safety thresholds.
