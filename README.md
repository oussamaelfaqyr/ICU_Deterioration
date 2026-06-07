# 🏥 ICU Deterioration Prediction

> An end-to-end Machine Learning Operations (MLOps) pipeline designed to predict patient deterioration in the Intensive Care Unit (ICU) and empower clinicians with actionable, real-time insights.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

In the fast-paced environment of an ICU, early detection of patient deterioration can save lives. This project provides a robust, production-ready machine learning system that not only predicts deterioration risks but also provides explainable insights into *why* a prediction was made. 

By combining modern data engineering, model orchestration, and an intuitive user interface, this platform bridges the gap between complex data science and clinical application.

## 🏗️ Architecture & Tech Stack

Our solution is built on a modern, scalable MLOps stack to ensure reliability, reproducibility, and continuous improvement:

* **Experiment Tracking & Model Registry**: [MLflow](https://mlflow.org/)
* **User Interface**: [Streamlit](https://streamlit.io/) for an interactive, real-time clinical dashboard.
* **Model Explainability**: [SHAP](https://shap.readthedocs.io/) for transparent feature importance.
* **Pipeline Orchestration**: [DVC (Data Version Control)](https://dvc.org/) for tracking data versions and orchestrating the ML pipeline.
* **Containerization**: **Docker & Docker Compose** for seamless deployment across environments.
* **Monitoring**: **Prometheus & Grafana** for system health and model drift tracking.
* **Automation**: **Airflow** for automated daily retraining and **GitHub Actions** for CI/CD pipelines.

---

## 📂 Project Structure

Here's an overview of the key files and directories in this repository:

### Core Application
* **`streamlit_app.py`**: The main Streamlit dashboard application for the interactive UI.
* **`api.py`**: FastAPI backend serving real-time predictions.
* **`build_streamlit_artifacts.py`**: Script to package the latest trained model into `streamlit_artifacts/` for the dashboard.
* **`run_local.bat`**: Windows batch script to easily start the local development environment.

### Machine Learning Pipeline
* **`icu_preprocessing_pipeline.py`**: Handles data cleaning, imputation, and feature engineering.
* **`icu_train.py`**: Trains the model and logs metrics/artifacts to MLflow.
* **`drift_detector.py`**: Monitors incoming data and calculates drift metrics.
* **`promote_model.py`**: Script to promote models to production in the MLflow Model Registry.

### Configuration & Orchestration
* **`dvc.yaml` & `dvc.lock`**: DVC pipeline definitions locking the data processing and training steps.
* **`docker-compose.yml` & `Dockerfile`**: Container configurations for spinning up all services.
* **`airflow/`**: Contains Airflow DAGs for scheduling data pipelines and retraining.
* **`monitoring/`**: Configurations for Prometheus and Grafana.

### Documentation
* **`ICU_Project_Report.md`**: Comprehensive deep-dive report of the methodology, models, and findings.
* **`DVC_SETUP.md`**: Instructions for configuring Data Version Control.

---

## 🚀 Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites
* Docker and Docker Compose
* Python 3.10+
* DVC installed locally

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "ICU project"
   ```

2. **Pull the required datasets and models:**
   *(Note: This requires appropriate DVC access permissions)*
   ```bash
   dvc pull
   ```

3. **Launch the platform:**
   Use Docker Compose to spin up all necessary services in the background:
   ```bash
   docker compose up -d
   ```

---

## 🖥️ Services Overview

Once the containers are up and running, you can access the following services through your browser:

| Service | Description | Local URL |
|---------|-------------|-----------|
| **Streamlit Dashboard** | Interactive UI for patient risk assessment | [http://localhost:8501](http://localhost:8501) |
| **FastAPI Backend** | Real-time inference API | [http://localhost:8000](http://localhost:8000) |
| **MLflow Tracking** | Experiment logs and model registry | [http://localhost:5000](http://localhost:5000) |
| **Prometheus** | Metrics and monitoring data | [http://localhost:9090](http://localhost:9090) |
| **Grafana** | Visual dashboards for system/model metrics | [http://localhost:3000](http://localhost:3000) |
| **Airflow** | Workflow orchestration and scheduling | [http://localhost:8080](http://localhost:8080) |

---

## 🔄 Continuous Integration & Retraining (CI/CD)

Machine learning models degrade over time. We've built an automated lifecycle to keep the model accurate and safe:

* **Automated Testing:** Every push to the `main` branch triggers GitHub Actions to run `pytest` and verify Docker builds.
* **Continuous Deployment:** Tagged releases (e.g., `v1.x`) are automatically deployed from the GitHub Container Registry.
* **Drift Detection & Retraining:** An Airflow DAG (`icu_daily_retrain`) monitors incoming data daily. If the Population Stability Index indicates significant data drift, the system automatically triggers `dvc repro` to retrain the model.

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, improving the documentation, or adding new features to the dashboard, your help makes this project better for everyone.

---
*Built with ❤️ to improve patient outcomes through data-driven healthcare.*
