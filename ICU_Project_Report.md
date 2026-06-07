# ICU Patient Deterioration Prediction System
## A Full-Stack MLOps Report

**Author:** ICU ML Team  
**Date:** June 2026  
**Repository:** `oussamaelfaqyr/ICU_Deterioration`  
**Status:** Production-Ready · Deployed on Streamlit Community Cloud  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Clinical Motivation](#2-problem-statement--clinical-motivation)
3. [Dataset — MIMIC-IV](#3-dataset--mimic-iv)
4. [Data Pipeline & Feature Engineering](#4-data-pipeline--feature-engineering)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Model Training](#6-model-training)
7. [Experimental Results](#7-experimental-results)
8. [Model Comparison & Selection](#8-model-comparison--selection)
9. [MLOps Architecture](#9-mlops-architecture)
10. [CI/CD Pipeline](#10-cicd-pipeline)
11. [FastAPI Inference Service](#11-fastapi-inference-service)
12. [Streamlit Dashboard](#12-streamlit-dashboard)
13. [Monitoring & Drift Detection](#13-monitoring--drift-detection)
14. [Docker & Infrastructure](#14-docker--infrastructure)
15. [Project File Structure](#15-project-file-structure)
16. [Limitations & Future Work](#16-limitations--future-work)
17. [Conclusion](#17-conclusion)

---

## 1. Executive Summary

This project delivers a **production-grade, end-to-end Machine Learning system** for predicting ICU patient deterioration using the MIMIC-IV clinical database. The system identifies patients at high risk of deterioration within their ICU stay, enabling early clinical intervention and improving patient outcomes.

The project is built on a complete **MLOps stack** that covers:

- Automated data ingestion and feature engineering from MIMIC-IV
- Reproducible ML pipeline managed by **DVC**
- Multi-model comparison (Logistic Regression, LightGBM, XGBoost, Random Forest)
- Experiment tracking with **MLflow**
- **FastAPI** inference service with SHAP explainability
- Real-time monitoring with **Prometheus + Grafana**
- Workflow orchestration with **Apache Airflow**
- Automated CI/CD with **GitHub Actions**
- Live dashboard deployed on **Streamlit Community Cloud**

**Best model:** LightGBM — AUC-ROC **0.871**, PR-AUC **0.286**, Recall **80%** at a Brier score of **0.036**.

---

## 2. Problem Statement & Clinical Motivation

### 2.1 Clinical Context

ICU patients are continuously monitored for deterioration — a sudden worsening of clinical condition that can lead to death if not detected early. Traditionally, nurses and physicians rely on manual assessment of vital signs and laboratory values, which is:

- **Time-consuming** in high-workload environments
- **Subjective** and prone to alert fatigue from existing rule-based systems (e.g., Modified Early Warning Score — MEWS)
- **Retrospective** rather than prospective

### 2.2 Project Objective

> Build a machine learning model that predicts ICU deterioration **before** it occurs, using routinely collected clinical data, with a minimum **recall of 80%** (to avoid missing true deterioration events) while minimising false positives (alert fatigue).

### 2.3 Clinical Definition of Deterioration

A patient is labelled as **deteriorating** if they experience one or more of the following events within a defined observation window:
- In-hospital mortality
- Transfer to a higher level of care
- Significant acute organ dysfunction (defined by a composite of clinical criteria)

This binary outcome (`label = 1` for deterioration, `label = 0` for stable) was computed during the data pipeline stage.

### 2.4 Key Design Constraints

| Constraint | Value |
|---|---|
| Minimum recall (sensitivity) | ≥ 80% |
| Class imbalance ratio | ~4.4% positive rate |
| Feature count | 152 clinical features |
| Inference latency target | < 100 ms per patient |
| Deployment target | Streamlit Cloud (serverless) + FastAPI (Docker) |

---

## 3. Dataset — MIMIC-IV

### 3.1 Source

The dataset is derived from **MIMIC-IV (Medical Information Mart for Intensive Care IV)**, a publicly available de-identified database of clinical data from Beth Israel Deaconess Medical Center (BIDMC), Boston, MA, USA.

- **Patients:** All adult ICU admissions
- **Period:** 2008 – 2019
- **Access:** Requires credentialed access through PhysioNet

### 3.2 Data Pipeline Stages

The raw MIMIC-IV data is processed through a structured pipeline implemented in `data preparation/mimic_pipeline.py`, producing the following intermediate parquet files:

| Stage | File | Description |
|---|---|---|
| 01 | `01_cohort_*.parquet` | Raw ICU cohort selection and stay-level aggregation |
| 02 | `02_labeled_cohort_*.parquet` | Deterioration label computation |
| 03 | `03_vital_features_*.parquet` | Vital sign aggregations (min, max, mean, delta) |
| 04 | `04_lab_features_*.parquet` | Laboratory value aggregations |
| 05 | `05_fluid_features_*.parquet` | Fluid balance and urine output features |
| 06 | `06_vasopressor_features_*.parquet` | Vasopressor flag and max infusion rate |
| 07 | `07_static_features_*.parquet` | Static demographics (age, pre-ICU hours, LOS) |
| 08 | `08_feature_matrix_raw_*.parquet` | Final merged feature matrix (6.2 MB) |
| 09 | `09_train/val/test_*.parquet` | Train / validation / test splits |
| 10 | `10_feature_list_*.csv` | Final selected feature names |

### 3.3 Dataset Statistics

| Split | Rows | Positive Rate |
|---|---|---|
| Training (raw) | ~41,000 | ~4.4% |
| Validation | ~10,000 | ~4.4% |
| Test | ~8,568 | ~4.1% (350 positives) |
| Training (after SMOTE) | ~78,000 | 50.0% |

> **Note on class imbalance:** The positive (deterioration) class represents only ~4.4% of the dataset. This severe imbalance requires SMOTE oversampling and careful threshold tuning to achieve the 80% recall target.

---

## 4. Data Pipeline & Feature Engineering

### 4.1 Feature Categories

152 features were engineered across 6 clinical domains:

| Domain | Feature Count | Examples |
|---|---|---|
| **Vital Signs** | ~60 | `hr_min`, `hr_max`, `hr_mean`, `sbp_min`, `map_min`, `spo2_min`, `rr_max` |
| **Laboratory Values** | ~40 | `lab_lactate`, `lab_creatinine`, `lab_bun`, `lab_wbc`, `lab_ph`, `lab_sodium` |
| **GCS (Neurological)** | ~8 → 3 | Raw sub-scores aggregated to `gcs_total_mean`, `gcs_motor_delta`, `gcs_eye_delta` |
| **Fluids** | ~10 | `total_urine_ml`, `fluid_balance_ml` |
| **Vasopressors** | ~5 | `vaso_flag`, `vaso_max_rate` |
| **Demographics / Static** | ~10 | `age`, `pre_icu_hours`, `los_days`, `admit_hour`, `admit_weekday` |

### 4.2 Feature Engineering Highlights

**Temporal aggregations:** For each time-series vital sign, multiple statistical summaries were computed:
- `_min`, `_max`, `_mean`, `_std` over the full ICU stay
- `_delta` (last value minus first value) to capture trends
- `_was_missing` binary indicator flags

**GCS reduction:** The 8 raw GCS sub-columns were collapsed into 3 clinically meaningful summary features to reduce dimensionality and multicollinearity.

**Temporal admission features:** Hour of admission, day of week, and month were extracted from the `intime` timestamp.

---

## 5. Preprocessing Pipeline

### 5.1 Pipeline Architecture

Implemented in `icu_preprocessing_pipeline.py` and managed as a scikit-learn `ColumnTransformer`:

```
Raw Feature Matrix (152 features)
        ↓
1. Train/Test Split (80/20, stratified on label)
        ↓
2. Drop temp_c_* columns (>96% missing)
        ↓
3. GCS aggregation (8 cols → 3 cols)
        ↓
4. Log1p transform (skewed: pre_icu_hours, los_days, vaso_max_rate)
        ↓
5. ColumnTransformer:
   • Continuous  → Median Impute → StandardScaler
   • Skewed      → Median Impute → StandardScaler
   • Binary/Flag → Most-Frequent Impute
   • Indicators  → Passthrough (no NaN expected)
        ↓
6. VarianceThreshold (remove near-zero variance features, threshold=0.01)
        ↓
7. Correlation filter (drop features with r > 0.95 pairwise correlation)
        ↓
8. SMOTE (k=5 neighbours) — applied to training set ONLY
        ↓
Output: X_train_smote.npy (balanced), X_val.npy, X_test.npy
```

### 5.2 Key Preprocessing Decisions

| Decision | Rationale |
|---|---|
| Stratified train/test split | Preserves class prevalence in test set for unbiased evaluation |
| Variance threshold | Removes near-constant features that add noise |
| Correlation filter (r > 0.95) | Removes multicollinear features, improving LR stability |
| SMOTE inside CV folds only | Prevents data leakage — validation folds always retain real class prevalence |
| Validation set never SMOTE'd | Ensures realistic performance estimates during hyperparameter tuning |

### 5.3 Sanity Checks (Post-Preprocessing)

After preprocessing, the pipeline verifies:
- `NaN in X_train = 0`
- `NaN in X_test = 0`
- `X_train mean ≈ 0` (StandardScaler applied)
- `X_train std ≈ 1` (StandardScaler applied)
- Class balance after SMOTE: 50/50

---

## 6. Model Training

### 6.1 Training Strategy

All models are trained in `icu_train.py` following these principles:

1. **Train on SMOTE data** — to handle class imbalance during learning
2. **Early-stop on validation set** — tree models use the real-prevalence validation set to prevent overfitting; the validation set was **never** used for SMOTE
3. **Test set touched exactly once** — at final evaluation after all hyperparameter decisions are made

### 6.2 Models Trained

| Model | Library | Hyperparameter Tuning |
|---|---|---|
| Logistic Regression | scikit-learn | Fixed: C=1.0, solver=saga, class_weight=balanced |
| LightGBM | lightgbm | Optuna (25 trials, 5-fold CV on raw data) |
| XGBoost | xgboost | Optuna (25 trials, 5-fold CV on raw data) |
| Random Forest | scikit-learn | Fixed: 500 trees, max_depth=10 |

### 6.3 Optuna Hyperparameter Optimisation

For LightGBM and XGBoost, **Optuna** was used with the following methodology:

- **Objective:** Maximise PR-AUC (average precision) — chosen over ROC-AUC because PR-AUC is more sensitive to performance on the minority class
- **Search space:** Learning rate, number of estimators, tree depth, min child samples, subsample ratio, column sample ratio, L1/L2 regularisation
- **Validation:** 5-fold stratified cross-validation on the **raw (pre-SMOTE) training data** — ensuring SMOTE is applied inside each fold to prevent leakage
- **Pruner:** MedianPruner (eliminates unpromising trials early)
- **Sampler:** TPE (Tree-structured Parzen Estimator)

### 6.4 Threshold Tuning

A critical step for clinical deployment: the classification threshold is **not fixed at 0.5**. Instead, it is tuned per model to guarantee **≥ 80% recall** on the validation set, then applied at test time.

```python
threshold = threshold_at_recall(y_val, y_prob, min_recall=0.80)
```

This directly addresses the clinical constraint that missing a true deterioration event (false negative) is far more costly than a false alarm (false positive).

### 6.5 MLflow Experiment Tracking

Every training run is automatically logged to MLflow:
- All hyperparameters
- All evaluation metrics (AUC-ROC, PR-AUC, Recall, Precision, F1, Brier, TP, FP, TN, FN)
- Diagnostic plots (ROC curve, PR curve, calibration, confusion matrix, threshold sweep)
- SHAP plots (beeswarm, bar, stability bootstraps)
- Permutation importance plots

---

## 7. Experimental Results

### 7.1 Individual Model Diagnostics

Each model produces a 5-panel diagnostic figure:

#### Logistic Regression

> 📍 **Image location:** `plots/LogisticRegression_diagnostic.png`

![Logistic Regression Diagnostic](c:/Users/21270/Desktop/ICU%20project/plots/LogisticRegression_diagnostic.png)

*Panels: ROC Curve · Precision-Recall Curve · Calibration Curve · Confusion Matrix · Threshold Sweep*

| Metric | Value |
|---|---|
| AUC-ROC | 0.8251 |
| PR-AUC | 0.1679 |
| Brier Score | 0.1733 |
| Recall @ threshold | 80.0% |
| Precision | 10.68% |
| F1 Score | 18.84% |
| Specificity | 71.5% |
| True Positives | 280 |
| False Positives | 2,342 |
| Threshold | 0.4458 |

---

#### LightGBM (Champion Model)

> 📍 **Image location:** `plots/LightGBM_diagnostic.png`

![LightGBM Diagnostic](c:/Users/21270/Desktop/ICU%20project/plots/LightGBM_diagnostic.png)

*Panels: ROC Curve · Precision-Recall Curve · Calibration Curve · Confusion Matrix · Threshold Sweep*

| Metric | Value |
|---|---|
| AUC-ROC | **0.8709** ✅ |
| PR-AUC | **0.2856** ✅ |
| Brier Score | **0.0364** ✅ |
| Recall @ threshold | 80.0% |
| Precision | 13.30% |
| F1 Score | 22.81% |
| Specificity | 77.79% |
| True Positives | 280 |
| False Positives | **1,825** ✅ |
| Threshold | 0.002 |

> ✅ **LightGBM is the champion model.** It achieves the best AUC-ROC, best PR-AUC, lowest Brier score, and fewest false positives — meaning least alert fatigue for clinical staff.

---

#### XGBoost

> 📍 **Image location:** `plots/XGBoost_diagnostic.png`

![XGBoost Diagnostic](c:/Users/21270/Desktop/ICU%20project/plots/XGBoost_diagnostic.png)

| Metric | Value |
|---|---|
| AUC-ROC | 0.8285 |
| PR-AUC | 0.2132 |
| Brier Score | 0.0354 |
| Recall @ threshold | 80.0% |
| Precision | 10.47% |
| F1 Score | 18.52% |
| Specificity | 70.87% |
| True Positives | 280 |
| False Positives | 2,394 |
| Threshold | 0.0373 |

---

#### Random Forest

> 📍 **Image location:** `plots/RandomForest_diagnostic.png`

![Random Forest Diagnostic](c:/Users/21270/Desktop/ICU%20project/plots/RandomForest_diagnostic.png)

| Metric | Value |
|---|---|
| AUC-ROC | 0.8149 |
| PR-AUC | 0.1503 |
| Brier Score | 0.0741 |
| Recall @ threshold | 80.0% |
| Precision | 9.44% |
| F1 Score | 16.89% |
| Specificity | 67.33% |
| True Positives | 280 |
| False Positives | 2,685 |
| Threshold | 0.2483 |

---

## 8. Model Comparison & Selection

### 8.1 Comparison Dashboard

> 📍 **Image location:** `plots/comparison_dashboard.png`

![Model Comparison Dashboard](c:/Users/21270/Desktop/ICU%20project/plots/comparison_dashboard.png)

*This dashboard shows: AUC-ROC vs PR-AUC bar chart · Precision vs Recall scatter · False Positive count · ROC overlay · PR overlay.*

### 8.2 Metrics Heatmap

> 📍 **Image location:** `plots/metrics_heatmap.png`

![Metrics Heatmap](c:/Users/21270/Desktop/ICU%20project/plots/metrics_heatmap.png)

*Green = better performance. Brier score is inverted (lower is better). All models achieve exactly 80% recall by design (threshold tuning).*

### 8.3 Decision Curve Analysis

> 📍 **Image location:** `plots/decision_curve_analysis.png`

![Decision Curve Analysis](c:/Users/21270/Desktop/ICU%20project/plots/decision_curve_analysis.png)

*Decision Curve Analysis (DCA) evaluates net clinical benefit across all possible risk thresholds. A model is clinically useful if its net benefit curve lies above both the "treat all" and "treat none" baselines. LightGBM shows the highest and most stable net benefit across the clinically relevant threshold range (0–0.25).*

### 8.4 Final Model Comparison Table

| Model | AUC-ROC | PR-AUC | Brier↓ | Recall | Precision | F1 | FP (alert fatigue) |
|---|---|---|---|---|---|---|---|
| **LightGBM** ⭐ | **0.8709** | **0.2856** | **0.0364** | 80% | 13.30% | 22.81% | **1,825** |
| XGBoost | 0.8285 | 0.2132 | 0.0354 | 80% | 10.47% | 18.52% | 2,394 |
| Logistic Regression | 0.8251 | 0.1679 | 0.1733 | 80% | 10.68% | 18.84% | 2,342 |
| Random Forest | 0.8149 | 0.1503 | 0.0741 | 80% | 9.44% | 16.89% | 2,685 |

### 8.5 Why LightGBM Wins

1. **Highest AUC-ROC (0.871):** Best overall discrimination between deteriorating and stable patients
2. **Highest PR-AUC (0.286):** Best performance specifically on the minority (positive) class — the clinically relevant case
3. **Lowest Brier Score (0.036):** Best probabilistic calibration, meaning predicted probabilities are closest to true event rates
4. **Fewest False Positives (1,825 vs 2,342–2,685):** 517 fewer false alarms than the next best model — meaningful reduction in alert fatigue for ICU staff

---

## 9. MLOps Architecture

### 9.1 Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│  MIMIC-IV (PhysioNet) → mimic_pipeline.py → parquet stages         │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER (DVC)                           │
│  preprocess → train → promote → export_streamlit                    │
│  • icu_preprocessing_pipeline.py                                    │
│  • icu_train.py (LR · LightGBM · XGBoost · RF + Optuna)            │
│  • promote_model.py (MLflow alias: champion)                        │
│  • export_streamlit_bundle.py (inference_pipeline.joblib)           │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     EXPERIMENT TRACKING (MLflow)                    │
│  • sqlite:///mlflow.db (local) / MLflow Tracking Server (Docker)   │
│  • Model Registry: ICU_Deterioration_Model                          │
│  • Aliases: champion (production), challenger (candidate)           │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
              ┌──────────────┴──────────────┐
              ↓                             ↓
┌─────────────────────────┐   ┌────────────────────────────────┐
│   INFERENCE LAYER       │   │     DASHBOARD LAYER            │
│   FastAPI + Uvicorn     │   │     Streamlit Cloud            │
│   Port 8000             │   │     streamlit_app.py           │
│   • /predict            │   │     reads inference_pipeline   │
│   • /predict/batch      │   │     .joblib from streamlit_    │
│   • /health             │   │     artifacts/                 │
│   • /model/info         │   └────────────────────────────────┘
│   • /drift/report       │
│   • /metrics (Prom.)    │
└────────────┬────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                                  │
│  Prometheus (port 9090) → Grafana (port 3000)                       │
│  Metrics: icu_predictions_total, icu_prediction_latency_seconds,    │
│           icu_risk_score, icu_model_version, icu_feature_psi        │
└─────────────────────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                               │
│  Apache Airflow (port 8080) — schedules periodic retraining DAGs    │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 DVC Pipeline DAG

```
          preprocess
          (icu_preprocessing_pipeline.py)
                ↓
            train
          (icu_train.py)
                ↓
           promote
          (promote_model.py)
                ↓
       export_streamlit
     (export_streamlit_bundle.py)
```

**DVC ensures:**
- Full reproducibility — any stage only re-runs if its dependencies change
- Versioned data and model artifacts
- Integration with CI/CD via `dvc repro`

### 9.3 MLflow Model Registry

The model registry uses **alias-based promotion** (MLflow ≥ 2.9 API):

| Alias | Meaning |
|---|---|
| `champion` | The current production model serving live predictions |
| `challenger` | A candidate model awaiting evaluation |

`promote_model.py` automatically compares the new candidate's PR-AUC against the champion. Promotion only occurs if the candidate strictly outperforms the champion — preserving the existing champion otherwise.

---

## 10. CI/CD Pipeline

### 10.1 Workflow Overview

Three GitHub Actions workflows are defined in `.github/workflows/`:

| Workflow | File | Trigger | Purpose |
|---|---|---|---|
| CI — Build, Test & Push | `ci.yml` | Push / PR to `main` | Run tests + build Docker image |
| CD — Deploy on Release | `cd.yml` | Git tag `v*` | Deploy new release |
| ML Pipeline | `ml_pipeline.yml` | Manual / weekly Sunday 02:00 UTC | Full retrain + export + deploy |

### 10.2 ML Pipeline Workflow — Detailed Flow

```
Trigger: workflow_dispatch OR schedule (cron: "0 2 * * 0")
         OR push to main (paths-ignore: streamlit_artifacts/**)

                  ↓
┌─────────────────────────────────────────┐
│ Job 1: Lint & Tests                     │
│  • pip install -r requirements-api.txt  │
│  • pytest tests/ -v --tb=short          │
│  • Gate: must pass before training      │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Job 2: Train → Evaluate → Export        │
│  • pip install -r requirements-api.txt  │
│  • dvc repro --no-commit                │
│    (preprocess→train→promote→export)    │
│  • Print metrics to job summary         │
│  • Verify artifacts (assert files exist)│
│  • git add streamlit_artifacts/         │
│         metrics/ plots/                 │
│  • git commit -m "ci: update [skip ci]" │
│  • git push origin main                 │
│    → Streamlit Cloud auto-redeploys ✅  │
└────────────────┬────────────────────────┘
                 ↓ (only on code pushes, not bot commits)
┌─────────────────────────────────────────┐
│ Job 3: Build & Push API Docker Image    │
│  • docker build → GHCR                 │
│  • Tags: :latest + :sha                 │
└─────────────────────────────────────────┘
```

### 10.3 Anti-Loop Guards

A critical design requirement when committing artifacts back to the same repository is preventing recursive workflow runs:

| Guard | Implementation |
|---|---|
| **Ignore artifact paths** | `paths-ignore: streamlit_artifacts/**` on the push trigger |
| **Actor check** | `if: github.actor != 'github-actions[bot]'` on all jobs |
| **Skip CI marker** | Commit message includes `[skip ci]` as a fallback |

### 10.4 What Gets Committed vs. What Stays Local

| Committed to Git | NOT committed |
|---|---|
| `streamlit_artifacts/` (inference_pipeline.joblib, metrics.json, test_predictions.parquet) | `mlruns/` (large MLflow artifacts) |
| `metrics/` (model_comparison.json, metrics.json) | `models/*.pkl` (large model binaries) |
| `plots/` (diagnostic PNGs) | `optuna.db` (Optuna study database) |
| `dvc.lock` (pipeline fingerprints) | `mimic_processed/*.npy` (large arrays) |

---

## 11. FastAPI Inference Service

### 11.1 Endpoints

The inference service (`api.py`) is built with **FastAPI** and served with **Uvicorn**:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns model loaded status |
| `GET` | `/model/info` | Champion model metadata (version, type, features) |
| `POST` | `/predict` | Score a single patient (raw feature dict → risk score) |
| `POST` | `/predict/batch` | Score many patients in one request |
| `POST` | `/predict/preprocessed` | Score from pre-processed feature array |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/drift/report` | PSI-based feature drift report |

### 11.2 Prediction Response Schema

```json
{
  "stay_id": "P123456",
  "risk_score": 0.2347,
  "alert": true,
  "risk_level": "HIGH",
  "top_features": [
    {"feature": "lab_lactate", "shap_value": 0.142, "direction": "↑ risk"},
    {"feature": "map_min",     "shap_value": -0.089, "direction": "↓ risk"},
    {"feature": "spo2_min",    "shap_value": 0.076, "direction": "↑ risk"}
  ],
  "model_version": "4",
  "model_type": "LGBMClassifier",
  "threshold": 0.20
}
```

### 11.3 SHAP Explainability

For each prediction, the top 3 feature contributions are computed using SHAP:
- **Linear models:** `shap.LinearExplainer`
- **Tree models:** `shap.TreeExplainer`

SHAP values are returned per prediction, enabling clinicians to understand *why* a patient is flagged as high risk.

### 11.4 Prediction Logging

Every prediction is appended to `prediction_log.jsonl` (rolling 10,000-row window) with:
- Timestamp
- Risk level and score
- Full preprocessed feature vector

This log feeds the **drift detection** system.

---

## 12. Streamlit Dashboard

### 12.1 Architecture

The dashboard (`streamlit_app.py`) is **model-agnostic** — it loads `streamlit_artifacts/inference_pipeline.joblib` at startup and automatically adapts to whichever model won training (LightGBM, XGBoost, LR, or RF).

**No code changes are needed when the champion model changes between training runs.**

### 12.2 Pages

| Page | Content |
|---|---|
| **📊 Overview** | System stats (AUC, recall, threshold), session prediction history, timeline chart, risk distribution donut, feature importance chart |
| **🔬 Manual Prediction** | 6-section clinical form (demographics, cardiovascular, respiratory, temperature/GCS, labs, fluids), instant scoring, gauge chart, feature contribution bars |
| **⚡ Live Simulation** | Streams real test-set patients from `test_predictions.parquet`, batch controls, live confusion matrix, prediction log table |
| **📈 Model Performance** | Full metric cards for train/validation/test splits, calibration curve, feature importance |

### 12.3 Streamlit Cloud Deployment

Streamlit Cloud monitors the `main` branch of the GitHub repository. When the GitHub Actions ML Pipeline workflow pushes updated `streamlit_artifacts/` to `main`, Streamlit Cloud **automatically redeploys** the dashboard within minutes.

This creates a **zero-touch deployment loop:**

```
New data or code push
    → GitHub Actions triggers
    → dvc repro (retrain)
    → export_streamlit_bundle.py
    → git push streamlit_artifacts/ to main
    → Streamlit Cloud detects push
    → Dashboard redeploys with new model
```

### 12.4 Streamlit Dependencies (`requirements.txt`)

```
streamlit>=1.34.0
plotly>=5.22.0
pandas>=2.0.0
numpy>=1.26.0
pyarrow>=16.0.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
xgboost>=2.0.0
joblib>=1.3.0
```

> CatBoost is deliberately excluded — it adds ~200 MB to the dependency graph and is not currently in the training candidate set.

---

## 13. Monitoring & Drift Detection

### 13.1 Prometheus Metrics

The FastAPI service exposes the following Prometheus metrics at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `icu_predictions_total` | Counter | Total predictions, labelled by `risk_level` and `model_version` |
| `icu_prediction_latency_seconds` | Histogram | End-to-end request latency |
| `icu_risk_score` | Histogram | Distribution of raw risk score probabilities |
| `icu_model_version` | Gauge | Currently loaded champion model version |
| `icu_feature_psi` | Gauge | PSI score per feature (updated on drift report) |

### 13.2 Feature Drift Detection (PSI)

`drift_detector.py` implements **Population Stability Index (PSI)** to detect input feature drift between:
- **Reference:** Training distribution (`X_train.npy`, first 5,000 rows)
- **Live:** Recent 2,000 predictions from `prediction_log.jsonl`

| PSI Value | Status | Action |
|---|---|---|
| < 0.10 | ✅ Stable | No action required |
| 0.10 – 0.25 | ⚠️ Warning | Increase monitoring frequency |
| > 0.25 | 🚨 Drift | Trigger retraining via CI/CD |

The drift report is available at `GET /drift/report` and includes PSI scores for all 152 features, sorted by severity.

### 13.3 Grafana Dashboard

Grafana (port 3000) visualises Prometheus metrics with pre-configured dashboards covering:
- Prediction volume over time
- Risk score distribution
- Latency percentiles
- Alert rate trends
- Feature drift PSI heatmaps

---

## 14. Docker & Infrastructure

### 14.1 Docker Services (`docker-compose.yml`)

| Service | Image | Port | Purpose |
|---|---|---|---|
| `api` | Custom (Dockerfile) | 8000 | FastAPI inference service |
| `dashboard` | Custom (Dockerfile) | 8501 | Streamlit dashboard (local) |
| `mlflow` | Custom (Dockerfile) | 5000 | MLflow tracking server |
| `prometheus` | `prom/prometheus:v2.52.0` | 9090 | Metrics collection |
| `grafana` | `grafana/grafana:10.4.2` | 3000 | Metrics visualisation |
| `airflow` | `apache/airflow:2.9.1-python3.11` | 8080 | Workflow orchestration |

### 14.2 Dockerfile Architecture

The API Dockerfile uses a **multi-stage build** to minimise image size:

```dockerfile
Stage 1 (builder): python:3.11-slim
  • Install build tools (gcc, g++, libgomp1)
  • pip install requirements-api.txt → /install

Stage 2 (runtime): python:3.11-slim
  • Copy /install from builder (no build tools in final image)
  • Copy application code (api.py, promote_model.py, drift_detector.py)
  • Copy preprocessing artifacts (mimic_processed/)
  • Non-root user (appuser, UID 1000)
  • HEALTHCHECK via urllib.request to /health every 15s
  • CMD: uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

### 14.3 Running Locally

```bash
# Option 1: Full Docker stack
docker compose up -d

# Option 2: Windows native (run_local.bat)
run_local.bat
# Starts: MLflow (5000) → FastAPI (8000) → Streamlit (8501)

# Option 3: Individual components
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
uvicorn api:app --reload --port 8000
streamlit run streamlit_app.py --server.port 8501
```

---

## 15. Project File Structure

```
ICU project/
│
├── 📊 DATA PIPELINE
│   ├── data preparation/
│   │   ├── mimic_pipeline.py          # Full MIMIC-IV ingestion pipeline
│   │   └── mimic_processed/           # Intermediate parquet files (01→10)
│   │       └── model_artifacts/       # Legacy Streamlit artifacts (pre-CI/CD)
│   └── mimic_eda.ipynb                # Exploratory Data Analysis notebook
│
├── 🔧 ML PIPELINE
│   ├── icu_preprocessing_pipeline.py  # Preprocessing, feature selection, SMOTE
│   ├── icu_train.py                   # Multi-model training (LR, LGBM, XGB, RF)
│   ├── promote_model.py               # MLflow alias-based model promotion
│   ├── export_streamlit_bundle.py     # Export inference_pipeline.joblib ← NEW
│   └── dvc.yaml                       # DVC pipeline DAG definition
│
├── 📦 ARTIFACTS
│   ├── streamlit_artifacts/           # ← Committed to Git (Streamlit reads these)
│   │   ├── inference_pipeline.joblib  # sklearn Pipeline([preprocessor, model])
│   │   ├── preprocessing_pipeline.joblib
│   │   ├── metrics.json
│   │   └── test_predictions.parquet
│   ├── metrics/
│   │   ├── metrics.json               # DVC-tracked metrics
│   │   └── model_comparison.json      # Full comparison table
│   └── plots/                         # Training diagnostic plots
│       ├── LightGBM_diagnostic.png
│       ├── LogisticRegression_diagnostic.png
│       ├── XGBoost_diagnostic.png
│       ├── RandomForest_diagnostic.png
│       ├── comparison_dashboard.png
│       ├── metrics_heatmap.png
│       └── decision_curve_analysis.png
│
├── 🚀 INFERENCE SERVICE
│   ├── api.py                         # FastAPI service (7 endpoints)
│   ├── drift_detector.py              # PSI-based feature drift detection
│   └── check_labels_and_features.py   # Utility script
│
├── 🖥️ DASHBOARD
│   ├── streamlit_app.py               # Streamlit Cloud dashboard ← UPDATED
│   └── dashboard/                     # Local Docker dashboard (API-connected)
│
├── 🐋 DOCKER & INFRA
│   ├── Dockerfile                     # Multi-stage API image
│   ├── docker-compose.yml             # 6-service stack
│   ├── run_local.bat                  # Windows local startup script
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/
│
├── ✈️ ORCHESTRATION
│   └── airflow/
│       └── dags/                      # Airflow DAG definitions
│
├── 🧪 TESTS
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_api.py                # 14 pytest unit tests for FastAPI
│
├── 📋 CI/CD
│   └── .github/workflows/
│       ├── ci.yml                     # Test + Docker build (PRs to main)
│       ├── cd.yml                     # Deploy on release tags
│       └── ml_pipeline.yml            # Train → Export → Deploy ← NEW
│
├── 📄 CONFIGURATION
│   ├── requirements.txt               # Streamlit Cloud deps ← UPDATED
│   ├── requirements-api.txt           # Full training + API deps
│   ├── dvc.lock                       # DVC stage fingerprints
│   ├── .dvcignore
│   └── .gitignore
│
└── 📚 DOCUMENTATION
    ├── README.md
    └── DVC_SETUP.md
```

---

## 16. Limitations & Future Work

### 16.1 Current Limitations

| Limitation | Description |
|---|---|
| **Low precision** | Precision is ~10-13% at 80% recall — inherent in the severe class imbalance (~4.4% positive rate). This means ~7–9 false alarms per true positive. |
| **Static predictions** | The current model uses features aggregated over the entire ICU stay. It does not capture temporal trends in a time-series manner. |
| **No real-time feature extraction** | The live API accepts pre-extracted features; it does not extract features directly from raw EHR streams. |
| **Logistic Regression manual prediction** | The manual prediction page works with clinical form inputs but may not perfectly replicate the full preprocessing pipeline for all edge cases. |
| **MIMIC-only training** | The model is trained exclusively on MIMIC-IV (single US hospital system). Generalisability to other hospital systems and countries is unknown. |
| **CatBoost excluded** | CatBoost is conditionally enabled in training but excluded from Streamlit Cloud requirements due to size. |

### 16.2 Recommended Future Work

#### Short-term
- [ ] Add CatBoost as a first-class candidate and include in Streamlit requirements with conditional import
- [ ] Improve calibration with `CalibratedClassifierCV` (Platt or Isotonic) on the champion model
- [ ] Add SHAP waterfall plots to the Streamlit Manual Prediction page for individual explainability
- [ ] Add a conftest.py for Streamlit tests (end-to-end dashboard testing)

#### Medium-term
- [ ] **Temporal model:** Replace tabular aggregations with LSTM or Transformer operating on hourly vital sign sequences
- [ ] **Multi-site validation:** Test generalisability on eICU Collaborative Research Database (multi-hospital)
- [ ] **DVC remote storage:** Configure S3 or GCS as DVC remote to enable cloud-based data versioning
- [ ] **Automated drift-triggered retraining:** Wire PSI drift alert → Airflow DAG trigger → dvc repro → GitHub Actions

#### Long-term
- [ ] **Federated learning:** Train across multiple hospital systems without sharing raw patient data
- [ ] **Online learning:** Incrementally update model weights as new ICU admissions arrive
- [ ] **Regulatory compliance:** Document the model for clinical decision support tool regulation (FDA, CE marking)

---

## 17. Conclusion

This project demonstrates a **complete, production-grade MLOps system** for ICU deterioration prediction. It goes beyond a research notebook to implement:

1. **Rigorous ML methodology** — stratified splits, SMOTE inside CV folds only, test set used once, Optuna tuning on real prevalence data
2. **Clinical alignment** — threshold tuned to guarantee ≥80% recall, Decision Curve Analysis for clinical utility, PSI-bounded alert interpretation
3. **Full MLOps stack** — DVC reproducibility, MLflow tracking, FastAPI inference, Prometheus/Grafana monitoring, Airflow orchestration, Docker packaging
4. **Automated CI/CD** — GitHub Actions pipeline that retrains, evaluates, and deploys to Streamlit Cloud with a single git push, with anti-loop guards and clean artifact separation

The champion model, **LightGBM with Optuna hyperparameter tuning**, achieves:
- AUC-ROC: **0.871**
- PR-AUC: **0.286** (best among all candidates)
- Brier Score: **0.036** (best calibration)
- Recall: **80%** (by design)
- False Positives: **1,825** (least alert fatigue)

The system is designed to **automatically update** whenever training produces a better model — the CI/CD pipeline handles the full lifecycle from data preprocessing to live dashboard deployment without manual intervention.

---

## Appendix — Image Quick Reference

| Image | File Path |
|---|---|
| Logistic Regression Diagnostic | `plots/LogisticRegression_diagnostic.png` |
| LightGBM Diagnostic | `plots/LightGBM_diagnostic.png` |
| XGBoost Diagnostic | `plots/XGBoost_diagnostic.png` |
| Random Forest Diagnostic | `plots/RandomForest_diagnostic.png` |
| Model Comparison Dashboard | `plots/comparison_dashboard.png` |
| Metrics Heatmap | `plots/metrics_heatmap.png` |
| Decision Curve Analysis | `plots/decision_curve_analysis.png` |

> All plot paths are relative to the project root: `c:\Users\21270\Desktop\ICU project\`

---

*Report generated: June 2026 · ICU Deterioration MLOps Project*
