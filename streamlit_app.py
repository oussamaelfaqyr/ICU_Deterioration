"""
ICU Deterioration Predictor — Standalone Streamlit App
======================================================
This app uses a pre-trained, lightweight Logistic Regression model 
(loaded from model_bundle.json) to perform predictions entirely 
natively in Python. 

No external FastAPI backend or heavy ML libraries are required, 
making it perfectly suited for Streamlit Community Cloud.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Predictor (Cloud)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%); }
.stat-card { background: linear-gradient(135deg, #161b2e 0%, #1a2040 100%); border: 1px solid rgba(99,179,237,0.15); border-radius: 16px; padding: 1.4rem; text-align: center; }
.stat-card .label { color: #8892a4; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; }
.stat-card .value { color: #e2e8f0; font-size: 2rem; font-weight: 700; line-height: 1; }
.badge-high   { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.5); color: #fc8181; padding: 3px 12px; border-radius: 20px; font-weight: 600; }
.badge-medium { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.5); color: #fbbf24; padding: 3px 12px; border-radius: 20px; font-weight: 600; }
.badge-low    { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.5); color: #34d399; padding: 3px 12px; border-radius: 20px; font-weight: 600; }
.risk-banner { border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0; }
.risk-high   { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.1)); border: 2px solid rgba(239,68,68,0.6); }
.risk-medium { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(180,83,9,0.1)); border: 2px solid rgba(245,158,11,0.6); }
.risk-low    { background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1)); border: 2px solid rgba(16,185,129,0.6); }
.info-box { background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.25); border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.88rem; color: #a0aec0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
COLOR_HIGH   = "#fc8181"
COLOR_MEDIUM = "#fbbf24"
COLOR_LOW    = "#34d399"
THRESHOLD_HIGH   = 0.20
THRESHOLD_MEDIUM = 0.10

@st.cache_resource
def load_model_bundle():
    bundle_path = Path("data preparation/mimic_processed/model_artifacts/model_bundle.json")
    if not bundle_path.exists():
        st.error(f"Cannot find model bundle at {bundle_path}")
        st.stop()
    with open(bundle_path, "r") as f:
        return json.load(f)

bundle = load_model_bundle()
feature_names = bundle["feature_names"]
mean = np.array(bundle["standardizer"]["mean"])
scale = np.array(bundle["standardizer"]["scale"])
weights = np.array(bundle["model"]["weights"])
bias = bundle["model"]["bias"]


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE (Native Python)
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_patient(features_dict: dict) -> float:
    # 1. Build vector matching the exact order of features
    x = np.array([features_dict.get(f, 0.0) for f in feature_names])
    # 2. Standardize
    x_scaled = (x - mean) / scale
    # 3. Logistic Regression
    logit = np.dot(x_scaled, weights) + bias
    # 4. Probability
    return float(sigmoid(logit))

def get_risk_level(score: float) -> str:
    if score >= THRESHOLD_HIGH: return "HIGH"
    if score >= THRESHOLD_MEDIUM: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def risk_gauge(score: float) -> go.Figure:
    color = COLOR_LOW
    if score >= THRESHOLD_HIGH: color = COLOR_HIGH
    elif score >= THRESHOLD_MEDIUM: color = COLOR_MEDIUM

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": "Deterioration Risk", "font": {"size": 14, "color": "#a0aec0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568", "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, THRESHOLD_MEDIUM * 100],              "color": "rgba(16,185,129,0.12)"},
                {"range": [THRESHOLD_MEDIUM * 100, THRESHOLD_HIGH * 100],  "color": "rgba(245,158,11,0.12)"},
                {"range": [THRESHOLD_HIGH * 100, 100],            "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": COLOR_HIGH, "width": 2},
                "thickness": 0.8,
                "value": THRESHOLD_HIGH * 100,
            },
        },
    ))
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#e2e8f0; font-size:1.8rem; font-weight:700;'>
        ICU Deterioration Risk Predictor (Cloud Edition)
    </h1>
    <p style='color:#718096; font-size:0.9rem;'>
        Standalone Serverless Engine · Logistic Regression Baseline · 98 Features
    </p>
    <hr/>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="info-box">
        Adjust the key clinical vitals below. The lightweight Logistic Regression model 
        calculates the deterioration risk instantly in the browser without relying on an external API.
    </div><br/>
""", unsafe_allow_html=True)

# ── Input Widgets ──
CLINICAL_FEATURES = {
    "Demographics":  {
        "age":          ("Age (years)",          18, 100, 65,   1),
    },
    "Cardiovascular": {
        "hr_max":          ("Max Heart Rate (bpm)",   40, 250, 90,   1),
        "sbp_min":         ("Min Systolic BP (mmHg)", 40, 250, 110,  1),
        "map_min":         ("Min Mean BP (mmHg)",     20, 150, 75,   1),
    },
    "Respiratory": {
        "rr_max":         ("Max Resp Rate (/min)",    5, 60,  18,   1),
        "spo2_min":       ("Min SpO2 (%)",            50, 100, 96,   1),
    },
    "Temperature & Metabolic": {
        "temp_c_max":  ("Max Temp (°C)",           32.0, 42.0, 37.0, 0.1),
        "lab_lactate": ("Lactate (mmol/L)",         0.0, 25.0,  1.2, 0.1),
        "lab_glucose": ("Glucose (mg/dL)",         50.0, 600.0, 120.0, 1.0),
    },
    "Labs": {
        "lab_wbc":       ("WBC (K/uL)",            0.0, 150.0, 10.0, 0.1),
        "lab_creatinine":("Creatinine",            0.1, 20.0,   1.0, 0.1),
        "lab_bun":       ("BUN (mg/dL)",           0.0, 200.0,  15.0, 1.0),
    },
}

user_inputs = {}
for section, feats in CLINICAL_FEATURES.items():
    with st.expander(f"**{section}**", expanded=(section in ["Cardiovascular", "Respiratory"])):
        cols = st.columns(min(len(feats), 3))
        for idx, (key, (label, mn, mx, default, step)) in enumerate(feats.items()):
            with cols[idx % len(cols)]:
                if isinstance(step, float):
                    user_inputs[key] = st.number_input(label, float(mn), float(mx), float(default), step=step, key=f"inp_{key}")
                else:
                    user_inputs[key] = st.number_input(label, int(mn), int(mx), int(default), step=step, key=f"inp_{key}")

st.markdown("<br/>", unsafe_allow_html=True)
col_btn, col_clr = st.columns([1, 5])
with col_btn:
    run = st.button("🚀 Score Patient", use_container_width=True, type="primary")

if run:
    # Build full feature vector defaults
    full_features = {f: bundle["standardizer"]["mean"][i] for i, f in enumerate(feature_names)}
    
    # Overwrite with user inputs
    for k, v in user_inputs.items():
        if k in full_features:
            full_features[k] = float(v)

    # Predict
    score = predict_patient(full_features)
    level = get_risk_level(score)

    risk_css = f"risk-{level.lower()}"
    risk_icon = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
    badge_css = f"badge-{level.lower()}"

    col_r, col_g = st.columns([1, 1])
    with col_r:
        st.markdown(f"""
            <div class="risk-banner {risk_css}">
                <div style="font-size:3rem; margin-bottom:0.3rem;">{risk_icon}</div>
                <span class="{badge_css}">{level} RISK</span>
                <div style="font-size:3.5rem; font-weight:800; color:#e2e8f0; margin:0.5rem 0;">
                    {score*100:.1f}%
                </div>
                <div style="color:#a0aec0; font-size:0.85rem;">
                    Alert: {"Yes ⚠️" if score >= THRESHOLD_HIGH else "No ✓"} &nbsp;|&nbsp;
                    Threshold: {THRESHOLD_HIGH*100:.0f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_g:
        st.plotly_chart(risk_gauge(score), use_container_width=True)
