"""
ICU Deterioration Predictor — Professional Streamlit Dashboard
==============================================================
Sections:
  1. Sidebar       : API status + navigation
  2. Overview      : System metrics, model info, live KPIs
  3. Manual Predict: Clinical feature inputs -> real-time prediction
  4. Live Simulation: Stream X_test rows through the API, show live metrics
  5. Drift Monitor : Call /drift/report and visualise PSI per feature
  6. History       : Rolling prediction log table + risk distribution chart

Run:
    uvicorn api:app --port 8000            # terminal 1
    streamlit run dashboard/app.py         # terminal 2
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Deterioration Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%);
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Stat cards ── */
.stat-card {
    background: linear-gradient(135deg, #161b2e 0%, #1a2040 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.stat-card:hover { border-color: rgba(99,179,237,0.4); transform: translateY(-2px); }
.stat-card .label { color: #8892a4; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; }
.stat-card .value { color: #e2e8f0; font-size: 2rem; font-weight: 700; line-height: 1; }
.stat-card .sub   { color: #63b3ed; font-size: 0.75rem; margin-top: 0.3rem; }

/* ── Alert badges ── */
.badge-high   { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.5); color: #fc8181; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-medium { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.5); color: #fbbf24; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-low    { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.5); color: #34d399; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* ── Risk result banner ── */
.risk-banner {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    animation: fadeIn 0.5s ease;
}
.risk-high   { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.1)); border: 2px solid rgba(239,68,68,0.6); }
.risk-medium { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(180,83,9,0.1)); border: 2px solid rgba(245,158,11,0.6); }
.risk-low    { background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1)); border: 2px solid rgba(16,185,129,0.6); }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}
.section-header h2 { color: #e2e8f0; font-size: 1.2rem; font-weight: 600; margin: 0; }

/* ── Info box ── */
.info-box {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #a0aec0;
    line-height: 1.6;
}

/* ── PSI color ── */
.psi-stable { color: #34d399; }
.psi-warn   { color: #fbbf24; }
.psi-drift  { color: #fc8181; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b2e 100%);
    border-right: 1px solid rgba(99,179,237,0.1);
}
section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; color: #a0aec0; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(59,130,246,0.4);
}

/* Slider */
.stSlider > div > div { background: #3b82f6; }

/* DataFrame */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Plotly bg match */
.js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_URL      = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR     = Path(__file__).parent.parent / "mimic_processed"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a0aec0", size=12),
    xaxis=dict(gridcolor="rgba(99,179,237,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(99,179,237,0.08)", zeroline=False),
    margin=dict(l=20, r=20, t=40, b=20),
)

COLOR_HIGH   = "#fc8181"
COLOR_MEDIUM = "#fbbf24"
COLOR_LOW    = "#34d399"
COLOR_BLUE   = "#63b3ed"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "sim_history"    not in st.session_state: st.session_state.sim_history    = []
if "sim_running"    not in st.session_state: st.session_state.sim_running    = False
if "sim_index"      not in st.session_state: st.session_state.sim_index      = 0
if "pred_history"   not in st.session_state: st.session_state.pred_history   = []


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def api_get(path: str, timeout: int = 5):
    r = requests.get(f"{API_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict, timeout: int = 5):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def get_model_info():
    return api_get("/model/info")

@st.cache_data(ttl=10)
def get_health():
    return api_get("/health")


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_test_data():
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"
    if not x_path.exists():
        return None, None
    X = np.load(str(x_path))
    y = np.load(str(y_path)) if y_path.exists() else None
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def risk_gauge(score: float, th_med: float, th_high: float) -> go.Figure:
    color = COLOR_LOW
    if score >= th_high:   color = COLOR_HIGH
    elif score >= th_med:  color = COLOR_MEDIUM

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score * 100, 1),
        delta={"reference": th_high * 100, "suffix": "% vs HIGH threshold"},
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": "Deterioration Risk", "font": {"size": 14, "color": "#a0aec0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568", "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, th_med * 100],              "color": "rgba(16,185,129,0.12)"},
                {"range": [th_med * 100, th_high * 100],  "color": "rgba(245,158,11,0.12)"},
                {"range": [th_high * 100, 100],            "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": COLOR_HIGH, "width": 2},
                "thickness": 0.8,
                "value": th_high * 100,
            },
        },
    ))
    fig.update_layout(height=280, **PLOTLY_THEME)
    return fig


def shap_bar(contribs: list) -> go.Figure:
    if not contribs:
        return None
    names  = [f["feature"] for f in contribs]
    values = [f["shap_value"] for f in contribs]
    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="SHAP Value (impact on risk)",
        height=220, **PLOTLY_THEME
    )
    return fig


def sim_timeline(history: list) -> go.Figure:
    if not history:
        return None
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["score"] * 100,
        mode="lines+markers",
        line=dict(color=COLOR_BLUE, width=2),
        marker=dict(
            size=8,
            color=[COLOR_HIGH if r == "HIGH" else COLOR_MEDIUM if r == "MEDIUM" else COLOR_LOW
                   for r in df["level"]],
            line=dict(color="#0d1117", width=1),
        ),
        name="Risk Score",
        hovertemplate="Patient %{x}<br>Risk: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=20, line_dash="dot", line_color=COLOR_HIGH,   annotation_text="HIGH threshold",   annotation_font_color=COLOR_HIGH)
    fig.add_hline(y=10, line_dash="dot", line_color=COLOR_MEDIUM, annotation_text="MEDIUM threshold", annotation_font_color=COLOR_MEDIUM)
    fig.update_layout(title="Real-time Risk Score Stream", xaxis_title="Patient Index", yaxis_title="Risk %", height=320, **PLOTLY_THEME)
    return fig


def psi_bar(features: list) -> go.Figure:
    if not features:
        return None
    df = pd.DataFrame(features).sort_values("psi", ascending=True).tail(20)
    colors = [COLOR_HIGH if r == "drift" else COLOR_MEDIUM if r == "warning" else COLOR_LOW for r in df["status"]]
    fig = go.Figure(go.Bar(
        x=df["psi"], y=df["feature"], orientation="h",
        marker_color=colors,
        text=df["psi"].round(3), textposition="outside",
    ))
    fig.add_vline(x=0.25, line_dash="dot", line_color=COLOR_HIGH,   annotation_text="Drift (0.25)")
    fig.add_vline(x=0.10, line_dash="dot", line_color=COLOR_MEDIUM, annotation_text="Warning (0.10)")
    fig.update_layout(title="Feature PSI (Top 20)", xaxis_title="PSI Score", height=500, **PLOTLY_THEME)
    return fig


def level_donut(history: list) -> go.Figure:
    if not history:
        return None
    df = pd.DataFrame(history)
    counts = df["level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55,
        marker_colors=[COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW],
        textinfo="percent+label",
    ))
    fig.update_layout(title="Risk Level Distribution", height=280, showlegend=False, **PLOTLY_THEME)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ICU Predictor")
    st.markdown("---")

    # API status
    try:
        health = get_health()
        api_ok = True
        st.success("🟢 API Connected")
        st.caption(f"Model v{health.get('model_version','?')} · {health.get('model_type','?')}")
    except Exception as e:
        api_ok = False
        st.error("🔴 API Offline")
        st.code(str(e), language=None)
        st.warning("Start the API first:\n```\nuvicorn api:app --port 8000\n```")
        st.stop()   # Hard stop — never show content without API

    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["📊 Overview", "🔬 Manual Prediction", "⚡ Live Simulation", "🌊 Drift Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### Settings")
    auto_refresh = st.checkbox("Auto-refresh Overview (30s)", value=False)
    if auto_refresh:
        import streamlit_autorefresh  # optional; graceful skip if missing
        count = streamlit_autorefresh.st_autorefresh(interval=30000, key="autorefresh")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#4a5568; font-size:0.75rem'>"
        "ICU Deterioration ML System<br/>MLOps Stack · 2024</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
        <h1 style='margin:0; color:#e2e8f0; font-size:1.8rem; font-weight:700;'>
            ICU Deterioration Risk Predictor
        </h1>
        <p style='margin:0.2rem 0 0 0; color:#718096; font-size:0.9rem;'>
            Real-time AI monitoring · Champion Model v{health.get('model_version','?')} · {health.get('model_type','?')}
        </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
        <div style='text-align:right; color:#718096; font-size:0.8rem; padding-top:0.6rem;'>
            {datetime.now().strftime('%d %b %Y, %H:%M')}
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if nav == "📊 Overview":
    try:
        info = get_model_info()
    except Exception as e:
        st.error(f"Could not load model info: {e}")
        st.stop()

    # ── Top KPI Cards ──
    th_high = info.get("threshold_high", 0.20)
    th_med  = info.get("threshold_medium", 0.10)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Model Version",    f"v{info.get('champion_version','?')}",      info.get('model_type','?')),
        (c2, "Features",         str(info.get('n_features', '?')),              "input dimensions"),
        (c3, "HIGH Threshold",   f"{th_high*100:.0f}%",                        "alert trigger"),
        (c4, "MED Threshold",    f"{th_med*100:.0f}%",                         "warning trigger"),
        (c5, "SHAP Explainer",   "Active" if info.get('shap_available') else "Off", "explainability"),
    ]
    for col, label, value, sub in cards:
        with col:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Session History ──
    hist = st.session_state.pred_history + st.session_state.sim_history
    if hist:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            fig = sim_timeline(hist)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with c_right:
            fig2 = level_donut(hist)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        total    = len(hist)
        n_high   = sum(1 for h in hist if h["level"] == "HIGH")
        n_medium = sum(1 for h in hist if h["level"] == "MEDIUM")
        avg_score = sum(h["score"] for h in hist) / total

        cc1, cc2, cc3, cc4 = st.columns(4)
        for col, label, val, sub in [
            (cc1, "Total Predictions", str(total), "this session"),
            (cc2, "High Risk Alerts",  str(n_high), f"{n_high/total*100:.0f}% of total"),
            (cc3, "Medium Risk",       str(n_medium), f"{n_medium/total*100:.0f}% of total"),
            (cc4, "Avg Risk Score",    f"{avg_score*100:.1f}%", "across all patients"),
        ]:
            with col:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">{label}</div>
                        <div class="value">{val}</div>
                        <div class="sub">{sub}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-box">
                📌 No predictions yet in this session.<br/>
                Use <b>Manual Prediction</b> to score a single patient, or run the
                <b>Live Simulation</b> to stream real test-set patients through the API.
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MANUAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🔬 Manual Prediction":
    try:
        info = get_model_info()
    except Exception as e:
        st.error(f"Could not load model info: {e}")
        st.stop()

    feature_names = info.get("feature_names", [])
    th_high = info.get("threshold_high", 0.20)
    th_med  = info.get("threshold_medium", 0.10)

    st.markdown('<div class="section-header"><h2>🔬 Manual Patient Risk Assessment</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            Adjust the key clinical vitals below. All remaining model features are set to their
            clinical baseline (0). The champion model scores this in real-time via the API.
        </div><br/>
    """, unsafe_allow_html=True)

    # ── Input Widgets ──
    CLINICAL_FEATURES = {
        "Demographics":  {
            "age":          ("Age (years)",          18, 100, 65,   1),
        },
        "Cardiovascular": {
            "heart_rate_max":  ("Max Heart Rate (bpm)",   40, 250, 90,   1),
            "sysbp_min":       ("Min Systolic BP (mmHg)", 40, 250, 110,  1),
            "diasbp_min":      ("Min Diastolic BP (mmHg)",20, 180, 70,   1),
            "meanbp_min":      ("Min Mean BP (mmHg)",     20, 150, 75,   1),
        },
        "Respiratory": {
            "resp_rate_max":  ("Max Resp Rate (/min)",    5, 60,  18,   1),
            "spo2_min":       ("Min SpO2 (%)",            50, 100, 96,   1),
        },
        "Temperature & Metabolic": {
            "tempc_max":   ("Max Temp (°C)",           32.0, 42.0, 37.0, 0.1),
            "lactate_max": ("Max Lactate (mmol/L)",     0.0, 25.0,  1.2, 0.1),
            "glucose_max": ("Max Glucose (mg/dL)",     50.0, 600.0, 120.0, 1.0),
        },
        "Labs": {
            "wbc_max":       ("Max WBC (K/uL)",        0.0, 150.0, 10.0, 0.1),
            "creatinine_max":("Max Creatinine",        0.1, 20.0,   1.0, 0.1),
            "bun_max":       ("Max BUN (mg/dL)",       0.0, 200.0,  15.0, 1.0),
            "sodium_min":    ("Min Sodium (mEq/L)",   100.0, 170.0, 138.0, 1.0),
            "potassium_max": ("Max Potassium (mEq/L)",  2.0, 10.0,   4.0, 0.1),
            "hematocrit_min":("Min Hematocrit (%)",    10.0, 60.0,  36.0, 0.1),
            "platelet_min":  ("Min Platelets (K/uL)",   10.0, 800.0, 200.0, 1.0),
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
        # Build full feature vector
        full_features = {f: 0.0 for f in feature_names}
        for k, v in user_inputs.items():
            if k in full_features:
                full_features[k] = float(v)

        with st.spinner("Sending to champion model…"):
            try:
                result = api_post("/predict/preprocessed", {"features_array": list(full_features.values()), "stay_id": "manual-dash"})
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        score  = result["risk_score"]
        level  = result["risk_level"]
        contribs = result.get("top_features", [])

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
                        Alert: {"Yes ⚠️" if result.get("alert") else "No ✓"} &nbsp;|&nbsp;
                        Threshold: {result.get("threshold",th_high)*100:.0f}% &nbsp;|&nbsp;
                        Model: {result.get("model_type","?")} v{result.get("model_version","?")}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_g:
            st.plotly_chart(risk_gauge(score, th_med, th_high), use_container_width=True)

        if contribs:
            fig = shap_bar(contribs)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP explainer not available for this model version.")

        # Log to session history
        st.session_state.pred_history.append({
            "ts": datetime.now().strftime("%H:%M:%S"),
            "score": score,
            "level": level,
            "source": "manual",
        })
        st.success("Prediction logged to session history.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "⚡ Live Simulation":
    try:
        info = get_model_info()
    except Exception as e:
        st.error(f"Could not load model info: {e}")
        st.stop()

    feature_names = info.get("feature_names", [])
    th_high = info.get("threshold_high", 0.20)
    th_med  = info.get("threshold_medium", 0.10)

    st.markdown('<div class="section-header"><h2>⚡ Live Test-Data Simulation</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            This streams real held-out test patients from <code>mimic_processed/X_test.npy</code>
            through the champion model API one-by-one, simulating a live ICU monitoring feed.
            Ground-truth labels from <code>y_test.npy</code> are displayed alongside predictions.
        </div><br/>
    """, unsafe_allow_html=True)

    X_test, y_test = load_test_data()
    if X_test is None:
        st.error(f"Test data not found at `{DATA_DIR}`. Run `dvc repro` first.")
        st.stop()

    n_total = len(X_test)

    # Controls
    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1])
    with col_c1:
        batch_n = st.number_input("Patients per batch", 1, 50, 10, key="batch_n")
    with col_c2:
        delay_ms = st.number_input("Delay between batches (ms)", 0, 5000, 500, key="delay_ms")
    with col_c3:
        st.markdown("<br/>", unsafe_allow_html=True)
        start = st.button("▶ Start Streaming", type="primary", use_container_width=True)
    with col_c4:
        st.markdown("<br/>", unsafe_allow_html=True)
        reset = st.button("⏹ Reset", use_container_width=True)

    if reset:
        st.session_state.sim_history = []
        st.session_state.sim_index   = 0
        st.rerun()

    if start:
        idx_start = st.session_state.sim_index
        idx_end   = min(idx_start + int(batch_n), n_total)

        if idx_start >= n_total:
            st.warning(f"All {n_total} test patients have been processed. Click Reset to start over.")
        else:
            progress_bar = st.progress(0, text="Streaming patients…")
            status_box   = st.empty()

            for i, idx in enumerate(range(idx_start, idx_end)):
                row   = X_test[idx]
                label = int(y_test[idx]) if y_test is not None else None

                # Build feature dict aligned to model's expected features
                feat_dict = {}
                for j, fname in enumerate(feature_names):
                    feat_dict[fname] = float(row[j]) if j < len(row) else 0.0

                try:
                    result = api_post("/predict/preprocessed", {"features_array": list(feat_dict.values()), "stay_id": f"sim-{idx}"})
                    score  = result["risk_score"]
                    level  = result["risk_level"]
                    alert  = result.get("alert", False)

                    entry = {
                        "ts":     datetime.now().strftime("%H:%M:%S"),
                        "score":  score,
                        "level":  level,
                        "alert":  alert,
                        "label":  label,
                        "source": "simulation",
                        "idx":    idx,
                    }
                    st.session_state.sim_history.append(entry)

                    badge = f'<span class="badge-{level.lower()}">{level}</span>'
                    truth = f"True label: <b>{'Deteriorated' if label==1 else 'Stable'}</b>" if label is not None else ""
                    status_box.markdown(
                        f"Patient #{idx} → Risk: <b>{score*100:.1f}%</b> {badge} &nbsp;{truth}",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.warning(f"Patient #{idx}: API error — {e}")

                progress_bar.progress((i + 1) / (idx_end - idx_start), text=f"Processed {i+1}/{idx_end - idx_start}")
                time.sleep(delay_ms / 1000)

            st.session_state.sim_index = idx_end
            st.success(f"Batch complete! Processed patients #{idx_start}–{idx_end-1} of {n_total}.")

    # ── Display History ──
    history = st.session_state.sim_history
    if history:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            fig = sim_timeline(history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col_r:
            fig2 = level_donut(history)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        # Stats
        n_high   = sum(1 for h in history if h["level"] == "HIGH")
        n_medium = sum(1 for h in history if h["level"] == "MEDIUM")
        n_low    = sum(1 for h in history if h["level"] == "LOW")
        avg_score = sum(h["score"] for h in history) / len(history)

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val, sub in [
            (c1, "Patients Scored",  str(len(history)), f"of {n_total} total"),
            (c2, "HIGH Alerts",      str(n_high),       f"{n_high/len(history)*100:.0f}%"),
            (c3, "MEDIUM Warnings",  str(n_medium),     f"{n_medium/len(history)*100:.0f}%"),
            (c4, "LOW Risk",         str(n_low),        f"{n_low/len(history)*100:.0f}%"),
            (c5, "Avg Risk Score",   f"{avg_score*100:.1f}%", "mean across stream"),
        ]:
            with col:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">{label}</div>
                        <div class="value">{val}</div>
                        <div class="sub">{sub}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Confusion-style breakdown (if labels available) ──
        labelled = [h for h in history if h["label"] is not None]
        if labelled:
            st.markdown("##### 🎯 Prediction vs Ground Truth")
            df_conf = pd.DataFrame(labelled)
            df_conf["predicted_pos"] = (df_conf["score"] >= info.get("threshold_high", 0.2)).astype(int)
            df_conf["correct"]       = (df_conf["predicted_pos"] == df_conf["label"])

            tp = ((df_conf["predicted_pos"] == 1) & (df_conf["label"] == 1)).sum()
            fp = ((df_conf["predicted_pos"] == 1) & (df_conf["label"] == 0)).sum()
            tn = ((df_conf["predicted_pos"] == 0) & (df_conf["label"] == 0)).sum()
            fn = ((df_conf["predicted_pos"] == 0) & (df_conf["label"] == 1)).sum()

            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)

            cc1, cc2, cc3, cc4 = st.columns(4)
            for col, label, val, sub in [
                (cc1, "Precision",         f"{prec*100:.1f}%", "TP / (TP+FP)"),
                (cc2, "Recall (Sensitivity)", f"{rec*100:.1f}%", "TP / (TP+FN)"),
                (cc3, "F1 Score",          f"{f1*100:.1f}%",  "harmonic mean"),
                (cc4, "Correct Calls",     f"{df_conf['correct'].mean()*100:.1f}%", "accuracy"),
            ]:
                with col:
                    st.markdown(f"""
                        <div class="stat-card">
                            <div class="label">{label}</div>
                            <div class="value">{val}</div>
                            <div class="sub">{sub}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # ── Raw table ──
        st.markdown("<br/>##### 📋 Prediction Log", unsafe_allow_html=True)
        df_show = pd.DataFrame(history[::-1]).rename(columns={
            "ts": "Time", "score": "Risk Score", "level": "Risk Level",
            "alert": "Alert", "label": "True Label", "idx": "Patient #"
        })
        df_show["Risk Score"] = df_show["Risk Score"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(df_show[["Patient #", "Time", "Risk Level", "Risk Score", "Alert", "True Label"]], use_container_width=True)

    else:
        st.markdown("""
            <div class="info-box">
                Click <b>▶ Start Streaming</b> above to begin processing real test patients through the API.
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DRIFT MONITOR
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🌊 Drift Monitor":
    st.markdown('<div class="section-header"><h2>🌊 Feature Drift Monitor</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            Compares the live prediction traffic (rolling log of the last 10,000 predictions)
            against the original training distribution using <b>Population Stability Index (PSI)</b>.
            <br/><br/>
            PSI &lt; 0.10 = <span class="psi-stable">Stable</span> &nbsp;|&nbsp;
            0.10 – 0.25 = <span class="psi-warn">Warning</span> &nbsp;|&nbsp;
            &gt; 0.25 = <span class="psi-drift">Drift Detected</span>
        </div><br/>
    """, unsafe_allow_html=True)

    if st.button("🔄 Fetch Drift Report", type="primary"):
        with st.spinner("Calling /drift/report…"):
            try:
                report = api_get("/drift/report", timeout=30)
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        status  = report.get("status", "unknown")
        max_psi = report.get("max_psi", 0.0)
        drifted = report.get("drifted_features", [])
        warned  = report.get("warning_features", [])
        n_live  = report.get("live_rows", 0)
        n_ref   = report.get("reference_rows", 0)

        # Status banner
        if status == "drift":
            st.error(f"🚨 **DRIFT DETECTED** — {len(drifted)} features have shifted significantly (max PSI: {max_psi:.4f})")
        elif status == "warning":
            st.warning(f"⚠️ **WARNING** — {len(warned)} features approaching drift threshold (max PSI: {max_psi:.4f})")
        elif status == "ok":
            st.success(f"✅ **Stable** — All features within acceptable range (max PSI: {max_psi:.4f})")
        elif status == "insufficient_data":
            st.info("📊 **Insufficient live data** — Need more predictions in the log for a reliable comparison.")
        else:
            st.warning(f"Status: {status}")

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub in [
            (c1, "Status",             status.upper(), "overall"),
            (c2, "Live Rows",          str(n_live),    "in prediction log"),
            (c3, "Reference Rows",     str(n_ref),     "training baseline"),
            (c4, "Max PSI",            f"{max_psi:.4f}", "worst feature"),
        ]:
            with col:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">{label}</div>
                        <div class="value">{val}</div>
                        <div class="sub">{sub}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        features = report.get("features", [])
        if features:
            fig = psi_bar(features)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Drifted features detail
            drifted_feats = [f for f in features if f["status"] == "drift"]
            warned_feats  = [f for f in features if f["status"] == "warning"]
            if drifted_feats:
                st.markdown("##### 🔴 Drifted Features")
                df_d = pd.DataFrame(drifted_feats)[["feature", "psi", "status"]].sort_values("psi", ascending=False)
                st.dataframe(df_d, use_container_width=True)
            if warned_feats:
                st.markdown("##### 🟡 Warning Features")
                df_w = pd.DataFrame(warned_feats)[["feature", "psi", "status"]].sort_values("psi", ascending=False)
                st.dataframe(df_w, use_container_width=True)
        else:
            st.info("No per-feature data returned by the API.")

    else:
        st.markdown("""
            <div class="info-box">
                Click <b>🔄 Fetch Drift Report</b> to run a PSI analysis comparing live traffic
                against the training baseline. Run the <b>Live Simulation</b> first to populate the
                prediction log with enough data for a meaningful comparison.
            </div>
        """, unsafe_allow_html=True)
