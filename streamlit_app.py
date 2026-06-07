"""
ICU Deterioration Predictor — Professional Standalone Dashboard
================================================================
Model-agnostic dashboard. Loads `streamlit_artifacts/inference_pipeline.joblib`
(produced by export_streamlit_bundle.py after every training run) so the
dashboard automatically serves whichever model won training — LightGBM,
XGBoost, Random Forest, or Logistic Regression — without any code changes.

Deploy directly to Streamlit Community Cloud.
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import preprocessor_helper  # registered for joblib unpickling

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Deterioration Predictor",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%); }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

.stat-card {
    background: linear-gradient(135deg, #161b2e 0%, #1a2040 100%);
    border: 1px solid rgba(99,179,237,0.15); border-radius: 16px;
    padding: 1.4rem 1.6rem; text-align: center;
    transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.stat-card:hover { border-color: rgba(99,179,237,0.4); transform: translateY(-2px); }
.stat-card .label { color: #8892a4; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; }
.stat-card .value { color: #e2e8f0; font-size: 2rem; font-weight: 700; line-height: 1; }
.stat-card .sub   { color: #63b3ed; font-size: 0.75rem; margin-top: 0.3rem; }

.badge-high   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.5);  color: #fc8181; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-medium { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.5); color: #fbbf24; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-low    { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.5); color: #34d399; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

.risk-banner { border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0; animation: fadeIn 0.5s ease; }
.risk-high   { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.1)); border: 2px solid rgba(239,68,68,0.6); }
.risk-medium { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(180,83,9,0.1)); border: 2px solid rgba(245,158,11,0.6); }
.risk-low    { background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1)); border: 2px solid rgba(16,185,129,0.6); }

.section-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(99,179,237,0.2); }
.section-header i { font-size: 1.25rem; }
.section-header h2 { color: #e2e8f0; font-size: 1.2rem; font-weight: 600; margin: 0; }

.info-box { background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.25); border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.88rem; color: #a0aec0; line-height: 1.6; }

.cloud-badge { background: linear-gradient(135deg, rgba(99,179,237,0.15), rgba(49,130,206,0.1)); border: 1px solid rgba(99,179,237,0.4); border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; color: #63b3ed; font-weight: 600; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b2e 100%); border-right: 1px solid rgba(99,179,237,0.1); }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; color: #a0aec0; }

.stButton > button { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.6rem 1.5rem; transition: all 0.2s ease; }
.stButton > button:hover { background: linear-gradient(135deg, #60a5fa, #3b82f6); transform: translateY(-1px); box-shadow: 0 4px 15px rgba(59,130,246,0.4); }
.js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("streamlit_artifacts")
COLOR_HIGH    = "#fc8181"
COLOR_MEDIUM  = "#fbbf24"
COLOR_LOW     = "#34d399"
COLOR_BLUE    = "#63b3ed"
TH_HIGH       = 0.20
TH_MED        = 0.10
PLOTLY_THEME  = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a0aec0", size=12),
    xaxis=dict(gridcolor="rgba(99,179,237,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(99,179,237,0.08)", zeroline=False),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [("sim_history", []), ("sim_index", 0), ("pred_history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  — model-agnostic via inference_pipeline.joblib
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_inference_pipeline():
    """Load the complete inference pipeline exported by export_streamlit_bundle.py."""
    pipe_path = ARTIFACTS_DIR / "inference_pipeline.joblib"
    if not pipe_path.exists():
        st.error(
            f"❌ `{pipe_path}` not found.\n\n"
            "Run `python export_streamlit_bundle.py` locally (or trigger the CI/CD "
            "pipeline) to generate Streamlit artifacts."
        )
        st.stop()
    return joblib.load(pipe_path)


@st.cache_resource
def load_preprocessor_meta():
    """Load raw preprocessing metadata for feature-name mapping and defaults."""
    meta_path = ARTIFACTS_DIR / "preprocessing_pipeline.joblib"
    if meta_path.exists():
        return joblib.load(meta_path)
    return None


@st.cache_data
def load_metrics():
    path = ARTIFACTS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_test_predictions():
    path = ARTIFACTS_DIR / "test_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# Eager load at startup (triggers st.stop on missing artifacts)
PIPELINE    = load_inference_pipeline()
PROC_META   = load_preprocessor_meta()

# Derive feature names and model type from the loaded pipeline
_model_step = PIPELINE.named_steps.get("model") or PIPELINE.steps[-1][1]
MODEL_TYPE  = type(_model_step).__name__

# Feature names from preprocessing metadata (these are the 98 post-selector names)
FEATURE_NAMES = list(PROC_META["feature_names"]) if PROC_META else []
N_FEATURES = len(FEATURE_NAMES)

# ── Detect pipeline input requirements ────────────────────────────────────────
# The pipeline may have a ColumnTransformer that needs ALL raw columns (118),
# not just the 98 post-transformation feature names.
try:
    _ct_step = PIPELINE.named_steps.get("preprocessor")
    if hasattr(_ct_step, "transformers_"):
        RAW_COLUMNS = []
        for _, _, cols in _ct_step.transformers_:
            RAW_COLUMNS.extend(list(cols) if hasattr(cols, "__iter__") and not isinstance(cols, str) else [cols])
        _PIPELINE_NEEDS_RAW = True
    else:
        RAW_COLUMNS = list(FEATURE_NAMES)
        _PIPELINE_NEEDS_RAW = False
except Exception:
    RAW_COLUMNS = list(FEATURE_NAMES)
    _PIPELINE_NEEDS_RAW = False

# Training-set means for filling in missing manual inputs
if PROC_META is not None:
    try:
        _ct = PROC_META["preprocessor"]
        _means = []
        for name, trans, cols in _ct.transformers_:
            if hasattr(trans, "named_steps") and hasattr(trans.named_steps.get("scaler", None), "mean_"):
                _means.extend(trans.named_steps["scaler"].mean_)
            elif name == "binary" or name == "indicators":
                _means.extend([0.0] * (len(cols) if hasattr(cols, "__len__") else 1))
        FEATURE_MEANS = np.array(_means) if _means else np.zeros(N_FEATURES)
    except Exception:
        FEATURE_MEANS = np.zeros(N_FEATURES)
else:
    FEATURE_MEANS = np.zeros(N_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_explainer():
    """Create a SHAP explainer directly on the model step (not the full pipeline)."""
    base = _model_step
    try:
        if hasattr(base, "feature_importances_"):
            # Tree models: LightGBM, XGBoost, RandomForest
            return shap.TreeExplainer(base)
        elif hasattr(base, "coef_"):
            # Linear models: LogisticRegression
            bg = np.zeros((1, base.coef_.shape[1]))
            return shap.LinearExplainer(base, bg)
    except Exception as e:
        warnings.warn(f"SHAP initialization failed: {e}")
    return None

EXPLAINER = load_explainer()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build input & preprocess
# ─────────────────────────────────────────────────────────────────────────────
def _build_raw_input(features_dict: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with ALL columns the pipeline expects."""
    if _PIPELINE_NEEDS_RAW:
        # Full pipeline: ColumnTransformer needs 118 raw columns
        row = {c: 0.0 for c in RAW_COLUMNS}
    else:
        # Simple pipeline: only needs the 98 post-transformation features
        row = {f: float(FEATURE_MEANS[i]) for i, f in enumerate(FEATURE_NAMES)}
    row.update({k: float(v) for k, v in features_dict.items() if k in row})
    return pd.DataFrame([row])


def _preprocess_for_shap(df: pd.DataFrame) -> np.ndarray:
    """Transform a DataFrame through ALL pipeline steps EXCEPT the model."""
    X = df
    for step_name, step_transformer in PIPELINE.steps[:-1]:
        X = step_transformer.transform(X)
    return X if isinstance(X, np.ndarray) else np.array(X)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE  — model-agnostic
# ─────────────────────────────────────────────────────────────────────────────
def predict_from_dict(features_dict: dict) -> float:
    """
    Build a single-row DataFrame from the user's inputs (filling missing
    features with defaults), then run it through the full inference pipeline.
    """
    df = _build_raw_input(features_dict)
    try:
        proba = PIPELINE.predict_proba(df)[0, 1]
    except AttributeError:
        proba = float(PIPELINE.predict(df)[0])
    return float(proba)


def risk_level(score: float) -> str:
    if score >= TH_HIGH: return "HIGH"
    if score >= TH_MED:  return "MEDIUM"
    return "LOW"


def top_features(features_dict: dict, n: int = 10) -> list:
    """
    Compute feature contributions using SHAP. Falls back to weight-based
    approximation if SHAP is unavailable.
    """
    df = _build_raw_input(features_dict)

    if EXPLAINER is not None:
        try:
            X_pre = _preprocess_for_shap(df)

            shap_vals = EXPLAINER.shap_values(X_pre)
            # Handle multiclass or different SHAP output formats
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

            contribs = shap_vals[0]
            if hasattr(contribs, "values"):  # for Explanation objects
                contribs = contribs.values

            idx = np.argsort(np.abs(contribs))[::-1][:n]
            return [{"feature": FEATURE_NAMES[i], "contribution": float(contribs[i])} for i in idx]
        except Exception as e:
            warnings.warn(f"SHAP computation failed: {e}")

    # --- Fallback: Approximate feature contributions ---
    row = {f: float(FEATURE_MEANS[i]) for i, f in enumerate(FEATURE_NAMES)}
    row.update({k: float(v) for k, v in features_dict.items() if k in row})
    x = np.array([row[f] for f in FEATURE_NAMES], dtype=np.float64)
    try:
        base_model = _model_step
        if hasattr(base_model, "coef_"):
            W = base_model.coef_.ravel()
            try:
                _prep = PIPELINE.named_steps["preprocessor"]
                scale = np.ones(N_FEATURES)
                mean  = np.zeros(N_FEATURES)
                offset = 0
                for name, trans, cols in _prep.transformers_:
                    nc = len(cols) if hasattr(cols, "__len__") else 1
                    if hasattr(trans, "named_steps"):
                        sc = trans.named_steps.get("scaler")
                        if sc and hasattr(sc, "mean_"):
                            mean[offset:offset+nc]  = sc.mean_
                            scale[offset:offset+nc] = sc.scale_
                    offset += nc
                x_scaled = (x - mean) / np.where(scale == 0, 1, scale)
                contribs  = x_scaled * W
            except Exception:
                contribs = x * W
        elif hasattr(base_model, "feature_importances_"):
            fi = base_model.feature_importances_
            contribs = fi * np.abs(x - FEATURE_MEANS)
        else:
            contribs = np.zeros(N_FEATURES)

        idx = np.argsort(np.abs(contribs))[::-1][:n]
        return [{"feature": FEATURE_NAMES[i], "contribution": float(contribs[i])} for i in idx]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gauge_chart(score: float) -> go.Figure:
    color = COLOR_HIGH if score >= TH_HIGH else COLOR_MEDIUM if score >= TH_MED else COLOR_LOW
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score * 100, 1),
        delta={"reference": TH_HIGH * 100, "suffix": "% vs HIGH threshold"},
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": "Deterioration Risk", "font": {"size": 14, "color": "#a0aec0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, TH_MED * 100],             "color": "rgba(16,185,129,0.12)"},
                {"range": [TH_MED * 100, TH_HIGH * 100], "color": "rgba(245,158,11,0.12)"},
                {"range": [TH_HIGH * 100, 100],           "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": COLOR_HIGH, "width": 2}, "thickness": 0.8, "value": TH_HIGH * 100},
        },
    ))
    fig.update_layout(height=280, **PLOTLY_THEME)
    return fig


def contrib_bar(contribs: list) -> go.Figure:
    names  = [c["feature"] for c in contribs]
    values = [c["contribution"] for c in contribs]
    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Top Driving Features (SHAP Values)",
        xaxis_title="SHAP Value (Impact on Risk)",
        height=320, **PLOTLY_THEME,
    )
    return fig


def feature_importance_chart() -> go.Figure:
    """Model-agnostic importance chart."""
    base = _model_step
    if hasattr(base, "feature_importances_"):
        importance = base.feature_importances_
        title = f"Top 20 Feature Importances ({MODEL_TYPE})"
        colors = [COLOR_BLUE] * N_FEATURES
    elif hasattr(base, "coef_"):
        importance = np.abs(base.coef_.ravel())
        title = "Top 20 Feature Importances (|weight|) — Red = raises risk, Green = lowers risk"
        colors = [COLOR_HIGH if base.coef_.ravel()[i] > 0 else COLOR_LOW for i in range(N_FEATURES)]
    else:
        return go.Figure()

    idx   = np.argsort(importance)[-20:]
    names = [FEATURE_NAMES[i] for i in idx]
    vals  = [importance[i] for i in idx]
    clrs  = [colors[i] for i in idx]

    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=clrs))
    fig.update_layout(title=title, xaxis_title="Importance", height=520, **PLOTLY_THEME)
    return fig


def timeline_chart(history: list) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["score"] * 100,
        mode="lines+markers", line=dict(color=COLOR_BLUE, width=2),
        marker=dict(
            size=8,
            color=[COLOR_HIGH if r == "HIGH" else COLOR_MEDIUM if r == "MEDIUM" else COLOR_LOW for r in df["level"]],
            line=dict(color="#0d1117", width=1),
        ),
        hovertemplate="Patient %{x}<br>Risk: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=TH_HIGH * 100, line_dash="dot", line_color=COLOR_HIGH,   annotation_text="HIGH",   annotation_font_color=COLOR_HIGH)
    fig.add_hline(y=TH_MED  * 100, line_dash="dot", line_color=COLOR_MEDIUM, annotation_text="MEDIUM", annotation_font_color=COLOR_MEDIUM)
    fig.update_layout(title="Real-time Risk Score Stream", xaxis_title="Patient", yaxis_title="Risk %", height=320, **PLOTLY_THEME)
    return fig


def donut_chart(history: list) -> go.Figure:
    df = pd.DataFrame(history)
    counts = df["level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker_colors=[COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW],
        textinfo="percent+label",
    ))
    fig.update_layout(title="Risk Level Distribution", height=280, showlegend=False, **PLOTLY_THEME)
    return fig


def calibration_chart(cal_data: list) -> go.Figure:
    df  = pd.DataFrame(cal_data)
    mid = (df["bin_start"] + df["bin_end"]) / 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid, y=df["mean_pred"], mode="lines+markers", name="Mean Predicted", line=dict(color=COLOR_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=mid, y=df["mean_true"], mode="lines+markers", name="Observed Rate",  line=dict(color=COLOR_HIGH, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(color="#4a5568", dash="dash")))
    fig.update_layout(title="Calibration Curve (Test Set)", xaxis_title="Mean Predicted Probability", yaxis_title="Observed Event Rate", height=320, **PLOTLY_THEME)
    return fig


def stat_card(col, label, value, sub):
    with col:
        st.markdown(f"""
            <div class="stat-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## <i class='fa-solid fa-hospital-user' style='color:#63b3ed; margin-right:8px;'></i> ICU Predictor", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<span class="cloud-badge"><i class="fa-solid fa-cloud" style="margin-right:6px;"></i> Cloud Edition · Serverless</span>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style='margin-top:0.8rem; padding:0.8rem; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3); border-radius:10px;'>
            <span style='color:#34d399; font-weight:600;'><i class='fa-solid fa-circle-check' style='margin-right:6px;'></i> Model Loaded</span><br/>
            <span style='color:#718096; font-size:0.78rem; margin-left: 20px;'>{MODEL_TYPE}<br/><span style='margin-left: 20px;'>{N_FEATURES} features · Serverless</span></span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["Overview", "Manual Prediction", "Live Simulation", "Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#4a5568; font-size:0.75rem;'>"
        "<i class='fa-solid fa-server' style='margin-right:4px;'></i> ICU Deterioration ML System<br/>MLOps Stack · Serverless Cloud</div>",
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
            Real-time AI monitoring · <b>{MODEL_TYPE}</b> · {N_FEATURES} Clinical Features
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
if nav == "Overview":
    metrics = load_metrics()
    test_m  = metrics.get("test", {})

    st.markdown('<div class="section-header"><i class="fa-solid fa-chart-pie" style="color:#63b3ed;"></i><h2>System Overview</h2></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    stat_card(c1, "Model Type",      MODEL_TYPE,                           "champion classifier")
    stat_card(c2, "Features",        str(N_FEATURES),                      "input dimensions")
    stat_card(c3, "Test AUROC",      f"{test_m.get('auroc', test_m.get('auc_roc', 0)):.3f}", "discrimination")
    stat_card(c4, "Test Recall",     f"{test_m.get('recall', 0)*100:.1f}%",  "sensitivity")
    stat_card(c5, "HIGH Threshold",  f"{TH_HIGH*100:.0f}%",                "alert trigger")

    st.markdown("<br/>", unsafe_allow_html=True)

    hist = st.session_state.pred_history + st.session_state.sim_history
    if hist:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(timeline_chart(hist), use_container_width=True)
        with col_r:
            st.plotly_chart(donut_chart(hist), use_container_width=True)

        total    = len(hist)
        n_high   = sum(1 for h in hist if h["level"] == "HIGH")
        n_medium = sum(1 for h in hist if h["level"] == "MEDIUM")
        avg      = sum(h["score"] for h in hist) / total

        c1, c2, c3, c4 = st.columns(4)
        stat_card(c1, "Total Predictions",  str(total),                       "this session")
        stat_card(c2, "High Risk Alerts",   str(n_high),                      f"{n_high/total*100:.0f}% of total")
        stat_card(c3, "Medium Risk",        str(n_medium),                    f"{n_medium/total*100:.0f}% of total")
        stat_card(c4, "Avg Risk Score",     f"{avg*100:.1f}%",                "across all patients")
    else:
        st.markdown("""
            <div class="info-box">
                <i class="fa-solid fa-circle-info" style="color:#63b3ed; margin-right:6px;"></i> No predictions yet in this session.<br/>
                Use <b>Manual Prediction</b> to score a single patient, or run the
                <b>Live Simulation</b> to stream real test-set patients through the model.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.plotly_chart(feature_importance_chart(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MANUAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Manual Prediction":
    st.markdown('<div class="section-header"><i class="fa-solid fa-heart-pulse" style="color:#63b3ed;"></i><h2>Manual Patient Risk Assessment</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="info-box">
            Adjust the key clinical vitals below. All remaining model features are filled
            with their training-set mean values. The <b>{MODEL_TYPE}</b> model scores
            this patient instantly — no internet connection to a backend required.
        </div><br/>
    """, unsafe_allow_html=True)

    # ── Basic clinical features (always shown) ──────────────────────────────────
    CLINICAL_FEATURES = {
        "Demographics": {
            "age":            ("Age (years)",             18, 100, 65,    1),
            "pre_icu_hours":  ("Hours before ICU admit",   0, 500, 57,    1),
            "los_days":       ("ICU LOS (days)",           0, 60,   3,    1),
        },
        "Cardiovascular": {
            "hr_max":         ("Max Heart Rate (bpm)",     40, 250,  90,   1),
            "hr_min":         ("Min Heart Rate (bpm)",     20, 200,  60,   1),
            "sbp_min":        ("Min Systolic BP (mmHg)",   40, 250, 110,   1),
            "map_min":        ("Min Mean BP (mmHg)",       20, 150,  75,   1),
        },
        "Respiratory": {
            "rr_max":         ("Max Resp Rate (/min)",      5, 60,   18,   1),
            "spo2_min":       ("Min SpO2 (%)",             50, 100,  96,   1),
            "fio2_max":       ("Max FiO2 (%)",              0, 100,   1,   1),
        },
        "Temperature & GCS": {
            "temp_c_max":     ("Max Temp (°C)",           32.0, 42.0, 37.0, 0.1),
            "gcs_total_mean": ("Mean GCS Total",           3, 15, 12, 1),
            "gcs_eye_min":    ("Min GCS Eye",               1, 4,     3,   1),
            "gcs_motor_min":  ("Min GCS Motor",             1, 6,     5,   1),
            "gcs_verb_min":   ("Min GCS Verbal",            1, 5,     4,   1),
        },
        "Labs": {
            "lab_lactate":    ("Lactate (mmol/L)",         0.0, 25.0,  1.2, 0.1),
            "lab_creatinine": ("Creatinine",               0.1, 20.0,  1.0, 0.1),
            "lab_bun":        ("BUN (mg/dL)",              0.0,200.0, 15.0, 1.0),
            "lab_wbc":        ("WBC (K/uL)",               0.0,150.0, 10.0, 0.1),
            "lab_hemoglobin": ("Hemoglobin",               0.0, 25.0, 12.0, 0.1),
            "lab_sodium":     ("Sodium (mEq/L)",          100.0,170.0,138.0, 1.0),
            "lab_potassium":  ("Potassium (mEq/L)",        2.0, 10.0,  4.0, 0.1),
            "lab_glucose":    ("Glucose (mg/dL)",         50.0,600.0,120.0, 1.0),
            "lab_ph":         ("Arterial pH",              6.8,  7.8,  7.4, 0.01),
        },
        "Fluids & Vasopressors": {
            "total_urine_ml":       ("Total Urine (mL)",    0.0, 10000.0, 1816.0, 10.0),
            "fluid_balance_ml":     ("Fluid Balance (mL)", -5000.0, 10000.0, 985.0, 10.0),
            "vaso_flag":            ("Vasopressors Used",   0, 1, 0, 1),
            "vaso_max_rate":        ("Max Vasopressor Rate", 0.0, 5.0, 0.0, 0.01),
        },
    }

    # ── Advanced features (shown on toggle) ───────────────────────────────────
    ADVANCED_FEATURES = {
        "Vital Signs — First / Last / Mean / Std / Delta": {
            "hr_first":   ("HR First (bpm)",      20, 250, 80, 1),
            "hr_last":    ("HR Last (bpm)",        20, 250, 80, 1),
            "hr_mean":    ("HR Mean (bpm)",        20, 250, 78, 1),
            "hr_std":     ("HR Std Dev",           0.0, 60.0, 5.0, 0.1),
            "hr_delta":   ("HR Delta (last−first)",-100, 100, 0, 1),
            "sbp_max":    ("Max SBP (mmHg)",       40, 300, 140, 1),
            "sbp_min":    ("Min SBP (mmHg)",       40, 250, 110, 1),
            "sbp_first":  ("SBP First (mmHg)",     40, 300, 120, 1),
            "sbp_last":   ("SBP Last (mmHg)",      40, 300, 120, 1),
            "sbp_mean":   ("SBP Mean (mmHg)",      40, 300, 120, 1),
            "sbp_std":    ("SBP Std Dev",          0.0, 80.0, 8.0, 0.1),
            "sbp_delta":  ("SBP Delta",           -150, 150, 0, 1),
            "map_max":    ("Max MAP (mmHg)",       20, 200, 90, 1),
            "map_first":  ("MAP First (mmHg)",     20, 200, 78, 1),
            "map_last":   ("MAP Last (mmHg)",      20, 200, 78, 1),
            "map_mean":   ("MAP Mean (mmHg)",      20, 200, 78, 1),
            "map_std":    ("MAP Std Dev",          0.0, 50.0, 5.0, 0.1),
            "map_delta":  ("MAP Delta",           -100, 100, 0, 1),
            "rr_min":     ("Min Resp Rate",         2, 60, 12, 1),
            "rr_first":   ("RR First (/min)",       2, 60, 16, 1),
            "rr_last":    ("RR Last (/min)",        2, 60, 16, 1),
            "rr_mean":    ("RR Mean (/min)",        2, 60, 16, 1),
            "rr_std":     ("RR Std Dev",           0.0, 30.0, 2.0, 0.1),
            "rr_delta":   ("RR Delta",            -30, 30, 0, 1),
            "spo2_max":   ("Max SpO2 (%)",         70, 100, 99, 1),
            "spo2_first": ("SpO2 First (%)",       50, 100, 97, 1),
            "spo2_last":  ("SpO2 Last (%)",        50, 100, 97, 1),
            "spo2_mean":  ("SpO2 Mean (%)",        50, 100, 97, 1),
            "spo2_std":   ("SpO2 Std Dev",         0.0, 20.0, 1.5, 0.1),
            "spo2_delta": ("SpO2 Delta",          -30, 30, 0, 1),
        },
        "FiO2 & PEEP (Ventilator)": {
            "fio2_min":   ("Min FiO2",             0.0, 100.0, 0.21, 0.01),
            "fio2_first": ("FiO2 First",           0.0, 100.0, 0.21, 0.01),
            "fio2_last":  ("FiO2 Last",            0.0, 100.0, 0.21, 0.01),
            "fio2_mean":  ("FiO2 Mean",            0.0, 100.0, 0.21, 0.01),
            "fio2_std":   ("FiO2 Std Dev",         0.0, 50.0, 0.0, 0.01),
            "fio2_delta": ("FiO2 Delta",          -1.0, 1.0, 0.0, 0.01),
            "peep_max":   ("Max PEEP (cmH2O)",     0, 40, 5, 1),
            "peep_min":   ("Min PEEP (cmH2O)",     0, 40, 5, 1),
            "peep_first": ("PEEP First",           0, 40, 5, 1),
            "peep_last":  ("PEEP Last",            0, 40, 5, 1),
            "peep_mean":  ("PEEP Mean",            0.0, 40.0, 5.0, 0.1),
            "peep_std":   ("PEEP Std Dev",         0.0, 20.0, 0.0, 0.1),
            "peep_delta": ("PEEP Delta",          -20, 20, 0, 1),
        },
        "Extended Labs": {
            "lab_bicarbonate": ("Bicarbonate (mEq/L)", 5.0, 45.0, 24.0, 0.1),
            "lab_bilirubin":   ("Bilirubin (mg/dL)",   0.0, 40.0, 0.8, 0.1),
            "lab_inr":         ("INR",                 0.5, 15.0, 1.1, 0.1),
            "lab_paco2":       ("PaCO2 (mmHg)",       10.0, 120.0, 40.0, 1.0),
            "lab_pao2":        ("PaO2 (mmHg)",        20.0, 600.0, 90.0, 1.0),
            "lab_platelets":   ("Platelets (K/uL)",    0.0, 1000.0, 220.0, 1.0),
        },
        "ICU Unit Type": {
            "unit_micu":      ("MICU (Medical ICU)",         0, 1, 0, 1),
            "unit_sicu":      ("SICU (Surgical ICU)",        0, 1, 0, 1),
            "unit_ccu":       ("CCU (Cardiac Care)",         0, 1, 0, 1),
            "unit_cvicu":     ("CVICU (Cardiovascular ICU)", 0, 1, 0, 1),
            "unit_nicu":      ("NICU (Neuro ICU)",           0, 1, 0, 1),
            "unit_tsicu":     ("TSICU (Trauma/Surg ICU)",    0, 1, 0, 1),
            "unit_micu_sicu": ("MICU/SICU (Combined)",       0, 1, 0, 1),
        },
        "Fluids & Urine (Extended)": {
            "total_input_ml":   ("Total Input (mL)",        0.0, 20000.0, 2500.0, 10.0),
            "urine_n_obs":      ("Urine Observations",      0, 100, 10, 1),
            "urine_rate_ml_hr": ("Urine Rate (mL/hr)",      0.0, 500.0, 60.0, 1.0),
        },
        "Observation Counts": {
            "hr_n_obs":   ("HR Observations",    0, 500, 20, 1),
            "sbp_n_obs":  ("SBP Observations",   0, 500, 15, 1),
            "map_n_obs":  ("MAP Observations",   0, 500, 15, 1),
            "rr_n_obs":   ("RR Observations",    0, 500, 15, 1),
            "spo2_n_obs": ("SpO2 Observations",  0, 500, 20, 1),
            "fio2_n_obs": ("FiO2 Observations",  0, 500, 5,  1),
            "peep_n_obs": ("PEEP Observations",  0, 500, 5,  1),
        },
        "Admission Info": {
            "admit_hour":    ("Admission Hour (0–23)",   0, 23, 12, 1),
            "admit_weekday": ("Admission Weekday (0=Mon)", 0, 6, 3, 1),
            "admit_month":   ("Admission Month (1–12)",  1, 12, 6, 1),
            "gender_male":   ("Gender Male",             0, 1, 0, 1),
            "emergency_flag":("Emergency Admission",     0, 1, 0, 1),
            "fluid_negative_flag": ("Fluid Negative",    0, 1, 0, 1),
            "oliguria_flag":       ("Oliguria Flag",     0, 1, 0, 1),
        },
        "Missingness Flags": {
            "sbp_was_missing":             ("SBP Missing",              0, 1, 0, 1),
            "rr_was_missing":              ("RR Missing",               0, 1, 0, 1),
            "fio2_was_missing":            ("FiO2 Missing",             0, 1, 0, 1),
            "hr_was_missing":              ("HR Missing",               0, 1, 0, 1),
            "map_was_missing":             ("MAP Missing",              0, 1, 0, 1),
            "peep_was_missing":            ("PEEP Missing",             0, 1, 0, 1),
            "spo2_was_missing":            ("SpO2 Missing",             0, 1, 0, 1),
            "lab_lactate_was_missing":     ("Lactate Missing",          0, 1, 0, 1),
            "lab_creatinine_was_missing":  ("Creatinine Missing",       0, 1, 0, 1),
            "lab_bun_was_missing":         ("BUN Missing",              0, 1, 0, 1),
            "lab_wbc_was_missing":         ("WBC Missing",              0, 1, 0, 1),
            "lab_hemoglobin_was_missing":  ("Hemoglobin Missing",       0, 1, 0, 1),
            "lab_platelets_was_missing":   ("Platelets Missing",        0, 1, 0, 1),
            "lab_sodium_was_missing":      ("Sodium Missing",           0, 1, 0, 1),
            "lab_potassium_was_missing":   ("Potassium Missing",        0, 1, 0, 1),
            "lab_bicarbonate_was_missing": ("Bicarbonate Missing",      0, 1, 0, 1),
            "lab_bilirubin_was_missing":   ("Bilirubin Missing",        0, 1, 0, 1),
            "lab_inr_was_missing":         ("INR Missing",              0, 1, 0, 1),
            "lab_ph_was_missing":          ("pH Missing",               0, 1, 0, 1),
            "lab_pao2_was_missing":        ("PaO2 Missing",             0, 1, 0, 1),
            "lab_paco2_was_missing":       ("PaCO2 Missing",            0, 1, 0, 1),
            "lab_glucose_was_missing":     ("Glucose Missing",          0, 1, 0, 1),
        },
    }

    # ── Render basic features ─────────────────────────────────────────────────
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

    # ── Toggle for advanced features ──────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    show_advanced = st.toggle(
        "Show Advanced Features (all 118 model inputs)",
        value=False,
        help="Enable to see and adjust ALL features the model uses, including vital sign statistics, observation counts, ICU unit type, admission info, and missingness flags."
    )

    if show_advanced:
        st.markdown("""
            <div class="info-box" style="margin-bottom:1rem;">
                <i class="fa-solid fa-flask" style="color:#63b3ed; margin-right:6px;"></i>
                <b>Advanced Mode</b> — These are the remaining model features that are normally
                auto-filled with default values. Adjust them for fine-grained control over the prediction.
            </div>
        """, unsafe_allow_html=True)
        for section, feats in ADVANCED_FEATURES.items():
            with st.expander(f"**{section}**", expanded=False):
                cols = st.columns(min(len(feats), 4))
                for idx, (key, (label, mn, mx, default, step)) in enumerate(feats.items()):
                    with cols[idx % len(cols)]:
                        if isinstance(step, float):
                            user_inputs[key] = st.number_input(label, float(mn), float(mx), float(default), step=step, key=f"adv_{key}")
                        else:
                            user_inputs[key] = st.number_input(label, int(mn), int(mx), int(default), step=step, key=f"adv_{key}")

    st.markdown("<br/>", unsafe_allow_html=True)
    run_btn = st.button("Score Patient", type="primary")

    if run_btn:
        score    = predict_from_dict(user_inputs)
        level    = risk_level(score)
        contribs = top_features(user_inputs)

        risk_css  = f"risk-{level.lower()}"
        badge_css = f"badge-{level.lower()}"
        
        if level == "HIGH":
            icon_html = '<i class="fa-solid fa-triangle-exclamation" style="color:#fc8181; font-size:3.5rem; margin-bottom: 0.5rem; display: inline-block;"></i>'
        elif level == "MEDIUM":
            icon_html = '<i class="fa-solid fa-circle-exclamation" style="color:#fbbf24; font-size:3.5rem; margin-bottom: 0.5rem; display: inline-block;"></i>'
        else:
            icon_html = '<i class="fa-solid fa-circle-check" style="color:#34d399; font-size:3.5rem; margin-bottom: 0.5rem; display: inline-block;"></i>'
            
        alert     = score >= TH_HIGH
        alert_html = 'Yes <i class="fa-solid fa-triangle-exclamation" style="color:#fc8181; margin-left:4px;"></i>' if alert else 'No <i class="fa-solid fa-circle-check" style="color:#34d399; margin-left:4px;"></i>'

        col_r, col_g = st.columns([1, 1])
        with col_r:
            st.markdown(f"""
                <div class="risk-banner {risk_css}">
                    <div>{icon_html}</div>
                    <span class="{badge_css}">{level} RISK</span>
                    <div style="font-size:3.5rem; font-weight:800; color:#e2e8f0; margin:0.5rem 0;">
                        {score*100:.1f}%
                    </div>
                    <div style="color:#a0aec0; font-size:0.85rem;">
                        Alert: {alert_html} &nbsp;|&nbsp;
                        Threshold: {TH_HIGH*100:.0f}% &nbsp;|&nbsp;
                        Model: {MODEL_TYPE}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_g:
            st.plotly_chart(gauge_chart(score), use_container_width=True)

        if contribs:
            st.plotly_chart(contrib_bar(contribs), use_container_width=True)

        st.session_state.pred_history.append({
            "ts": datetime.now().strftime("%H:%M:%S"),
            "score": score, "level": level, "source": "manual",
        })
        st.success("Prediction logged to session history.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Live Simulation":
    st.markdown('<div class="section-header"><i class="fa-solid fa-bolt" style="color:#63b3ed;"></i><h2>Live Test-Data Simulation</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            This streams <b>real held-out test patients</b> from the pre-computed
            <code>streamlit_artifacts/test_predictions.parquet</code> file through
            the model one-by-one, simulating a live ICU monitoring feed.
            Ground-truth labels are shown alongside model predictions.
        </div><br/>
    """, unsafe_allow_html=True)

    test_df = load_test_predictions()
    if test_df is None:
        st.error("Test predictions file not found. Trigger the CI/CD pipeline to generate `streamlit_artifacts/test_predictions.parquet`.")
        st.stop()

    n_total = len(test_df)
    score_col = next((c for c in ["pred_proba", "score", "risk_score", "y_pred_proba"] if c in test_df.columns), None)
    label_col = next((c for c in ["label", "y_true", "target", "deterioration"] if c in test_df.columns), None)

    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1])
    with col_c1:
        batch_n  = st.number_input("Patients per batch", 1, 100, 20, key="batch_n")
    with col_c2:
        delay_ms = st.number_input("Delay per patient (ms)", 0, 2000, 100, key="delay_ms")
    with col_c3:
        st.markdown("<br/>", unsafe_allow_html=True)
        start = st.button("Start Streaming", type="primary", use_container_width=True)
    with col_c4:
        st.markdown("<br/>", unsafe_allow_html=True)
        reset = st.button("Reset", use_container_width=True)

    if reset:
        st.session_state.sim_history = []
        st.session_state.sim_index   = 0
        st.rerun()

    if start:
        idx_start = st.session_state.sim_index
        idx_end   = min(idx_start + int(batch_n), n_total)

        if idx_start >= n_total:
            st.warning(f"All {n_total} test patients processed. Click Reset to start over.")
        else:
            prog   = st.progress(0, text="Streaming patients…")
            status = st.empty()

            for i, idx in enumerate(range(idx_start, idx_end)):
                row   = test_df.iloc[idx]
                score = float(row[score_col]) if score_col else 0.5
                label = int(row[label_col]) if label_col else None
                level = risk_level(score)
                alert = score >= TH_HIGH

                st.session_state.sim_history.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "score": score, "level": level, "alert": alert,
                    "label": label, "source": "simulation", "idx": idx,
                })

                badge = f'<span class="badge-{level.lower()}">{level}</span>'
                truth = f"True: <b>{'Deteriorated' if label==1 else 'Stable'}</b>" if label is not None else ""
                status.markdown(
                    f"Patient #{idx} → Risk: <b>{score*100:.1f}%</b> {badge} &nbsp;{truth}",
                    unsafe_allow_html=True
                )
                prog.progress((i + 1) / (idx_end - idx_start))
                time.sleep(delay_ms / 1000)

            st.session_state.sim_index = idx_end
            st.success(f"Batch complete! Processed patients #{idx_start}–{idx_end-1} of {n_total}.")

    history = st.session_state.sim_history
    if history:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.plotly_chart(timeline_chart(history), use_container_width=True)
        with col_r:
            st.plotly_chart(donut_chart(history), use_container_width=True)

        n_high   = sum(1 for h in history if h["level"] == "HIGH")
        n_medium = sum(1 for h in history if h["level"] == "MEDIUM")
        n_low    = sum(1 for h in history if h["level"] == "LOW")
        avg      = sum(h["score"] for h in history) / len(history)

        c1, c2, c3, c4, c5 = st.columns(5)
        stat_card(c1, "Patients Scored",  str(len(history)),       f"of {n_total} total")
        stat_card(c2, "HIGH Alerts",      str(n_high),             f"{n_high/len(history)*100:.0f}%")
        stat_card(c3, "MEDIUM Warnings",  str(n_medium),           f"{n_medium/len(history)*100:.0f}%")
        stat_card(c4, "LOW Risk",         str(n_low),              f"{n_low/len(history)*100:.0f}%")
        stat_card(c5, "Avg Risk Score",   f"{avg*100:.1f}%",       "mean across stream")

        labelled = [h for h in history if h.get("label") is not None]
        if labelled:
            st.markdown("<br/>##### <i class='fa-solid fa-bullseye' style='color:#63b3ed; margin-right:6px;'></i> Prediction vs Ground Truth", unsafe_allow_html=True)
            df_c = pd.DataFrame(labelled)
            df_c["pred_pos"] = (df_c["score"] >= TH_HIGH).astype(int)
            tp = int(((df_c["pred_pos"] == 1) & (df_c["label"] == 1)).sum())
            fp = int(((df_c["pred_pos"] == 1) & (df_c["label"] == 0)).sum())
            tn = int(((df_c["pred_pos"] == 0) & (df_c["label"] == 0)).sum())
            fn = int(((df_c["pred_pos"] == 0) & (df_c["label"] == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

            cc1, cc2, cc3, cc4 = st.columns(4)
            stat_card(cc1, "Precision",              f"{prec*100:.1f}%", "TP/(TP+FP)")
            stat_card(cc2, "Recall (Sensitivity)",   f"{rec*100:.1f}%",  "TP/(TP+FN)")
            stat_card(cc3, "F1 Score",               f"{f1*100:.1f}%",   "harmonic mean")
            stat_card(cc4, "Accuracy",               f"{acc*100:.1f}%",  "correct calls")

        st.markdown("<br/>##### <i class='fa-solid fa-list' style='color:#63b3ed; margin-right:6px;'></i> Prediction Log", unsafe_allow_html=True)
        df_show = pd.DataFrame(history[::-1]).rename(columns={
            "ts": "Time", "score": "Risk Score", "level": "Risk Level",
            "alert": "Alert", "label": "True Label", "idx": "Patient #"
        })
        df_show["Risk Score"] = df_show["Risk Score"].apply(lambda x: f"{x*100:.1f}%")
        display_cols = [c for c in ["Patient #", "Time", "Risk Level", "Risk Score", "Alert", "True Label"] if c in df_show.columns]
        st.dataframe(df_show[display_cols], use_container_width=True)
    else:
        st.markdown("""
            <div class="info-box">
                Click <b>Start Streaming</b> to begin processing real test patients through the model.
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Model Performance":
    metrics = load_metrics()
    st.markdown('<div class="section-header"><i class="fa-solid fa-chart-line" style="color:#63b3ed;"></i><h2>Model Performance</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="info-box">
            Held-out evaluation metrics for the <b>{MODEL_TYPE}</b> champion model.
            AUROC measures overall discrimination; Recall (sensitivity) is critical in ICU settings
            to minimise missed deterioration events.
        </div><br/>
    """, unsafe_allow_html=True)

    split = st.radio("Dataset split:", ["test", "validation", "train"], horizontal=True)
    m = metrics.get(split, {})

    auroc_key = "auroc" if "auroc" in m else "auc_roc"
    auprc_key = "auprc" if "auprc" in m else "pr_auc"

    c1, c2, c3, c4 = st.columns(4)
    stat_card(c1, "AUROC",    f"{m.get(auroc_key, 0):.4f}",              "discrimination")
    stat_card(c2, "Recall",   f"{m.get('recall', 0)*100:.1f}%",          "sensitivity")
    stat_card(c3, "Precision",f"{m.get('precision', 0)*100:.1f}%",       "PPV")
    stat_card(c4, "F1 Score", f"{m.get('f1', 0)*100:.1f}%",              "harmonic mean")

    st.markdown("<br/>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    stat_card(c5, "Accuracy",    f"{m.get('accuracy', 0)*100:.1f}%",    "overall correct")
    stat_card(c6, "Specificity", f"{m.get('specificity', 0)*100:.1f}%", "true neg rate")
    stat_card(c7, "AUPRC",       f"{m.get(auprc_key, 0):.4f}",          "precision-recall AUC")
    stat_card(c8, "Brier Score", f"{m.get('brier', 0):.4f}",            "calibration error")

    st.markdown("<br/>", unsafe_allow_html=True)

    cal = m.get("calibration_curve")
    col_a, col_b = st.columns(2)
    with col_a:
        if cal:
            st.plotly_chart(calibration_chart(cal), use_container_width=True)
        else:
            st.info("Calibration curve not available for this split.")
    with col_b:
        st.plotly_chart(feature_importance_chart(), use_container_width=True)
