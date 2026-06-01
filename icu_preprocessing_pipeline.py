"""
ICU Deterioration — Preprocessing Pipeline
==========================================
Run after EDA. Expects the raw feature matrix parquet + feature list CSV.
Output: X_train, X_test, y_train, y_test (numpy arrays) + fitted pipeline object.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────

base_dir = os.path.join(os.getcwd(), "data preparation/mimic_processed")

parquet_path = sorted(glob.glob(os.path.join(base_dir, "**", "*feature_matrix_raw*.parquet"), recursive=True))[0]
df = pd.read_parquet(parquet_path)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────────
# 1. TRAIN / TEST SPLIT  (before any fitting)
# ─────────────────────────────────────────────

TARGET = "label"
DROP_COLS = ["stay_id", "intime", TARGET]

y = df[TARGET].astype(int)
X = df.drop(columns=DROP_COLS, errors="ignore")

# Extract temporal features from intime before dropping
if "intime" in df.columns:
    X["admit_hour"]    = pd.to_datetime(df["intime"]).dt.hour
    X["admit_weekday"] = pd.to_datetime(df["intime"]).dt.weekday
    X["admit_month"]   = pd.to_datetime(df["intime"]).dt.month

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
print(f"Train: {X_train_raw.shape[0]:,} | Test: {X_test_raw.shape[0]:,}")
print(f"Positive rate — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")


# ─────────────────────────────────────────────
# 2. COLUMN CATEGORISATION
# ─────────────────────────────────────────────

# 2a. Drop temp_c_* (>96% missing — unrecoverable)
TEMP_COLS = [c for c in X_train_raw.columns if c.startswith("temp_c_")]
X_train_raw = X_train_raw.drop(columns=TEMP_COLS)
X_test_raw  = X_test_raw.drop(columns=TEMP_COLS)
print(f"Dropped {len(TEMP_COLS)} temp_c_* columns.")

# 2b. Indicator columns (_was_missing) — keep as binary, no imputation needed
INDICATOR_COLS = [c for c in X_train_raw.columns if c.endswith("_was_missing")]

# 2c. Binary / flag columns — leave as-is
BINARY_COLS = [
    c for c in X_train_raw.columns
    if c not in INDICATOR_COLS
    and X_train_raw[c].dropna().isin([0, 1]).all()
    and X_train_raw[c].nunique(dropna=True) == 2
]

# 2d. GCS sub-columns — will be reduced
GCS_COLS = [c for c in X_train_raw.columns if c.startswith("gcs_")]

# 2e. Skewed features — need log1p before scaling
SKEWED_COLS = ["pre_icu_hours", "los_days", "vaso_max_rate"]
SKEWED_COLS = [c for c in SKEWED_COLS if c in X_train_raw.columns]

# 2f. Remaining continuous features
SPECIAL_COLS = set(INDICATOR_COLS + BINARY_COLS + GCS_COLS + SKEWED_COLS)
CONTINUOUS_COLS = [
    c for c in X_train_raw.columns
    if c not in SPECIAL_COLS
    and pd.api.types.is_numeric_dtype(X_train_raw[c])
]

print(f"\nColumn groups:"
      f"\n  Indicators (_was_missing) : {len(INDICATOR_COLS)}"
      f"\n  Binary / flags            : {len(BINARY_COLS)}"
      f"\n  GCS sub-features          : {len(GCS_COLS)}"
      f"\n  Skewed (log1p)            : {len(SKEWED_COLS)}"
      f"\n  Continuous                : {len(CONTINUOUS_COLS)}")


# ─────────────────────────────────────────────
# 3. MANUAL GCS AGGREGATION
#    Create 3 summary features; drop the raw sub-columns.
# ─────────────────────────────────────────────

def aggregate_gcs(frame):
    frame = frame.copy()
    motor_mean = frame.get("gcs_motor_mean", pd.Series(np.nan, index=frame.index))
    verb_mean  = frame.get("gcs_verb_mean",  pd.Series(np.nan, index=frame.index))
    eye_mean   = frame.get("gcs_eye_mean",   pd.Series(np.nan, index=frame.index))
    motor_d    = frame.get("gcs_motor_delta", pd.Series(np.nan, index=frame.index))
    eye_d      = frame.get("gcs_eye_delta",   pd.Series(np.nan, index=frame.index))

    frame["gcs_total_mean"]  = motor_mean + verb_mean + eye_mean          # 3–15 scale
    frame["gcs_motor_delta"] = motor_d                                    # trend
    frame["gcs_eye_delta"]   = eye_d                                      # trend
    frame = frame.drop(columns=[c for c in GCS_COLS if c in frame.columns], errors="ignore")
    return frame

X_train_raw = aggregate_gcs(X_train_raw)
X_test_raw  = aggregate_gcs(X_test_raw)

# Update column list after GCS aggregation
NEW_GCS_COLS = ["gcs_total_mean", "gcs_motor_delta", "gcs_eye_delta"]
NEW_GCS_COLS = [c for c in NEW_GCS_COLS if c in X_train_raw.columns]
CONTINUOUS_COLS = CONTINUOUS_COLS + NEW_GCS_COLS
print(f"GCS reduced to {len(NEW_GCS_COLS)} aggregated features.")


# ─────────────────────────────────────────────
# 4. LOG1P TRANSFORM  (on train stats → apply to test)
# ─────────────────────────────────────────────

# Winsorize vaso_max_rate at 99th percentile (train only)
if "vaso_max_rate" in X_train_raw.columns:
    cap_99 = X_train_raw["vaso_max_rate"].quantile(0.99)
    X_train_raw["vaso_max_rate"] = X_train_raw["vaso_max_rate"].clip(upper=cap_99)
    X_test_raw["vaso_max_rate"]  = X_test_raw["vaso_max_rate"].clip(upper=cap_99)
    print(f"vaso_max_rate winsorized at {cap_99:.2f}")

for col in SKEWED_COLS:
    if col in X_train_raw.columns:
        X_train_raw[col] = np.log1p(X_train_raw[col].clip(lower=0))
        X_test_raw[col]  = np.log1p(X_test_raw[col].clip(lower=0))

print(f"Log1p applied to: {SKEWED_COLS}")


# ─────────────────────────────────────────────
# 5. SKLEARN PIPELINE
#    - Continuous  : median impute → standard scale
#    - Skewed      : already log-transformed → median impute → standard scale
#    - Binary/flags: most-frequent impute (handles rare NaN)
#    - Indicators  : passthrough (already 0/1, no NaN expected)
# ─────────────────────────────────────────────

# Re-derive lists after transformations
all_current_cols = X_train_raw.columns.tolist()

INDICATOR_COLS = [c for c in all_current_cols if c.endswith("_was_missing")]
BINARY_COLS    = [
    c for c in all_current_cols
    if c not in INDICATOR_COLS
    and X_train_raw[c].dropna().isin([0, 1]).all()
    and X_train_raw[c].nunique(dropna=True) == 2
]
SKEWED_PROC    = [c for c in SKEWED_COLS if c in all_current_cols]
CONTINUOUS_PROC = [
    c for c in all_current_cols
    if c not in set(INDICATOR_COLS + BINARY_COLS + SKEWED_PROC)
    and pd.api.types.is_numeric_dtype(X_train_raw[c])
]

continuous_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

skewed_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

binary_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("continuous", continuous_pipe,  CONTINUOUS_PROC),
        ("skewed",     skewed_pipe,       SKEWED_PROC),
        ("binary",     binary_pipe,       BINARY_COLS),
        ("indicators", "passthrough",     INDICATOR_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

preprocessor.fit(X_train_raw)
X_train_pp = preprocessor.transform(X_train_raw)
X_test_pp  = preprocessor.transform(X_test_raw)

# Recover feature names
feature_names = preprocessor.get_feature_names_out()
print(f"\nAfter preprocessing: {X_train_pp.shape[1]} features")


# ─────────────────────────────────────────────
# 6. FEATURE SELECTION
#    6a. Remove near-zero variance
#    6b. Remove highly correlated pairs (r > 0.95)
# ─────────────────────────────────────────────

# 6a. Near-zero variance
vt = VarianceThreshold(threshold=0.01)
X_train_pp = vt.fit_transform(X_train_pp)
X_test_pp  = vt.transform(X_test_pp)
feature_names = feature_names[vt.get_support()]
print(f"After variance threshold: {X_train_pp.shape[1]} features")

# 6b. High-correlation filter (>0.95) — computed on train set
corr_matrix = np.corrcoef(X_train_pp, rowvar=False)
upper_tri    = np.triu(np.abs(corr_matrix), k=1)
drop_idx     = set(np.where(upper_tri > 0.95)[1])
keep_idx     = [i for i in range(X_train_pp.shape[1]) if i not in drop_idx]

X_train_pp   = X_train_pp[:, keep_idx]
X_test_pp    = X_test_pp[:, keep_idx]
feature_names = feature_names[keep_idx]
print(f"After correlation filter (r>0.95): {X_train_pp.shape[1]} features")


# ─────────────────────────────────────────────
# 7. CLASS IMBALANCE — SMOTE (train only)
# ─────────────────────────────────────────────

print(f"\nBefore SMOTE — positive rate: {y_train.mean():.3f}")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_pp, y_train)
print(f"After  SMOTE — positive rate: {y_train_bal.mean():.3f}")
print(f"Train set size after SMOTE: {X_train_bal.shape[0]:,}")


# ─────────────────────────────────────────────
# 8. SAVE OUTPUTS
# ─────────────────────────────────────────────

output_dir = os.path.join(os.getcwd(), "mimic_processed")
os.makedirs(output_dir, exist_ok=True)

# Save processed arrays
np.save(os.path.join(output_dir, "X_train.npy"), X_train_bal)
np.save(os.path.join(output_dir, "X_test.npy"),  X_test_pp)
np.save(os.path.join(output_dir, "y_train.npy"), y_train_bal)
np.save(os.path.join(output_dir, "y_test.npy"),  y_test.values)

# Save feature names
pd.Series(feature_names).to_csv(
    os.path.join(output_dir, "feature_names_final.csv"), index=False, header=["feature"]
)

# Save fitted preprocessor + filters for inference
pipeline_artifacts = {
    "preprocessor":  preprocessor,
    "variance_threshold": vt,
    "keep_idx":      keep_idx,
    "feature_names": feature_names,
    "skewed_cols":   SKEWED_COLS,
    "vaso_cap_99":   cap_99 if "vaso_max_rate" in X_train_raw.columns else None,
    "gcs_cols_dropped": GCS_COLS,
    "temp_cols_dropped": TEMP_COLS,
}
joblib.dump(pipeline_artifacts, os.path.join(output_dir, "preprocessing_pipeline.joblib"))

print(f"\n✓ Saved to {output_dir}:")
print(f"  X_train.npy  {X_train_bal.shape}")
print(f"  X_test.npy   {X_test_pp.shape}")
print(f"  y_train.npy  {y_train_bal.shape}")
print(f"  y_test.npy   {y_test.shape}")
print(f"  feature_names_final.csv ({len(feature_names)} features)")
print(f"  preprocessing_pipeline.joblib")


# ─────────────────────────────────────────────
# 9. QUICK SANITY CHECKS
# ─────────────────────────────────────────────

print("\n── Sanity checks ──")
print(f"NaN in X_train: {np.isnan(X_train_bal).sum()}")
print(f"NaN in X_test : {np.isnan(X_test_pp).sum()}")
print(f"X_train mean (should be ~0): {X_train_bal.mean():.4f}")
print(f"X_train std  (should be ~1): {X_train_bal.std():.4f}")
print(f"Class balance after SMOTE  : {np.bincount(y_train_bal.astype(int))}")
print(f"Test class balance (real)  : {np.bincount(y_test.values.astype(int))}")
print("\nPreprocessing complete.")
