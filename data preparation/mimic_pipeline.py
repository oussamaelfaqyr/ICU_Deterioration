"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     MIMIC-IV ICU Deterioration Detection — Full Data Processing Pipeline    ║
║     Project: planning-with-ai-bd713                                         ║
║     Dataset: physionet-data.mimiciv_3_1_icu / mimiciv_3_1_hosp              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import logging
import shutil
from pathlib import Path
from datetime import datetime

# ─── Third-party ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from google.cloud import bigquery

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these values only)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID    = "planning-with-ai-bd713"
BQ_ICU        = "physionet-data.mimiciv_3_1_icu"
BQ_HOSP       = "physionet-data.mimiciv_3_1_hosp"

# Output directory — all processed files saved here
OUTPUT_DIR    = Path("./mimic_processed")

# Observation window: first N hours of ICU stay used as features
OBS_HOURS     = 24

# Prediction horizon: detect deterioration N hours ahead (after a 2h gap)
GAP_HOURS     = 2
HORIZON_HOURS = 24

# Minimum ICU stay length to include in cohort
MIN_LOS_DAYS  = 1.0   # = 24 hours

# ─────────────────────────────────────────────────────────────────────────────
#  ITEM ID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VITAL_ITEMS = {
    220045: "hr",       # Heart Rate
    220052: "map",      # Mean Arterial Pressure (invasive)
    220179: "sbp",      # Systolic Blood Pressure
    220277: "spo2",     # SpO2 / Pulse Oximetry
    220210: "rr",       # Respiratory Rate
    223762: "temp_c",   # Temperature Celsius
    678:    "temp_f",   # Temperature Fahrenheit (convert to C)
    220739: "gcs_eye",  # GCS Eye Opening
    223901: "gcs_motor",# GCS Motor Response
    223900: "gcs_verb", # GCS Verbal Response
    223835: "fio2",     # Fraction of Inspired Oxygen
    220339: "peep",     # PEEP (ventilator)
}

VASOPRESSOR_ITEMS = [
    221906,  # Norepinephrine  ← strongest signal
    221289,  # Epinephrine
    221662,  # Dopamine
    222315,  # Vasopressin
    221749,  # Phenylephrine
    221653,  # Dobutamine
]

URINE_ITEMS = [
    226559,  # Foley
    226560,  # Void
    226561,  # Condom catheter
    226584,  # GU Irrigant Output
    226563,  # Suprapubic
    226564,  # R nephrostomy
]

INTUBATION_ITEMS  = [225792]   # Invasive Mechanical Ventilation
DIALYSIS_ITEMS    = [225441, 225805]  # Hemodialysis, CRRT

# ─────────────────────────────────────────────────────────────────────────────
#  OUTLIER CLIPPING RANGES  (physiologically plausible bounds)
# ─────────────────────────────────────────────────────────────────────────────

CLIP_RANGES = {
    "hr":       (0,   300),
    "map":      (0,   300),
    "sbp":      (0,   400),
    "spo2":     (0,   100),
    "rr":       (0,    60),
    "temp_c":   (25,   45),
    "temp_f":   (77,  113),
    "gcs_eye":  (1,     4),
    "gcs_motor":(1,     6),
    "gcs_verb": (1,     5),
    "fio2":     (0.21,  1.0),
    "peep":     (0,    35),
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mimic_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
#  BIGQUERY CLIENT
# ─────────────────────────────────────────────────────────────────────────────

client = bigquery.Client(project=PROJECT_ID)

def bq(sql: str, label: str = "") -> pd.DataFrame:
    """Execute a BigQuery SQL query and return a pandas DataFrame."""
    if label:
        log.info(f"[BQ] Running: {label}")
    job = client.query(sql)
    df  = job.to_dataframe()
    if label:
        log.info(f"[BQ] Done — {len(df):,} rows returned")
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — BUILD BASE COHORT
# ─────────────────────────────────────────────────────────────────────────────

def build_cohort() -> pd.DataFrame:
    """
    Join icustays + patients + admissions.
    Filter: LOS >= MIN_LOS_DAYS.
    Returns one row per ICU stay with demographics and exact death timestamp.
    """
    sql = f"""
    SELECT
        i.stay_id,
        i.subject_id,
        i.hadm_id,
        i.intime,
        i.outtime,
        i.first_careunit,
        ROUND(i.los, 4)                         AS los_days,

        -- Patient demographics
        p.anchor_age                            AS age,
        p.gender,
        p.dod,                                  -- date of death (date only)

        -- Admission details
        a.deathtime,                            -- exact death timestamp
        a.admittime,                            -- hospital admission time
        a.admission_type,
        a.race,
        a.hospital_expire_flag,

        -- Pre-ICU time in hospital (hours)
        ROUND(
            TIMESTAMP_DIFF(i.intime, a.admittime, MINUTE) / 60.0,
            2
        )                                       AS pre_icu_hours

    FROM `{BQ_ICU}.icustays`       i
    JOIN `{BQ_HOSP}.patients`      p USING (subject_id)
    JOIN `{BQ_HOSP}.admissions`    a USING (hadm_id)

    WHERE i.los >= {MIN_LOS_DAYS}
    ORDER BY i.intime
    """

    cohort = bq(sql, "Cohort construction")

    # Convert timestamps
    for col in ["intime", "outtime", "deathtime", "admittime"]:
        cohort[col] = pd.to_datetime(cohort[col])

    # Compute prediction window boundaries
    cohort["obs_end"]    = cohort["intime"] + pd.Timedelta(hours=OBS_HOURS)
    cohort["pred_start"] = cohort["obs_end"] + pd.Timedelta(hours=GAP_HOURS)
    cohort["pred_end"]   = cohort["pred_start"] + pd.Timedelta(hours=HORIZON_HOURS)

    log.info(f"Cohort size after LOS filter: {len(cohort):,} stays")
    return cohort


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — COMPUTE COMPOSITE DETERIORATION LABEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_labels(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Composite endpoint (label = 1) if ANY of the following occurs
    within the prediction window [pred_start, pred_end]:
        1. In-hospital death
        2. Vasopressor initiation
        3. Mechanical ventilation (intubation)
        4. Renal replacement therapy

    Also removes stays where the event ALREADY occurred inside the observation
    window (no future prediction possible).
    """
    stay_ids = ", ".join(str(s) for s in cohort["stay_id"].tolist())

    # ── Vasopressors ──────────────────────────────────────────────────────────
    vaso_items = ", ".join(str(i) for i in VASOPRESSOR_ITEMS)
    sql_vaso = f"""
    SELECT DISTINCT stay_id,
        MIN(starttime) AS vaso_start
    FROM `{BQ_ICU}.inputevents`
    WHERE stay_id IN ({stay_ids})
      AND itemid IN ({vaso_items})
      AND amount > 0
    GROUP BY stay_id
    """
    vaso = bq(sql_vaso, "Vasopressor events")
    vaso["vaso_start"] = pd.to_datetime(vaso["vaso_start"])

    # ── Mechanical ventilation ───────────────────────────────────────────────
    intub_items = ", ".join(str(i) for i in INTUBATION_ITEMS)
    sql_intub = f"""
    SELECT DISTINCT stay_id,
        MIN(starttime) AS intub_start
    FROM `{BQ_ICU}.procedureevents`
    WHERE stay_id IN ({stay_ids})
      AND itemid IN ({intub_items})
    GROUP BY stay_id
    """
    intub = bq(sql_intub, "Intubation events")
    intub["intub_start"] = pd.to_datetime(intub["intub_start"])

    # ── Renal replacement therapy ─────────────────────────────────────────────
    dialysis_items = ", ".join(str(i) for i in DIALYSIS_ITEMS)
    sql_rrt = f"""
    SELECT DISTINCT stay_id,
        MIN(starttime) AS rrt_start
    FROM `{BQ_ICU}.procedureevents`
    WHERE stay_id IN ({stay_ids})
      AND itemid IN ({dialysis_items})
    GROUP BY stay_id
    """
    rrt = bq(sql_rrt, "RRT events")
    rrt["rrt_start"] = pd.to_datetime(rrt["rrt_start"])

    # ── Merge events onto cohort ──────────────────────────────────────────────
    df = (cohort
          .merge(vaso,  on="stay_id", how="left")
          .merge(intub, on="stay_id", how="left")
          .merge(rrt,   on="stay_id", how="left"))

    # ── Label logic ───────────────────────────────────────────────────────────
    def in_window(event_time, start, end):
        """Return True if event_time falls inside (start, end]."""
        return event_time.notna() & (event_time > start) & (event_time <= end)

    # Death in prediction window
    label_death = in_window(df["deathtime"],   df["pred_start"], df["pred_end"])
    # Vasopressor in prediction window
    label_vaso  = in_window(df["vaso_start"],  df["pred_start"], df["pred_end"])
    # Intubation in prediction window
    label_intub = in_window(df["intub_start"], df["pred_start"], df["pred_end"])
    # RRT in prediction window
    label_rrt   = in_window(df["rrt_start"],   df["pred_start"], df["pred_end"])

    df["label"] = (label_death | label_vaso | label_intub | label_rrt).astype(int)

    # ── Remove stays where event ALREADY happened in observation window ───────
    event_in_obs = (
        in_window(df["deathtime"],   df["intime"], df["obs_end"]) |
        in_window(df["vaso_start"],  df["intime"], df["obs_end"]) |
        in_window(df["intub_start"], df["intime"], df["obs_end"]) |
        in_window(df["rrt_start"],   df["intime"], df["obs_end"])
    )
    n_before = len(df)
    df = df[~event_in_obs].reset_index(drop=True)
    log.info(f"Removed {n_before - len(df):,} stays with event inside obs window")

    # ── Summary ───────────────────────────────────────────────────────────────
    pos_rate = df["label"].mean() * 100
    log.info(f"Label distribution: {df['label'].sum():,} positive "
             f"/ {len(df):,} total ({pos_rate:.1f}% positive)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — EXTRACT VITAL SIGN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_vitals(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    For each vital sign in the observation window, compute:
    mean, min, max, std, first, last, delta (last-first), n_obs.
    Adds a was_missing binary indicator for each vital.
    """
    item_list = ", ".join(str(i) for i in VITAL_ITEMS.keys())
    stay_ids  = ", ".join(str(s) for s in cohort["stay_id"].tolist())

    sql = f"""
    SELECT
        c.stay_id,
        c.itemid,
        c.valuenum,
        c.charttime
    FROM `{BQ_ICU}.chartevents`  c
    JOIN `{BQ_ICU}.icustays`     i USING (stay_id)
    WHERE c.stay_id IN ({stay_ids})
      AND c.itemid  IN ({item_list})
      AND c.valuenum IS NOT NULL
      AND c.charttime BETWEEN i.intime
          AND TIMESTAMP_ADD(i.intime, INTERVAL {OBS_HOURS} HOUR)
    """
    raw = bq(sql, "Vital signs extraction")
    raw["charttime"] = pd.to_datetime(raw["charttime"])

    # ── Map itemid → vital name ───────────────────────────────────────────────
    raw["vital"] = raw["itemid"].map(VITAL_ITEMS)

    # ── Normalize temperature °F → °C ─────────────────────────────────────────
    mask_f = raw["vital"] == "temp_f"
    raw.loc[mask_f, "valuenum"] = (raw.loc[mask_f, "valuenum"] - 32) * 5 / 9
    raw.loc[mask_f, "vital"]    = "temp_c"   # merge into single temp column

    # ── Clip outliers ─────────────────────────────────────────────────────────
    for vital, (lo, hi) in CLIP_RANGES.items():
        mask = raw["vital"] == vital
        raw.loc[mask, "valuenum"] = raw.loc[mask, "valuenum"].clip(lo, hi)

    # ── Compute aggregate features per stay × vital ───────────────────────────
    def agg_vital(grp):
        vals = grp.sort_values("charttime")["valuenum"]
        return pd.Series({
            "mean":  vals.mean(),
            "min":   vals.min(),
            "max":   vals.max(),
            "std":   vals.std(),
            "first": vals.iloc[0],
            "last":  vals.iloc[-1],
            "delta": vals.iloc[-1] - vals.iloc[0],
            "n_obs": len(vals),
        })

    agg = (raw
           .groupby(["stay_id", "vital"])
           .apply(agg_vital)
           .reset_index())

    # ── Pivot to wide format: one column per vital × stat ────────────────────
    wide = agg.pivot_table(
        index="stay_id",
        columns="vital",
        values=["mean", "min", "max", "std", "first", "last", "delta", "n_obs"]
    )
    wide.columns = [f"{v}_{s}" for s, v in wide.columns]
    wide = wide.reset_index()

    # ── Add was_missing indicators ────────────────────────────────────────────
    unique_vitals = raw["vital"].unique()
    all_stays     = cohort[["stay_id"]]
    wide = all_stays.merge(wide, on="stay_id", how="left")

    for vital in unique_vitals:
        col = f"{vital}_mean"
        if col in wide.columns:
            wide[f"{vital}_was_missing"] = wide[col].isna().astype(int)

    log.info(f"Vital features shape: {wide.shape} "
             f"({wide.shape[1]-1} features × {len(wide):,} stays)")
    return wide


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — EXTRACT LAB FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_labs(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key lab values for each ICU stay from labevents (hosp module).
    Join via hadm_id. Last-value-carried-forward (LVCF) within observation window.
    """
    LAB_ITEMS = {
        50813: "lactate",
        50912: "creatinine",
        51006: "bun",
        51301: "wbc",
        51222: "hemoglobin",
        51265: "platelets",
        50983: "sodium",
        50971: "potassium",
        50882: "bicarbonate",
        50885: "bilirubin",
        51002: "troponin",
        51237: "inr",
        50820: "ph",
        50821: "pao2",
        50818: "paco2",
        50931: "glucose",
    }
    item_list = ", ".join(str(i) for i in LAB_ITEMS.keys())
    hadm_ids  = ", ".join(str(h) for h in cohort["hadm_id"].tolist())

    sql = f"""
    SELECT
        l.hadm_id,
        l.itemid,
        l.valuenum,
        l.charttime
    FROM `{BQ_HOSP}.labevents` l
    WHERE l.hadm_id  IN ({hadm_ids})
      AND l.itemid   IN ({item_list})
      AND l.valuenum IS NOT NULL
    """
    raw = bq(sql, "Lab values extraction")
    raw["charttime"] = pd.to_datetime(raw["charttime"])
    raw["lab"]       = raw["itemid"].map(LAB_ITEMS)

    # ── Keep only values within observation window ────────────────────────────
    ref = cohort[["hadm_id", "intime", "obs_end"]]
    raw = raw.merge(ref, on="hadm_id", how="left")
    raw = raw[(raw["charttime"] >= raw["intime"]) &
              (raw["charttime"] <= raw["obs_end"])].copy()

    # ── LVCF: take the last observed value per stay × lab ────────────────────
    lvcf = (raw
            .sort_values("charttime")
            .groupby(["hadm_id", "lab"])["valuenum"]
            .last()
            .reset_index()
            .rename(columns={"valuenum": "value"}))

    wide = lvcf.pivot_table(index="hadm_id", columns="lab", values="value")
    wide.columns = [f"lab_{c}" for c in wide.columns]
    wide = wide.reset_index()

    # ── Merge onto cohort (via hadm_id) ───────────────────────────────────────
    result = cohort[["stay_id", "hadm_id"]].merge(wide, on="hadm_id", how="left")
    result = result.drop(columns=["hadm_id"])

    # ── was_missing indicators ────────────────────────────────────────────────
    for lab in LAB_ITEMS.values():
        col = f"lab_{lab}"
        if col in result.columns:
            result[f"lab_{lab}_was_missing"] = result[col].isna().astype(int)

    log.info(f"Lab features shape: {result.shape}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — FLUID BALANCE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_fluid_balance(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cumulative fluid balance and urine output rate
    for each ICU stay over the observation window.
    """
    stay_ids     = ", ".join(str(s) for s in cohort["stay_id"].tolist())
    urine_items  = ", ".join(str(i) for i in URINE_ITEMS)

    # ── Total fluid input ─────────────────────────────────────────────────────
    sql_input = f"""
    SELECT inp.stay_id,
        SUM(inp.amount) AS total_input_ml
    FROM `{BQ_ICU}.inputevents` inp
    JOIN `{BQ_ICU}.icustays`    i USING (stay_id)
    WHERE inp.stay_id   IN ({stay_ids})
      AND inp.amountuom = 'mL'
      AND inp.starttime BETWEEN i.intime
          AND TIMESTAMP_ADD(i.intime, INTERVAL {OBS_HOURS} HOUR)
    GROUP BY inp.stay_id
    """
    fluid_in = bq(sql_input, "Fluid input")

    # ── Urine output ──────────────────────────────────────────────────────────
    sql_urine = f"""
    SELECT o.stay_id,
        SUM(o.value)   AS total_urine_ml,
        COUNT(*)       AS urine_n_obs
    FROM `{BQ_ICU}.outputevents` o
    JOIN `{BQ_ICU}.icustays`     i USING (stay_id)
    WHERE o.stay_id IN ({stay_ids})
      AND o.itemid  IN ({urine_items})
      AND o.charttime BETWEEN i.intime
          AND TIMESTAMP_ADD(i.intime, INTERVAL {OBS_HOURS} HOUR)
    GROUP BY o.stay_id
    """
    urine = bq(sql_urine, "Urine output")

    # ── Merge & compute balance ───────────────────────────────────────────────
    result = (cohort[["stay_id"]]
              .merge(fluid_in, on="stay_id", how="left")
              .merge(urine,    on="stay_id", how="left"))

    result["fluid_balance_ml"] = (
        result["total_input_ml"].fillna(0) - result["total_urine_ml"].fillna(0)
    )
    result["urine_rate_ml_hr"] = result["total_urine_ml"] / OBS_HOURS

    # Flags
    result["fluid_negative_flag"]   = (result["fluid_balance_ml"] < 0).astype(int)
    result["oliguria_flag"]         = (result["urine_rate_ml_hr"] < 0.5 * 70).astype(int)
    # Note: 0.5 mL/kg/h × 70 kg average weight; ideally use patientweight from inputevents

    log.info(f"Fluid balance features shape: {result.shape}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — VASOPRESSOR FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_vasopressors(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag for any vasopressor use in observation window.
    Also computes max rate (haemodynamic severity proxy).
    """
    stay_ids   = ", ".join(str(s) for s in cohort["stay_id"].tolist())
    vaso_items = ", ".join(str(i) for i in VASOPRESSOR_ITEMS)

    sql = f"""
    SELECT inp.stay_id,
        1                    AS vaso_flag,
        MAX(inp.rate)        AS vaso_max_rate
    FROM `{BQ_ICU}.inputevents` inp
    JOIN `{BQ_ICU}.icustays`    i USING (stay_id)
    WHERE inp.stay_id IN ({stay_ids})
      AND inp.itemid  IN ({vaso_items})
      AND inp.amount  > 0
      AND inp.starttime BETWEEN i.intime
          AND TIMESTAMP_ADD(i.intime, INTERVAL {OBS_HOURS} HOUR)
    GROUP BY inp.stay_id
    """
    vaso = bq(sql, "Vasopressor features")

    result = cohort[["stay_id"]].merge(vaso, on="stay_id", how="left")
    result["vaso_flag"]     = result["vaso_flag"].fillna(0).astype(int)
    result["vaso_max_rate"] = result["vaso_max_rate"].fillna(0)

    log.info(f"Vasopressor flag: {result['vaso_flag'].sum():,} stays with vasopressors in obs window")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — STATIC DEMOGRAPHIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_static_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical static features.
    Returns cleaned static feature columns.
    """
    static = cohort[[
        "stay_id", "age", "gender", "first_careunit",
        "admission_type", "pre_icu_hours", "los_days"
    ]].copy()

    # Binary encode gender
    static["gender_male"] = (static["gender"] == "M").astype(int)

    # Binary emergency flag
    static["emergency_flag"] = static["admission_type"].str.contains(
        "EMERGENCY|URGENT", case=False, na=False
    ).astype(int)

    # One-hot encode care unit
    care_units = ["MICU", "SICU", "CCU", "CVICU", "NICU", "TSICU", "MICU/SICU"]
    for unit in care_units:
        col = "unit_" + unit.replace("/", "_").lower()
        static[col] = (static["first_careunit"] == unit).astype(int)

    # Drop raw categoricals
    static = static.drop(columns=["gender", "first_careunit", "admission_type"])

    log.info(f"Static features shape: {static.shape}")
    return static


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — IMPUTE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    Impute all numeric NaN values using training-set medians.
    was_missing indicators must already be added before calling this function.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifier and label columns
    exclude = {"stay_id", "label"}
    impute_cols = [c for c in numeric_cols if c not in exclude]

    # Compute medians on training set only to prevent leakage
    train_medians = df.loc[train_mask, impute_cols].median()

    df[impute_cols] = df[impute_cols].fillna(train_medians)

    n_remaining = df[impute_cols].isna().sum().sum()
    log.info(f"Imputation complete. Remaining NaN: {n_remaining}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 — TEMPORAL TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by intime, then split 70% / 15% / 15% chronologically.
    NEVER use random splits — temporal leakage inflates metrics by 5-15 AUROC pts.
    """
    df = df.sort_values("intime").reset_index(drop=True)
    n  = len(df)

    n_train = int(0.70 * n)
    n_val   = int(0.85 * n)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_val].copy()
    test  = df.iloc[n_val:].copy()

    for split_name, split_df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        pos = split_df["label"].mean() * 100
        log.info(f"{split_name}: {len(split_df):,} stays | "
                 f"{split_df['label'].sum():,} positive ({pos:.1f}%)")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 10 — SAVE PROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, name: str, fmt: str = "parquet") -> Path:
    """
    Save a DataFrame to disk. Supported formats: parquet (default), csv, json.
    Returns the path where the file was saved.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M")
    fpath = OUTPUT_DIR / f"{name}_{ts}.{fmt}"
    stable_path = OUTPUT_DIR / f"{name}.{fmt}"

    if fmt == "parquet":
        df.to_parquet(fpath, index=False, compression="snappy")
        df.to_parquet(stable_path, index=False, compression="snappy")
    elif fmt == "csv":
        df.to_csv(fpath, index=False)
        df.to_csv(stable_path, index=False)
    elif fmt == "json":
        df.to_json(fpath, orient="records", lines=True)
        df.to_json(stable_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    size_mb = fpath.stat().st_size / 1_048_576
    stable_mb = stable_path.stat().st_size / 1_048_576
    log.info(f"Saved {name} → {fpath}  ({size_mb:.2f} MB,  {len(df):,} rows)")
    log.info(f"Updated canonical artifact → {stable_path}  ({stable_mb:.2f} MB)")
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 70)
    log.info("   MIMIC-IV ICU Deterioration Pipeline — START")
    log.info(f"   Project : {PROJECT_ID}")
    log.info(f"   Obs window: {OBS_HOURS}h  |  Gap: {GAP_HOURS}h  |  Horizon: {HORIZON_HOURS}h")
    log.info("=" * 70)

    # ── Step 1 — Cohort ───────────────────────────────────────────────────────
    cohort = build_cohort()
    save(cohort, "01_cohort")

    # ── Step 2 — Labels ───────────────────────────────────────────────────────
    labeled = compute_labels(cohort)
    save(labeled, "02_labeled_cohort")

    # ── Step 3 — Vitals ───────────────────────────────────────────────────────
    vitals = extract_vitals(labeled)
    save(vitals, "03_vital_features")

    # ── Step 4 — Labs ─────────────────────────────────────────────────────────
    labs = extract_labs(labeled)
    save(labs, "04_lab_features")

    # ── Step 5 — Fluid balance ────────────────────────────────────────────────
    fluids = extract_fluid_balance(labeled)
    save(fluids, "05_fluid_features")

    # ── Step 6 — Vasopressors (feature version, inside obs window) ───────────
    vaso_feat = extract_vasopressors(labeled)
    save(vaso_feat, "06_vasopressor_features")

    # ── Step 7 — Static features ──────────────────────────────────────────────
    static = build_static_features(labeled)
    save(static, "07_static_features")

    # ── Step 8 — Merge all features ───────────────────────────────────────────
    log.info("Merging all feature tables...")
    final = labeled[["stay_id", "intime", "label"]].copy()
    for feat_df in [static, vitals, labs, fluids, vaso_feat]:
        final = final.merge(feat_df, on="stay_id", how="left")

    log.info(f"Final feature matrix: {final.shape[0]:,} rows × {final.shape[1]} columns")
    save(final, "08_feature_matrix_raw")

    # ── Step 9 — Temporal split ───────────────────────────────────────────────
    train, val, test = temporal_split(final)

    # ── Step 10 — Impute (using training set medians only) ───────────────────
    train_mask = final.index.isin(train.index)
    final = impute(final, train_mask)

    # Re-split after imputation
    train = final.iloc[:int(0.70 * len(final))].copy()
    val   = final.iloc[int(0.70 * len(final)):int(0.85 * len(final))].copy()
    test  = final.iloc[int(0.85 * len(final)):].copy()

    # ── Step 11 — Save final splits ───────────────────────────────────────────
    save(train, "09_train")
    save(val,   "09_val")
    save(test,  "09_test")

    # Save column reference list
    feat_cols = [c for c in final.columns if c not in {"stay_id", "intime", "label"}]
    col_ref   = pd.DataFrame({"feature": feat_cols})
    save(col_ref, "10_feature_list", fmt="csv")

    log.info("=" * 70)
    log.info(f"   Pipeline complete. All files saved to: {OUTPUT_DIR.resolve()}")
    log.info(f"   Feature count : {len(feat_cols)}")
    log.info(f"   Train rows    : {len(train):,}")
    log.info(f"   Val rows      : {len(val):,}")
    log.info(f"   Test rows     : {len(test):,}")
    log.info("=" * 70)

    return train, val, test, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train, val, test, feat_cols = run_pipeline()
