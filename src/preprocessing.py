"""Data loading, validation and cleaning for the WESAD HRV feature dataset.

Source: https://huggingface.co/datasets/GeorgiaCh96/WESAD_raw_data
(HRV features derived from the WESAD dataset, Schmidt et al., ICMI 2018).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "WESAD_raw_data.csv"
HOLDOUT_PARQUET = ROOT / "data" / "processed" / "holdout.parquet"
MODEL_PATH = ROOT / "models" / "stress_rf.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

DATASET_URL = (
    "https://huggingface.co/datasets/GeorgiaCh96/WESAD_raw_data/"
    "resolve/main/WESAD_raw_data.csv"
)

SUBJECT_COL = "subject id"
CONDITION_COL = "condition"
TIME_COL = "Time"

# 30 unique base HRV features. Excluded on purpose:
#   - log/sqrt/Box-Cox/Yeo-Johnson/square transforms (monotonic -> redundant for trees)
#   - KURT_REL_RR, SKEW_REL_RR (exact copies of KURT, SKEW in this file)
#   - Time (encodes protocol order -> leaks the condition)
#   - SSSQ (self-reported stress questionnaire score -> label leakage)
FEATURES = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR",
    "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW",
    "MEAN_REL_RR", "MEDIAN_REL_RR", "SDRR_REL_RR", "RMSSD_REL_RR",
    "SDSD_REL_RR", "SDRR_RMSSD_REL_RR",
    "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU", "HF", "HF_PCT", "HF_NU",
    "TP", "LF_HF", "HF_LF",
]

CONDITIONS = {"baseline", "stress", "amusement", "meditation"}
LABEL_NAMES = {0: "no_stress", 1: "stress"}

REQUIRED_COLUMNS = [*FEATURES, SUBJECT_COL, CONDITION_COL, TIME_COL]


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the raw dataframe does not match the expected schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    unknown = set(df[CONDITION_COL].unique()) - CONDITIONS
    if unknown:
        raise ValueError(f"Unexpected condition labels: {sorted(unknown)}")
    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, replace inf with NaN, add binary label.

    Missing values are NOT imputed here; imputation is part of the saved
    sklearn pipeline so training and inference stay identical.
    """
    out = df.drop_duplicates().copy()  # subject 10 is duplicated in the source CSV
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    out["label"] = (out[CONDITION_COL] == "stress").astype(int)
    out = out.rename(columns={SUBJECT_COL: "subject_id"})
    out = out.sort_values(["subject_id", TIME_COL]).reset_index(drop=True)
    return out[["subject_id", TIME_COL, CONDITION_COL, "label", *FEATURES]]


def features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    """Validate a single feature dict and convert it to a 1-row model input."""
    missing = [f for f in FEATURES if f not in features]
    unknown = [k for k in features if k not in FEATURES]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing features: {missing}")
        if unknown:
            parts.append(f"unknown features: {unknown}")
        raise ValueError("; ".join(parts))
    bad = [k for k, v in features.items() if v is None or not math.isfinite(float(v))]
    if bad:
        raise ValueError(f"non-finite values for features: {bad}")
    return pd.DataFrame([[float(features[f]) for f in FEATURES]], columns=FEATURES)
