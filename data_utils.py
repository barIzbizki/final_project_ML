from __future__ import annotations

import numpy as np
import pandas as pd

from config import ALL_FEATURE_COLUMNS, DATA_PATH, ID_COLUMN, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, TARGET_COLUMN


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def encode_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        unique_values = set(target.dropna().unique())
        if unique_values.issubset({0, 1}):
            return target.astype(int)
    normalized = target.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "no": 0, "presence": 1, "absence": 0,
        "positive": 1, "negative": 0, "disease": 1,
        "no disease": 0, "1": 1, "0": 0,
    }
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        unknown = normalized[encoded.isna()].unique()
        raise ValueError(f"Unknown target values: {unknown}")
    return encoded.astype(int)


def load_original_data() -> pd.DataFrame:
    ensure_directories()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    required = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])
    if ID_COLUMN not in df.columns:
        df.insert(0, ID_COLUMN, np.arange(1, len(df)+1))
    return df
