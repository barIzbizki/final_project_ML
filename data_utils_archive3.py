from __future__ import annotations

import numpy as np
import pandas as pd

from config_archive3 import ALL_FEATURE_COLUMNS, DATA_PATH, ID_COLUMN, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, TARGET_COLUMN
from data_utils import encode_target


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
