"""Loading and cleaning of the UCI Heart Disease data (processed.*.data files)."""

import numpy as np
import pandas as pd

from heartnn.settings import DATA_DIR

RAW_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
SOURCES = ["cleveland", "hungarian", "switzerland", "va"]

NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BIN_COLS = ["sex", "fbs", "exang"]
NOM_COLS = ["cp", "restecg", "thal"]
ORD_COLS = ["slope", "ca"]
FEATURES = NUM_COLS + BIN_COLS + NOM_COLS + ORD_COLS


def load_raw():
    frames = []
    for src in SOURCES:
        d = pd.read_csv(DATA_DIR / f"processed.{src}.data", names=RAW_COLS, na_values="?")
        d["source"] = src
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def clean(df_raw):
    """Fixes that need no statistics from the data, so they are safe before the split."""
    df = df_raw.drop_duplicates().copy()

    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df.loc[df["chol"] == 0, "chol"] = np.nan
    
    # Binary classification
    y = (df["num"] > 0).astype(int)
    return df[FEATURES], y


def load_clean_data():
    return clean(load_raw())
