"""Imputation, encoding and scaling. Fit on the training split only."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartnn.data import BIN_COLS, NOM_COLS, NUM_COLS, ORD_COLS


def build_preprocessor():
    num_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ])

    bin_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
        ])

    nom_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=-1)),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    ord_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ])

    return ColumnTransformer([
            ("num", num_pipeline, NUM_COLS),
            ("bin", bin_pipeline, BIN_COLS),
            ("nom", nom_pipeline, NOM_COLS),
            ("ord", ord_pipeline, ORD_COLS),
        ])
