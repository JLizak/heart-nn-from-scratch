"""Imputation, encoding and scaling. Fit on the training split only."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartnn.data import BIN_COLS, NOM_COLS, NUM_COLS, ORD_COLS, load_clean_data


def _numeric_pipeline(normalize):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if normalize:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps)


def build_preprocessor(normalize=True):
    num_pipeline = _numeric_pipeline(normalize)

    bin_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
        ])

    nom_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=-1)),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    ord_pipeline = _numeric_pipeline(normalize)

    return ColumnTransformer([
            ("num", num_pipeline, NUM_COLS),
            ("bin", bin_pipeline, BIN_COLS),
            ("nom", nom_pipeline, NOM_COLS),
            ("ord", ord_pipeline, ORD_COLS),
        ])


def load_splits(normalize=True, random_state=42):
    """Stratified 60/20/20 split: X_train, X_val, X_test, y_train, y_val, y_test."""
    X, y = load_clean_data()

    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=random_state
    )

    pre = build_preprocessor(normalize)
    X_train = pre.fit_transform(X_train)
    X_val = pre.transform(X_val)
    X_test = pre.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_column_splits(normalize=True, random_state=42):
    """load_splits with y as float column vectors (N, 1), the shape MLP works with."""
    *X_splits, y_train, y_val, y_test = load_splits(normalize, random_state)
    y_splits = [y.to_numpy(dtype=float).reshape(-1, 1) for y in (y_train, y_val, y_test)]
    return (*X_splits, *y_splits)
