from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "placement.csv"

TARGET = "status"

# mba_p and specialisation are NOT used: our target users are undergrads
# who have not done a master's yet, so they can't answer those questions.
NUM_COLS = ["ssc_p", "hsc_p", "degree_p", "etest_p"]

CAT_COLS = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex"]


def load_data(path=DATA_PATH):
    """Load the CSV and do the basic cleaning."""
    df = pd.read_csv(path)

    # sl_no is just a row number -> not useful
    # salary exists ONLY for placed students -> it would leak the answer, so we drop it
    df = df.drop(columns=["sl_no", "salary", "mba_p", "specialisation"], errors="ignore")

    # Target: Placed -> 1 , Not Placed -> 0
    df[TARGET] = df[TARGET].map({"Placed": 1, "Not Placed": 0})

    return df


def split_xy(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y


def build_preprocessor():
    """
    Numeric columns     -> fill missing with median, then scale
    Categorical columns -> fill missing with most frequent, then one-hot encode

    Everything is inside a Pipeline, so the same steps are applied
    to train, test, and to new data coming from the app (no data leakage).
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
