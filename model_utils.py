"""Loads (or rebuilds) the exact model from logistic_reg.ipynb."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = Path(__file__).parent
MODEL_PATH = BASE / "logistic_reg.pkl"
SCALER_PATH = BASE / "scaler.pkl"
METRICS_PATH = BASE / "metrics.json"
DATA_PATH = BASE / "diabetes_prediction_dataset.csv"

# Column order must match the notebook's X = df.drop('diabetes', axis=1)
FEATURES = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

# LabelEncoder sorts classes alphabetically (uppercase first) -> these codes
GENDER_CODES = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_CODES = {"No Info": 0, "current": 1, "ever": 2,
                 "former": 3, "never": 4, "not current": 5}


def train_from_csv():
    """Same steps as the notebook: encode -> 80/20 split -> scale -> fit."""
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    X, y = df[FEATURES], df["diabetes"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(scaler.transform(X_tr), y_tr)
    pred = clf.predict(scaler.transform(X_te))
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics))
    return clf, scaler, metrics


def load_artifacts():
    """Use saved .pkl files if both exist, otherwise rebuild from the CSV."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        clf, scaler = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
        return clf, scaler, metrics
    if DATA_PATH.exists():
        return train_from_csv()
    raise FileNotFoundError(
        "Need logistic_reg.pkl + scaler.pkl, or diabetes_prediction_dataset.csv "
        "in the app folder.")
