# ml/model.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: np.ndarray, preds: np.ndarray
) -> Tuple[float, float, float]:
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)

    return (
        float(precision),
        float(recall),
        float(fbeta),
    )


def inference(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def save_artifacts(
    model: LogisticRegression,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "trained_model.pkl")
    joblib.dump(encoder, out_dir / "encoder.pkl")
    joblib.dump(lb, out_dir / "lb.pkl")


def load_artifacts(out_dir: Path):
    model = joblib.load(out_dir / "trained_model.pkl")
    encoder = joblib.load(out_dir / "encoder.pkl")
    lb = joblib.load(out_dir / "lb.pkl")
    return model, encoder, lb
