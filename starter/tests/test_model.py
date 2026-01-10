# tests/test_model.py
import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _tiny_df():
    return pd.DataFrame(
        [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "salary": "<=50K",
            },
            {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 13,
                "native-country": "United-States",
                "salary": "<=50K",
            },
            {
                "age": 38,
                "workclass": "Private",
                "fnlgt": 215646,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Divorced",
                "occupation": "Handlers-cleaners",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "salary": "<=50K",
            },
            {
                "age": 53,
                "workclass": "Private",
                "fnlgt": 234721,
                "education": "11th",
                "education-num": 7,
                "marital-status": "Married-civ-spouse",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "Black",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "salary": ">50K",
            },
        ]
    )


def test_train_model_returns_fitted_model():
    df = _tiny_df()
    X, y, encoder, lb = process_data(df, CAT_FEATURES, label="salary", training=True)
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_inference_shape_matches_rows():
    df = _tiny_df()
    X, y, encoder, lb = process_data(df, CAT_FEATURES, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_returns_three_floats():
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 0])
    p, r, f = compute_model_metrics(y, preds)
    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(f, float)
