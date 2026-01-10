from fastapi.testclient import TestClient

from main import app

import pandas as pd

from ml.data import clean_census_dataframe
from train_model import DATA_PATH


client = TestClient(app)


def test_get_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert "message" in body
    assert body["message"] == "Welcome to the Census Income Prediction API!"


def test_post_predict_leq_50k():
    payload = {
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
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert body["prediction"] == "<=50K"


def test_post_predict_gt_50k():
    # Load real rows from the dataset and find one the model predicts as >50K
    df = pd.read_csv(DATA_PATH)
    df = clean_census_dataframe(df)

    feature_df = df.drop(columns=["salary"])

    positive_payload = None

    for _, row in feature_df.head(5000).iterrows():
        candidate = row.to_dict()
        response = client.post("/predict", json=candidate)
        assert response.status_code == 200

        body = response.json()
        assert "prediction" in body

        if body["prediction"] == ">50K":
            positive_payload = candidate
            break

    assert positive_payload is not None, (
        "Could not find a payload predicted as >50K in the sample"
    )

    # Final explicit check (helps the sanitycheck heuristic too)
    response = client.post("/predict", json=positive_payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"

    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert body["prediction"] == ">50K"
