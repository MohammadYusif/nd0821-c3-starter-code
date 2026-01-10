# main.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from ml.data import clean_census_dataframe, process_data
from ml.model import inference, load_artifacts
from train_model import CAT_FEATURES  # reuse same list


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"

app = FastAPI(title="Census Income Prediction API")

_model = None
_encoder = None
_lb = None


def _ensure_artifacts_loaded() -> None:
    global _model, _encoder, _lb

    if _model is not None:
        return

    # If model artifacts missing, do a quick train so the API can still run after clone.
    if not (MODEL_DIR / "trained_model.pkl").exists():
        from train_model import main as train_main

        train_main()

    _model, _encoder, _lb = load_artifacts(MODEL_DIR)


class CensusPerson(BaseModel):
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
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
    })

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")


class PredictionOut(BaseModel):
    prediction: Literal["<=50K", ">50K"]


@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict", response_model=PredictionOut)
def predict(person: CensusPerson) -> PredictionOut:
    _ensure_artifacts_loaded()

    row = person.model_dump(by_alias=True)
    df = pd.DataFrame([row])
    df = clean_census_dataframe(df)

    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=_encoder,
        lb=_lb,
    )

    pred = inference(_model, X)[0]
    # lb encodes strings; but our inference returns 0/1 due to training
    label = ">50K" if int(pred) == 1 else "<=50K"
    return PredictionOut(prediction=label)
