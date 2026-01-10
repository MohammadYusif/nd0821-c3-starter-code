# train_model.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import clean_census_dataframe, process_data
from ml.model import compute_model_metrics, inference, save_artifacts, train_model
from ml.slice import compute_slices_and_write


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"
SLICE_OUT = ROOT / "slice_output.txt"

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


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = clean_census_dataframe(data)

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["salary"],
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(
        f"Test metrics: precision={precision:.4f} "
        f"recall={recall:.4f} "
        f"fbeta={fbeta:.4f}"
    )
    save_artifacts(model, encoder, lb, MODEL_DIR)

    # Required: compute slice metrics and write to slice_output.txt
    compute_slices_and_write(
        data=test,
        categorical_features=CAT_FEATURES,
        slice_feature="education",
        model=model,
        encoder=encoder,
        lb=lb,
        out_path=SLICE_OUT,
        label="salary",
    )
    print(f"Wrote slice metrics to: {SLICE_OUT}")


if __name__ == "__main__":
    main()
