# ml/data.py
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def clean_census_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw census dataframe:
    - strip whitespace from column names
    - strip whitespace from string values
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    return df


def process_data(
    data: pd.DataFrame,
    categorical_features: List[str],
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], OneHotEncoder, Optional[LabelBinarizer]]:
    """
    Process the census data using one-hot encoding for categorical features and
    a label binarizer for the target label.
    """
    data = clean_census_dataframe(data)

    X = data.drop(columns=[label]) if label and label in data.columns else data.copy()
    y = data[label].values if label and label in data.columns else None

    X_categorical = X[categorical_features].copy()
    X_continuous = X.drop(columns=categorical_features).copy()

    if training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = encoder.fit_transform(X_categorical)

        if y is not None:
            lb = LabelBinarizer()
            y = lb.fit_transform(y).ravel()
    else:
        if encoder is None:
            raise ValueError("encoder must be provided when training=False")
        X_cat = encoder.transform(X_categorical)

        if y is not None:
            if lb is None:
                raise ValueError(
                    "lb must be provided when training=False and label exists"
                )
            y = lb.transform(y).ravel()

    X_cont = X_continuous.to_numpy(dtype=float)
    X_processed = np.concatenate([X_cont, X_cat], axis=1)

    return X_processed, y, encoder, lb
