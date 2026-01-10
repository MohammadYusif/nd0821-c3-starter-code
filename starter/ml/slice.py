# ml/slice.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ml.data import clean_census_dataframe, process_data
from ml.model import compute_model_metrics, inference


def compute_slices_and_write(
    data: pd.DataFrame,
    categorical_features: List[str],
    slice_feature: str,
    model,
    encoder,
    lb,
    out_path: Path,
    label: str = "salary",
) -> None:
    data = clean_census_dataframe(data)

    lines: list[str] = []
    for value in sorted(data[slice_feature].dropna().unique()):
        df_slice = data[data[slice_feature] == value]
        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        p, r, f = compute_model_metrics(y_slice, preds)
        lines.append(
            f"{slice_feature}={value} | "
            f"precision={p:.4f} recall={r:.4f} fbeta={f:.4f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
