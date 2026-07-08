from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "customer_segmentation_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SEGMENTED_DATA_PATH = BASE_DIR / "segmented_retail_data.pkl"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def validate_artifacts() -> None:
    missing_files = [
        path.name
        for path in [MODEL_PATH, SCALER_PATH]
        if not path.exists()
    ]
    if missing_files:
        print(f"Error: Missing artifact(s): {', '.join(missing_files)}.")
        sys.exit(1)

    try:
        model = load_pickle(MODEL_PATH)
        scaler = load_pickle(SCALER_PATH)
        segmented_data = load_pickle(SEGMENTED_DATA_PATH) if SEGMENTED_DATA_PATH.exists() else None
    except Exception as exc:
        print(f"Error: Failed to load artifacts. Details: {exc}")
        sys.exit(1)

    print("All required artifacts are present and valid.")
    print(f"Model type: {type(model).__name__}")
    print(f"Scaler type: {type(scaler).__name__}")
    print(f"Feature columns: {list(getattr(scaler, 'feature_names_in_', []))}")
    print(f"Number of model features: {getattr(model, 'n_features_in_', 'unknown')}")

    if segmented_data is not None:
        print(f"Segmented data shape: {segmented_data.shape}")
        print(f"Segmented data columns: {list(segmented_data.columns)}")

    print("\nStart the API server:")
    print("  uvicorn main:app --reload")
    print("\nStart the Streamlit UI:")
    print("  streamlit run streamlit.py")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    validate_artifacts()
