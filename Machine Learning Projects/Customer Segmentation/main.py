from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schema import (
    BatchPredictionRequest,
    CustomerData,
    ModelMetadata,
    PredictionResponse,
    SegmentProfile,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "customer_segmentation_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SEGMENTED_DATA_PATH = BASE_DIR / "segmented_retail_data.pkl"

FEATURE_COLUMNS = ["Quantity", "UnitPrice", "TotalAmount", "Country_Encoded"]
DEFAULT_COUNTRY = "United Kingdom"

SEGMENT_LABELS = {
    0: "Core Retail Buyers",
    1: "Exceptional Bulk Buyers",
    2: "International Growth Buyers",
    3: "Premium High-Value Buyers",
}

SEGMENT_DESCRIPTIONS = {
    0: "Frequent everyday purchases with moderate basket value. Best for retention, replenishment offers, and cross-sell campaigns.",
    1: "Rare, very large quantity orders with extreme revenue impact. Best handled with direct account care and stock planning.",
    2: "Higher-value international purchases. Good fit for localized offers, shipping incentives, and market expansion tests.",
    3: "Low quantity but very high unit-price orders. Best for premium product launches, concierge support, and margin-focused campaigns.",
}


app = FastAPI(
    title="Customer Segmentation API",
    version="1.0.0",
    description="Predict retail customer segments from transaction-level purchase features.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact: {path.name}")

    with path.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def get_artifacts() -> tuple[Any, Any, pd.DataFrame | None]:
    try:
        model = _load_pickle(MODEL_PATH)
        scaler = _load_pickle(SCALER_PATH)
        segmented_data = _load_pickle(SEGMENTED_DATA_PATH) if SEGMENTED_DATA_PATH.exists() else None
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc

    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in getattr(scaler, "feature_names_in_", FEATURE_COLUMNS)]
    if missing_features:
        raise RuntimeError(f"Scaler artifact is missing expected features: {missing_features}")

    return model, scaler, segmented_data


def get_country_encoding() -> dict[str, int]:
    _, _, segmented_data = get_artifacts()
    if segmented_data is None or not {"Country", "Country_Encoded"}.issubset(segmented_data.columns):
        return {DEFAULT_COUNTRY: 35}

    country_rows = segmented_data[["Country", "Country_Encoded"]].drop_duplicates()
    return {
        str(row.Country): int(row.Country_Encoded)
        for row in country_rows.sort_values("Country_Encoded").itertuples(index=False)
    }


def build_feature_frame(customer: CustomerData) -> pd.DataFrame:
    country_encoding = get_country_encoding()
    country_encoded = customer.country_encoded

    if country_encoded is None:
        if customer.country not in country_encoding:
            allowed = ", ".join(country_encoding.keys())
            raise HTTPException(
                status_code=422,
                detail=f"Unknown country '{customer.country}'. Use one of: {allowed}, or provide country_encoded.",
            )
        country_encoded = country_encoding[customer.country]

    total_amount = customer.total_amount
    if total_amount is None:
        total_amount = customer.quantity * customer.unit_price

    return pd.DataFrame(
        [
            {
                "Quantity": customer.quantity,
                "UnitPrice": customer.unit_price,
                "TotalAmount": total_amount,
                "Country_Encoded": country_encoded,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def profile_for_segment(segment: int) -> SegmentProfile:
    _, _, segmented_data = get_artifacts()
    label = SEGMENT_LABELS.get(segment, f"Segment {segment}")
    description = SEGMENT_DESCRIPTIONS.get(segment, "Machine-learning generated customer segment.")

    if segmented_data is None or "Segment" not in segmented_data.columns:
        return SegmentProfile(segment=segment, label=label, description=description)

    rows = segmented_data[segmented_data["Segment"] == segment]
    if rows.empty:
        return SegmentProfile(segment=segment, label=label, description=description)

    return SegmentProfile(
        segment=segment,
        label=label,
        description=description,
        records=int(len(rows)),
        avg_quantity=float(rows["Quantity"].mean()),
        avg_unit_price=float(rows["UnitPrice"].mean()),
        avg_total_amount=float(rows["TotalAmount"].mean()),
    )


def predict_one(customer: CustomerData) -> PredictionResponse:
    model, scaler, _ = get_artifacts()
    feature_frame = build_feature_frame(customer)
    scaled_features = scaler.transform(feature_frame)
    segment = int(model.predict(scaled_features)[0])

    distances = getattr(model, "transform", lambda values: np.array([]))(scaled_features)
    confidence = None
    if distances.size:
        nearest_distance = float(np.min(distances[0]))
        confidence = float(1 / (1 + nearest_distance))

    return PredictionResponse(
        segment=segment,
        label=SEGMENT_LABELS.get(segment, f"Segment {segment}"),
        confidence=confidence,
        features={column: float(feature_frame.iloc[0][column]) for column in FEATURE_COLUMNS},
        profile=profile_for_segment(segment),
    )


@app.on_event("startup")
def warm_artifacts() -> None:
    get_artifacts()


@app.get("/", tags=["Status"])
def root() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "message": "Customer Segmentation API is running.",
            "docs": "/docs",
            "health": "/health",
        },
    )


@app.get("/health", tags=["Status"])
def health() -> dict[str, str]:
    get_artifacts()
    return {"status": "ok", "model": MODEL_PATH.name, "scaler": SCALER_PATH.name}


@app.get("/metadata", response_model=ModelMetadata, tags=["Model"])
def metadata() -> ModelMetadata:
    model, scaler, segmented_data = get_artifacts()
    model_labels = getattr(model, "labels_", [])
    unique_segments = sorted({int(segment) for segment in model_labels}) if len(model_labels) else sorted(SEGMENT_LABELS.keys())

    return ModelMetadata(
        feature_columns=list(getattr(scaler, "feature_names_in_", FEATURE_COLUMNS)),
        countries=get_country_encoding(),
        segments=[profile_for_segment(segment) for segment in unique_segments],
        training_records=int(len(segmented_data)) if segmented_data is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
def predict(customer_data: CustomerData) -> PredictionResponse:
    try:
        return predict_one(customer_data)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Model"])
def predict_batch(payload: BatchPredictionRequest) -> list[PredictionResponse]:
    if not payload.customers:
        raise HTTPException(status_code=422, detail="At least one customer is required.")

    return [predict_one(customer) for customer in payload.customers]
