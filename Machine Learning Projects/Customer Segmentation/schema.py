from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ModelRequest(BaseModel):
    title: str = Field(default="Customer Segmentation")
    description: str | None = Field(
        default="API for customer segmentation using machine learning"
    )


class CustomerData(BaseModel):
    quantity: float = Field(
        ...,
        gt=0,
        example=6,
        description="Number of items purchased in the transaction.",
    )
    unit_price: float = Field(
        ...,
        ge=0,
        example=2.55,
        description="Price per unit for the purchased item.",
    )
    country: str = Field(
        default="United Kingdom",
        example="United Kingdom",
        description="Customer country from the training dataset.",
    )
    country_encoded: int | None = Field(
        default=None,
        ge=0,
        example=35,
        description="Optional encoded country value. If omitted, the API derives it from country.",
    )
    total_amount: float | None = Field(
        default=None,
        ge=0,
        example=15.3,
        description="Optional transaction total. If omitted, quantity * unit_price is used.",
    )

    @model_validator(mode="after")
    def normalize_country(self) -> "CustomerData":
        self.country = self.country.strip()
        if not self.country and self.country_encoded is None:
            raise ValueError("country is required when country_encoded is not supplied")
        return self


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerData] = Field(..., min_length=1, max_length=100)


class SegmentProfile(BaseModel):
    segment: int
    label: str
    description: str
    records: int | None = None
    avg_quantity: float | None = None
    avg_unit_price: float | None = None
    avg_total_amount: float | None = None


class PredictionResponse(BaseModel):
    segment: int
    label: str
    confidence: float | None = Field(
        default=None,
        description="Distance-derived fit score. Higher means closer to the predicted cluster center.",
    )
    features: dict[str, float]
    profile: SegmentProfile


class ModelMetadata(BaseModel):
    feature_columns: list[str]
    countries: dict[str, int]
    segments: list[SegmentProfile]
    training_records: int | None = None
