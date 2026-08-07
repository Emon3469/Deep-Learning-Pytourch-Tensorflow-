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
from fastapi.responses import HTMLResponse, JSONResponse

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


def build_dashboard_html() -> str:
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Customer Segmentation Studio</title>
    <style>
        :root {
            --bg: #f6f3eb;
            --panel: rgba(255, 255, 255, 0.9);
            --line: rgba(34, 37, 33, 0.12);
            --ink: #1d2320;
            --muted: #68736e;
            --accent: #164b3d;
            --accent-2: #b86d38;
            --shadow: 0 28px 80px rgba(29, 35, 32, 0.12);
            --radius: 24px;
        }

        * { box-sizing: border-box; }
        html, body { margin: 0; min-height: 100%; }
        body {
            font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(22, 75, 61, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(184, 109, 56, 0.16), transparent 24%),
                linear-gradient(180deg, #f8f5ee 0%, #f0eadf 100%);
        }

        .page {
            width: min(1240px, calc(100% - 32px));
            margin: 0 auto;
            padding: 30px 0 40px;
        }

        .hero {
            border: 1px solid var(--line);
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(20, 34, 29, 0.96), rgba(22, 75, 61, 0.94));
            color: #fff8ef;
            padding: 30px;
            box-shadow: var(--shadow);
        }

        .eyebrow {
            margin: 0 0 10px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            color: rgba(255, 248, 239, 0.78);
            font-weight: 800;
        }

        h1, h2, h3 { margin: 0; }
        h1 {
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2.2rem, 5vw, 4.4rem);
            line-height: 0.96;
            max-width: 12ch;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 18px;
            margin-top: 20px;
            align-items: end;
        }

        .hero-copy {
            color: rgba(255, 248, 239, 0.84);
            max-width: 70ch;
            line-height: 1.6;
            margin: 14px 0 0;
        }

        .pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
        .pill {
            border: 1px solid rgba(255, 248, 239, 0.18);
            background: rgba(255, 248, 239, 0.08);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .panel-grid {
            display: grid;
            grid-template-columns: 0.95fr 1.05fr 0.85fr;
            gap: 18px;
            margin-top: 18px;
        }

        .panel {
            border: 1px solid var(--line);
            border-radius: var(--radius);
            background: var(--panel);
            box-shadow: var(--shadow);
            padding: 22px;
            backdrop-filter: blur(10px);
        }

        .status {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 12px;
            line-height: 1.5;
        }

        .chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 12px;
            background: #eef5f1;
            color: var(--accent);
            font-size: 0.86rem;
            font-weight: 700;
        }

        .stat {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 12px;
            background: linear-gradient(180deg, rgba(22, 75, 61, 0.04), rgba(22, 75, 61, 0.01));
        }

        .stat strong {
            display: block;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2rem;
            color: var(--accent);
        }

        .stat span { color: var(--muted); font-size: 0.88rem; }

        .form-grid { display: grid; gap: 12px; }
        .field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
        label { display: block; font-size: 0.86rem; font-weight: 800; margin-bottom: 6px; }
        input, select, button, textarea {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 16px;
            font: inherit;
            min-height: 50px;
            background: #fffdf8;
            color: var(--ink);
        }
        input, select, textarea { padding: 12px 14px; }
        textarea { min-height: 104px; resize: vertical; }
        button {
            border: 0;
            background: var(--accent);
            color: white;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 24px rgba(22, 75, 61, 0.18);
        }
        button:hover { background: #0f3c31; }
        .secondary { background: #b86d38; }
        .secondary:hover { background: #9d572a; }

        .result-box, .list-box {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: #fffefb;
            padding: 16px;
            margin-top: 14px;
        }

        .result-box h3, .list-box h3 { margin-bottom: 10px; }
        .badge-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 0; }
        .badge {
            border-radius: 999px;
            padding: 7px 10px;
            background: #eef5f1;
            color: var(--accent);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .muted { color: var(--muted); }
        .divider { height: 1px; background: var(--line); margin: 16px 0; }
        .small { font-size: 0.84rem; }

        @media (max-width: 980px) {
            .hero-grid, .panel-grid, .field-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <p class="eyebrow">Retail Segmentation Studio</p>
            <div class="hero-grid">
                <div>
                    <h1>Customer segmentation with a cleaner, lighter UI.</h1>
                    <p class="hero-copy">
                        Predict retail customer segments from transaction features, inspect model metadata, and
                        review the output in a dashboard that is ready for Render without extra frontend tooling.
                    </p>
                    <div class="pill-row">
                        <span class="pill">FastAPI</span>
                        <span class="pill">Lightweight HTML</span>
                        <span class="pill">Render-ready</span>
                        <span class="pill">Single-file UI</span>
                    </div>
                </div>
                <div>
                    <div class="status" id="healthStatus">Loading service status...</div>
                </div>
            </div>
        </section>

        <section class="panel-grid">
            <aside class="panel">
                <p class="eyebrow">Model Snapshot</p>
                <div class="stat"><strong id="featureCount">-</strong><span>Feature columns</span></div>
                <div class="stat"><strong id="segmentCount">-</strong><span>Known segments</span></div>
                <div class="stat"><strong id="countryCount">-</strong><span>Mapped countries</span></div>
                <p class="muted small" id="trainingRecords"></p>
            </aside>

            <section class="panel">
                <p class="eyebrow">Predict</p>
                <form id="predictForm" class="form-grid">
                    <div class="field-grid">
                        <div>
                            <label for="quantity">Quantity</label>
                            <input id="quantity" type="number" min="0.01" step="0.01" value="6" required>
                        </div>
                        <div>
                            <label for="unitPrice">Unit price</label>
                            <input id="unitPrice" type="number" min="0" step="0.01" value="2.55" required>
                        </div>
                    </div>
                    <div class="field-grid">
                        <div>
                            <label for="country">Country</label>
                            <select id="country"></select>
                        </div>
                        <div>
                            <label for="countryEncoded">Country encoded override</label>
                            <input id="countryEncoded" type="number" min="0" step="1" placeholder="Optional">
                        </div>
                    </div>
                    <div>
                        <label for="totalAmount">Total amount</label>
                        <input id="totalAmount" type="number" min="0" step="0.01" placeholder="Optional. Leave blank to auto-calculate.">
                    </div>
                    <button type="submit">Predict segment</button>
                </form>

                <div class="result-box" id="resultBox">
                    <h3>Prediction</h3>
                    <p class="muted">Submit a transaction to see the predicted segment, confidence, and profile summary.</p>
                </div>
            </section>

            <aside class="panel">
                <p class="eyebrow">Quick Facts</p>
                <div class="chips" id="segmentPills"></div>
                <div class="divider"></div>
                <div class="list-box">
                    <h3>API endpoints</h3>
                    <p class="muted small">GET /health, GET /metadata, POST /predict, POST /predict/batch</p>
                    <p class="muted small">The page uses the live API so the same model powers both the UI and JSON endpoints.</p>
                </div>
            </aside>
        </section>
    </div>

    <script>
        const countrySelect = document.getElementById('country');
        const segmentPills = document.getElementById('segmentPills');
        const healthStatus = document.getElementById('healthStatus');
        const featureCount = document.getElementById('featureCount');
        const segmentCount = document.getElementById('segmentCount');
        const countryCount = document.getElementById('countryCount');
        const trainingRecords = document.getElementById('trainingRecords');
        const resultBox = document.getElementById('resultBox');

        function currency(value) {
            return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
        }

        function profileMarkup(result) {
            const confidence = result.confidence == null ? 'Unknown' : `${Math.round(result.confidence * 100)}%`;
            return `
                <p class="eyebrow">Predicted segment</p>
                <h3>${result.label}</h3>
                <div class="badge-row">
                    <span class="badge">Segment ${result.segment}</span>
                    <span class="badge">Confidence ${confidence}</span>
                </div>
                <p class="muted" style="margin-top:12px;line-height:1.6;">${result.profile.description}</p>
                <div class="divider"></div>
                <p class="small muted"><strong>Inputs:</strong> Quantity ${currency(result.features.Quantity)}, Unit price ${currency(result.features.UnitPrice)}, Total amount ${currency(result.features.TotalAmount)}, Country encoded ${result.features.Country_Encoded}</p>
            `;
        }

        async function loadMetadata() {
            const [healthRes, metadataRes] = await Promise.all([fetch('/health'), fetch('/metadata')]);
            const health = await healthRes.json();
            const metadata = await metadataRes.json();

            healthStatus.innerHTML = `<strong style="color:var(--accent-2)">Status:</strong> ${health.status.toUpperCase()} | Model ${health.model} | Scaler ${health.scaler}`;
            featureCount.textContent = metadata.feature_columns.length;
            segmentCount.textContent = metadata.segments.length;
            countryCount.textContent = Object.keys(metadata.countries).length;
            trainingRecords.textContent = metadata.training_records ? `${metadata.training_records.toLocaleString()} training rows indexed.` : 'Training row count not available in the artifact bundle.';

            countrySelect.innerHTML = Object.keys(metadata.countries)
                .sort()
                .map(country => `<option value="${country}">${country}</option>`)
                .join('');
            countrySelect.value = 'United Kingdom';

            segmentPills.innerHTML = metadata.segments.map(segment => `
                <span class="chip">${segment.label}</span>
            `).join('');
        }

        document.getElementById('predictForm').addEventListener('submit', async (event) => {
            event.preventDefault();
            const payload = {
                quantity: Number(document.getElementById('quantity').value),
                unit_price: Number(document.getElementById('unitPrice').value),
                country: countrySelect.value,
            };

            const totalAmount = document.getElementById('totalAmount').value;
            const countryEncoded = document.getElementById('countryEncoded').value;
            if (totalAmount !== '') payload.total_amount = Number(totalAmount);
            if (countryEncoded !== '') payload.country_encoded = Number(countryEncoded);

            resultBox.innerHTML = '<p class="muted">Predicting segment...</p>';
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const error = await response.json();
                resultBox.innerHTML = `<p class="muted">${error.detail || 'Prediction failed.'}</p>`;
                return;
            }

            const result = await response.json();
            resultBox.innerHTML = profileMarkup(result);
        });

        loadMetadata().catch((error) => {
            healthStatus.textContent = `Unable to load service data: ${error.message}`;
            healthStatus.style.color = '#b33a3a';
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Status"])
def root() -> HTMLResponse:
        return HTMLResponse(content=build_dashboard_html())


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
