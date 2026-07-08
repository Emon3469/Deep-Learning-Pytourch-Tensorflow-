from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
original_path = list(sys.path)
sys.path = [
    item
    for item in sys.path
    if Path(item or ".").resolve() != CURRENT_DIR
]
import streamlit as st

sys.path = original_path

import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000"
DEFAULT_COUNTRIES = {"United Kingdom": 35}


st.set_page_config(
    page_title="Customer Segmentation Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --ink: #1c2522;
            --muted: #66736c;
            --paper: #f7f4ed;
            --surface: #fffdf8;
            --surface-soft: #ede7dc;
            --sage: #6f8178;
            --sage-dark: #263d35;
            --copper: #b86b3c;
            --copper-dark: #8f4f2d;
            --line: rgba(28, 37, 34, 0.14);
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(111, 129, 120, 0.20), transparent 31%),
                radial-gradient(circle at 88% 6%, rgba(184, 107, 60, 0.10), transparent 28%),
                linear-gradient(135deg, #f8f5ee 0%, #ebe5da 52%, #f7f4ed 100%);
            color: var(--ink);
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            letter-spacing: 0;
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: #1f2f29;
        }

        section[data-testid="stSidebar"] * {
            color: #fbf7ee;
        }

        .hero {
            border: 1px solid var(--line);
            background:
                linear-gradient(115deg, rgba(28, 37, 34, 0.96), rgba(38, 61, 53, 0.92)),
                repeating-linear-gradient(45deg, rgba(255,255,255,0.045) 0 1px, transparent 1px 18px);
            padding: 34px 36px;
            border-radius: 8px;
            box-shadow: 0 26px 70px rgba(28, 37, 34, 0.18);
            margin-bottom: 20px;
        }

        .hero h1 {
            color: #fffaf0;
            font-size: clamp(2.2rem, 5vw, 4.8rem);
            line-height: 0.95;
            margin: 0 0 12px;
        }

        .hero p {
            color: rgba(255, 250, 240, 0.82);
            max-width: 760px;
            font-size: 1.05rem;
            margin: 0;
        }

        .status-strip {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 22px;
        }

        .chip {
            border: 1px solid rgba(255,255,255,0.24);
            color: #fffaf0;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.88rem;
            background: rgba(255,250,240,0.08);
        }

        div[data-testid="metric-container"] {
            border: 1px solid var(--line);
            background: rgba(255, 253, 248, 0.82);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: none;
        }

        div[data-testid="metric-container"] label,
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
            color: var(--muted);
        }

        .result-panel {
            border-left: 7px solid var(--copper);
            background: rgba(255, 253, 248, 0.86);
            padding: 24px;
            border-radius: 8px;
            border-top: 1px solid var(--line);
            border-right: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
        }

        .result-panel h2 {
            margin-top: 0;
            font-size: 2.2rem;
        }

        .segment-note {
            font-size: 1.02rem;
            color: var(--muted);
        }

        .stButton > button {
            background: var(--copper);
            color: #fffaf0;
            border: 0;
            border-radius: 8px;
            font-weight: 700;
            padding: 0.65rem 1rem;
        }

        .stButton > button:hover {
            background: var(--copper-dark);
            color: #fffaf0;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            background: rgba(255, 253, 248, 0.72);
            border-radius: 8px;
            border: 1px solid var(--line);
            padding: 8px 14px;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--sage-dark);
            border-color: var(--sage-dark);
        }

        button[data-baseweb="tab"][aria-selected="true"] p {
            color: #fffaf0;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stFileUploader"] section {
            border-color: var(--line);
        }

        a,
        .stMarkdown a {
            color: var(--copper-dark);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=15)
def get_json(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
    response.raise_for_status()
    return response.json()


def predict_batch(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"customers": customers},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.2f}"


def render_status(metadata: dict[str, Any] | None, health: dict[str, Any] | None) -> None:
    status = "Connected" if health else "API offline"
    records = metadata.get("training_records") if metadata else None
    segments = len(metadata.get("segments", [])) if metadata else 0
    countries = len(metadata.get("countries", {})) if metadata else len(DEFAULT_COUNTRIES)

    st.markdown(
        f"""
        <div class="hero">
            <h1>Customer Segmentation Studio</h1>
            <p>Classify retail transactions into operational customer segments, then turn the result into a practical marketing and account-care decision.</p>
            <div class="status-strip">
                <span class="chip">API: {status}</span>
                <span class="chip">Training records: {records or 'Unavailable'}</span>
                <span class="chip">Segments: {segments or 'Unavailable'}</span>
                <span class="chip">Countries: {countries}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(result: dict[str, Any]) -> None:
    profile = result.get("profile", {})
    confidence = result.get("confidence")

    st.markdown(
        f"""
        <div class="result-panel">
            <h2>{result.get('label', 'Segment')}</h2>
            <p class="segment-note">{profile.get('description', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Segment ID", result.get("segment", "n/a"))
    metric_cols[1].metric("Fit score", f"{confidence:.2%}" if confidence is not None else "n/a")
    metric_cols[2].metric("Avg segment order", money(profile.get("avg_total_amount")))
    metric_cols[3].metric("Records in segment", f"{profile.get('records', 0):,}")

    st.subheader("Model Features Used")
    st.dataframe(
        pd.DataFrame([result.get("features", {})]),
        use_container_width=True,
        hide_index=True,
    )


def build_payload(countries: dict[str, int]) -> dict[str, Any]:
    with st.sidebar:
        st.header("API")
        st.caption("FastAPI should be running on port 8000.")
        st.code("uvicorn main:app --reload", language="bash")

    st.subheader("Transaction Composer")
    col_a, col_b, col_c = st.columns([1, 1, 1.2])

    with col_a:
        quantity = st.number_input("Quantity", min_value=1.0, value=6.0, step=1.0)
        unit_price = st.number_input("Unit price", min_value=0.0, value=2.55, step=0.05, format="%.2f")

    with col_b:
        country_names = list(countries.keys()) or list(DEFAULT_COUNTRIES.keys())
        default_country_index = country_names.index("United Kingdom") if "United Kingdom" in country_names else 0
        country = st.selectbox("Country", country_names, index=default_country_index)
        override_total = st.toggle("Override total amount", value=False)

    with col_c:
        computed_total = quantity * unit_price
        total_amount = st.number_input(
            "Total amount",
            min_value=0.0,
            value=float(computed_total),
            step=1.0,
            format="%.2f",
            disabled=not override_total,
        )
        st.metric("Computed basket value", money(computed_total))

    payload = {
        "quantity": quantity,
        "unit_price": unit_price,
        "country": country,
    }
    if override_total:
        payload["total_amount"] = total_amount

    return payload


def render_single_prediction(metadata: dict[str, Any] | None) -> None:
    countries = (metadata or {}).get("countries", DEFAULT_COUNTRIES)
    payload = build_payload(countries)

    run_prediction = st.button("Predict Segment", type="primary", use_container_width=True)
    if run_prediction:
        try:
            render_prediction_result(predict(payload))
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")


def render_segments(metadata: dict[str, Any] | None) -> None:
    st.subheader("Segment Playbook")
    segments = (metadata or {}).get("segments", [])
    if not segments:
        st.info("Start the FastAPI server to load segment profiles.")
        return

    for segment in segments:
        with st.container(border=True):
            top = st.columns([1.5, 1, 1, 1])
            top[0].markdown(f"### {segment['label']}")
            top[1].metric("Segment", segment["segment"])
            top[2].metric("Avg quantity", f"{segment.get('avg_quantity') or 0:,.1f}")
            top[3].metric("Avg order", money(segment.get("avg_total_amount")))
            st.write(segment["description"])


def render_batch(metadata: dict[str, Any] | None) -> None:
    st.subheader("Batch Scoring")
    st.write("Upload a CSV with `quantity`, `unit_price`, and `country`. `total_amount` is optional.")
    sample = pd.DataFrame(
        [
            {"quantity": 6, "unit_price": 2.55, "country": "United Kingdom"},
            {"quantity": 24, "unit_price": 4.25, "country": "Germany"},
            {"quantity": 1, "unit_price": 950.0, "country": "Singapore"},
        ]
    )
    st.download_button(
        "Download Sample CSV",
        data=sample.to_csv(index=False),
        file_name="customer_segmentation_sample.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if not uploaded:
        return

    frame = pd.read_csv(uploaded)
    st.dataframe(frame.head(20), use_container_width=True)

    required_columns = {"quantity", "unit_price", "country"}
    missing = required_columns - set(frame.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return

    if st.button("Score Uploaded Rows", use_container_width=True):
        records = frame.to_dict(orient="records")
        try:
            predictions = predict_batch(records)
        except requests.RequestException as exc:
            st.error(f"Batch scoring failed: {exc}")
            return

        scored = frame.copy()
        scored["segment"] = [item["segment"] for item in predictions]
        scored["label"] = [item["label"] for item in predictions]
        scored["fit_score"] = [item.get("confidence") for item in predictions]
        st.dataframe(scored, use_container_width=True)

        buffer = io.StringIO()
        scored.to_csv(buffer, index=False)
        st.download_button(
            "Download Scored CSV",
            data=buffer.getvalue(),
            file_name="customer_segments_scored.csv",
            mime="text/csv",
        )


def main() -> None:
    inject_theme()
    health = get_json("/health")
    metadata = get_json("/metadata") if health else None

    render_status(metadata, health)

    tab_predict, tab_segments, tab_batch = st.tabs(
        ["Predict", "Segment Intelligence", "Batch Scoring"]
    )
    with tab_predict:
        render_single_prediction(metadata)
    with tab_segments:
        render_segments(metadata)
    with tab_batch:
        render_batch(metadata)

    if not health:
        st.warning("Start the FastAPI server first: `uvicorn main:app --reload`")


if __name__ == "__main__":
    main()
