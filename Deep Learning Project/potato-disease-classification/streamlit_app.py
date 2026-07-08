from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st

from src.potato_classifier import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    predict_with_keras,
    read_image,
)


st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon=":seedling:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_local_model(model_path: str):
    from src.potato_classifier import load_keras_model

    return load_keras_model(model_path)


def result_badge(label: str) -> str:
    palette = {
        "Healthy": ("#1f8a52", "#e4f7ed"),
        "Early Blight": ("#9a5a00", "#fff1d8"),
        "Late Blight": ("#a12f2f", "#ffe4e2"),
    }
    color, background = palette.get(label, ("#334155", "#e2e8f0"))
    return (
        f"<span style='color:{color};background:{background};padding:0.35rem "
        "0.65rem;border-radius:999px;font-weight:700;'>"
        f"{label}</span>"
    )


st.markdown(
    """
    <style>
        .stApp {
            background: #f6f2e8;
            color: #1f2933;
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #ded6c7;
            border-radius: 8px;
            padding: 1rem;
        }
        section[data-testid="stSidebar"] {
            background: #20352a;
        }
        section[data-testid="stSidebar"] * {
            color: #f7f0dd;
        }
        .upload-panel {
            border: 1px solid #ded6c7;
            border-radius: 8px;
            padding: 1rem;
            background: #fffaf0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Potato Leaf Disease Classifier")
st.caption("Upload a potato leaf image and run prediction directly from the trained local model.")

H5_MODEL_PATH = Path(__file__).resolve().parent / "potatoes.h5"

with st.sidebar:
    st.header("Trained Model")
    model_options = {
        "potatoes.h5": H5_MODEL_PATH,
        "saved_models/1.keras": DEFAULT_MODEL_PATH,
    }
    selected_model_name = st.radio(
        "Model file",
        tuple(model_options.keys()),
    )
    selected_model_path = model_options[selected_model_name]
    normalize = st.checkbox(
        "Scale pixels to 0-1",
        value=False,
        help="Enable this only if the selected model was trained with normalized image tensors.",
    )
    st.divider()
    st.write("Classes")
    for class_name in CLASS_NAMES:
        st.write(f"- {class_name}")

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.subheader("Image")
    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG, or PNG",
        type=("jpg", "jpeg", "png"),
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = read_image(image_bytes)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        image_bytes = None
        image = None
        st.info("Choose a clear leaf image to begin.")

with right:
    st.subheader("Prediction")

    if image is None or image_bytes is None:
        st.metric("Result", "Waiting")
        st.progress(0)
    else:
        try:
            with st.spinner("Running model..."):
                model = load_local_model(str(selected_model_path))
                result = predict_with_keras(
                    image,
                    model=model,
                    model_path=selected_model_path,
                    normalize=normalize,
                )

            st.markdown(result_badge(result.predicted_class), unsafe_allow_html=True)
            st.metric("Confidence", f"{result.confidence * 100:.2f}%")
            st.progress(min(max(result.confidence, 0.0), 1.0))
            st.caption(f"Source: {result.model_path}")

            st.write("Class scores")
            for class_name, probability in result.probabilities.items():
                st.write(f"{class_name}: {probability * 100:.2f}%")
                st.progress(min(max(probability, 0.0), 1.0))

            with st.expander("Image details"):
                st.write(
                    {
                        "filename": uploaded_file.name,
                        "size": image.size,
                        "mode": image.mode,
                        "bytes": len(BytesIO(image_bytes).getvalue()),
                    }
                )
        except Exception as exc:
            st.error(str(exc))
            st.info(
                "For local predictions, use a Python environment with a working TensorFlow "
                "install and the model files `potatoes.h5` or `saved_models/1.keras`."
            )
