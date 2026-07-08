# Potato Disease Classification

This project contains local model artifacts, FastAPI prediction services, GCP helper code, and a Streamlit frontend for potato leaf disease classification.

## Run the Streamlit App

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The Streamlit app supports three prediction sources:

- Local Keras model: `saved_models/1.keras`
- Local TFLite model: `tf-lite-models/2.tflite`
- FastAPI endpoint: defaults to `http://localhost:8000/predict`

## Run the Local API

```powershell
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Then open the Streamlit app and choose `FastAPI endpoint` if you want the UI to call the API instead of loading the model directly.

## Shared Prediction Logic

The local API and Streamlit app both use `src/potato_classifier.py` for image loading, resizing, class names, model loading, and confidence formatting. This keeps the UI and API predictions aligned.
