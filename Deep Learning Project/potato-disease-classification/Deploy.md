# Potato Disease Classification Deploy Guide

This project can be deployed as one or two Render services.

## What To Deploy

- Streamlit UI: `streamlit_app.py`
- FastAPI API: `api/main.py`
- Required artifacts: `potatoes.h5` or `saved_models/1.keras`, plus `tf-lite-models/2.tflite` if you use the TFLite path

## Streamlit Service

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/potato-disease-classification`.
4. Add a `runtime.txt` file with `python-3.11.13` so Render uses a TensorFlow-compatible Python version.
5. Set the Build Command to `pip install -r requirements.txt`.
6. Set the Start Command to `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`.
7. Add `API_URL` only if the Streamlit app should call the separate API service.
8. Deploy and open the service URL.

## FastAPI Service

1. Create a second Render Web Service from the same repo folder.
2. Use the same build command.
3. Set the Start Command to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
4. Keep the model files in the deployment root.
5. Deploy and open the API service URL.

## Verification

1. Open `/ping` on the API service.
2. Upload an image in the UI and confirm the predicted class is returned.
3. If the UI uses the API, confirm `API_URL` points to the public Render URL.

## Why This Fix Works

Render can default to a newer Python release that does not have a compatible TensorFlow wheel for this app. Pinning Python 3.11 keeps `tensorflow==2.21.0` installable during the Render build.
