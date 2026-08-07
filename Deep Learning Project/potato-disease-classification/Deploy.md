# Potato Disease Classification Deploy Guide

This project can be deployed as one or two Render services.

## What To Deploy

- Streamlit UI: `streamlit_app.py`
- FastAPI API: `api/main.py`
- Required artifacts: `potatoes.h5` or `saved_models/1.keras`, plus `tf-lite-models/2.tflite` if you use the TFLite path

## Streamlit Service

1. Push this folder to GitHub.
2. If you want Render to manage both services from a blueprint, use the repo-root [render.yaml](../../render.yaml) file.
3. In Render, create a new Web Service from the blueprint or configure the UI service manually.
4. Keep `runtime.txt` in this folder so Render uses Python 3.11.13.
5. Keep `requirements.txt` pinned to `tensorflow==2.20.0`.
6. If you configure the UI manually, set the Build Command to `pip install -r requirements.txt`.
7. If you configure the UI manually, set the Start Command to `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`.
8. Add `API_URL` only if the Streamlit app should call the separate API service.
9. Deploy and open the service URL.

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

Render can default to a newer Python release that does not have a compatible TensorFlow wheel for this app. Pinning Python 3.11 and using `tensorflow==2.20.0` keeps the Render build on a supported Linux wheel.
