# Customer Segmentation Deploy Guide

This project is best deployed as two separate Render services.

## What To Deploy

- FastAPI API: `main.py`
- Streamlit dashboard: `streamlit.py`
- Required artifacts: `customer_segmentation_model.pkl`, `scaler.pkl`, `segmented_retail_data.pkl`

The API root (`/`) now serves a lightweight HTML dashboard, so you can deploy only the API service if you do not need Streamlit.

## FastAPI Service

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Machine Learning Projects/Customer Segmentation`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
6. Deploy the API first.
7. Confirm `/` shows the dashboard and `/health` and `/metadata` work.

## Streamlit Dashboard

1. Create a second Render Web Service from the same folder.
2. Use the same build command.
3. Set the Start Command to `streamlit run streamlit.py --server.address 0.0.0.0 --server.port $PORT`.
4. Set `API_URL` to the public API URL.
5. Deploy the dashboard and test a prediction.

## Verification

1. Check that the API can warm the pickle artifacts on startup.
2. Confirm the dashboard can load metadata and run predictions against the API.
