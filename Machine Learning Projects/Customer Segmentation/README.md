# Customer Segmentation Studio

FastAPI segmentation service with a lightweight HTML dashboard at the root route and an optional Streamlit dashboard. The API predicts customer segments from transaction features, and the dashboards can call the API or run locally against the bundled artifacts.

## Local Run

API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Dashboard:

```bash
pip install -r requirements.txt
streamlit run streamlit.py
```

Open `http://localhost:8000/` to use the built-in HTML dashboard, or `http://localhost:8501/` for the Streamlit dashboard.

## Render Deployment

Deploy the API and the dashboard as separate Web Services if you want the Streamlit UI. The API root already serves a lightweight HTML dashboard.

### API Service

1. Create a new Web Service in Render.
2. Set the root directory to this folder.
3. Use `pip install -r requirements.txt` as the build command.
4. Use `uvicorn main:app --host 0.0.0.0 --port $PORT` as the start command.
5. Deploy and confirm `/` shows the HTML dashboard and `/health` returns `{"status": "ok"}`.

### Dashboard Service

1. Create a second Web Service in Render.
2. Use the same root directory and build command.
3. Use `streamlit run streamlit.py --server.address 0.0.0.0 --server.port $PORT` as the start command.
4. Set `API_URL` to the public URL of the API service.
5. Deploy and confirm the dashboard can fetch metadata and predictions.

## Required Artifacts

Keep `customer_segmentation_model.pkl`, `scaler.pkl`, and `segmented_retail_data.pkl` in the deployment root so the service can warm up successfully.
