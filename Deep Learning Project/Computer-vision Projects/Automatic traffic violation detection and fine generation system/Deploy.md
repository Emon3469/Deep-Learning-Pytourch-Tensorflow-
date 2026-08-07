# Automatic Traffic Violation Detection Deploy Guide

This guide deploys the FastAPI traffic-violation app to Render.

## What To Deploy

- Web app: `app.py`
- Required folders: `templates/`, `static/`, `Uploads/`, `outputs/`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/Computer-vision Projects/Automatic traffic violation detection and fine generation system`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
6. Keep the template and static folders in the deployment root.
7. Deploy and open the service URL.

## Verification

1. Open the root path and confirm the JSON health response.
2. Upload a sample image and verify detection and fine generation.
3. Check that generated files can be written to `Uploads/` and `outputs/`.
