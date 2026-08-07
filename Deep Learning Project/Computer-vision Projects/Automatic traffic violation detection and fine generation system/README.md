# Automatic Traffic Violation Detection and Fine Generation System

FastAPI application for traffic-violation detection, license plate handling, and fine generation from uploaded media.

## Local Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Render Deployment

1. Create a new Web Service in Render.
2. Set the root directory to this folder.
3. Use `pip install -r requirements.txt` as the build command.
4. Use `uvicorn app:app --host 0.0.0.0 --port $PORT` as the start command.
5. Deploy and confirm the API root responds, then test file upload and fine generation.

## Required Assets

Make sure the template, static, upload, output, and model-related folders remain in the deployment root so the app can render its UI and save generated artifacts.
