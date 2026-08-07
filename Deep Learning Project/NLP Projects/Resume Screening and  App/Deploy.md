# Resume Screening Classifier Deploy Guide

This guide deploys the Flask resume classifier to Render.

## What To Deploy

- Web app: `app.py`
- Required artifacts: `clf.pkl`, `tfidf.pkl`, `encoder.pkl`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/NLP Projects/Resume Screening and  App`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `gunicorn app:app`.
6. Add any missing model files to the deployment root before the first deploy.
7. Deploy and open the service URL.

## Verification

1. Open `/api/health` and confirm the service responds.
2. Submit resume text to `/api/predict`.
3. Upload a PDF, DOCX, or TXT file and confirm the prediction returns.
4. Check logs for missing pickle files if the app fails to start.
