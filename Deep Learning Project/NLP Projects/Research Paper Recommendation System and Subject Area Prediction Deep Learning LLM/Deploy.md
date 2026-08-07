# Research Paper Recommendation Deploy Guide

This guide deploys the Flask research recommender to Render.

## What To Deploy

- Web app: `main.py`
- Required artifacts: `models/rec_model.pkl`, `models/embeddings.pkl`, `models/sentences.pkl`, `models/vocab.pkl`, `dataset/arxiv_data_210930-054931.csv`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/NLP Projects/Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `gunicorn main:app`.
6. Keep the `models/` and `dataset/` folders beside `main.py`.
7. Deploy and open the service URL.

## Verification

1. Open `/api/health`.
2. Submit a paper title and abstract to `/api/recommend`.
3. Confirm recommendations and subject areas render correctly.
