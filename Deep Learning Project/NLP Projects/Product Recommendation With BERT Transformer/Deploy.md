# Mobile Product Recommendation Deploy Guide

This guide deploys the Flask product recommender to Render.

## What To Deploy

- Web app: `main.py`
- Required artifact: `product_embeddings.pkl`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/NLP Projects/Product Recommendation With BERT Transformer`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `gunicorn main:app`.
6. Keep `product_embeddings.pkl` beside `main.py`.
7. Deploy and open the service URL.

## Optional Environment Variables

- `SENTENCE_TRANSFORMER_MODEL`: override the embedding model name.

## Verification

1. Open `/api/health`.
2. Send a recommendation request to `/api/recommend`.
3. Confirm fallback search still works if `sentence-transformers` is unavailable.
