# Healthcare Customer Support Chatbot Deploy Guide

This guide deploys the Flask healthcare chatbot to Render.

## What To Deploy

- Web app: `main.py`
- Required artifacts: `chatbot_model/`, `trained_chatbot/`, `domain_specific_chatbot_data.csv`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/NLP Projects/Healthcare Customer Support Chatbot Using Transformer`.
4. Add a `runtime.txt` file with `python-3.11.13` so Render uses a stable Python version for the transformer stack.
5. Set the Build Command to `pip install -r requirements.txt`.
6. Set the Start Command to `gunicorn main:app`.
7. Keep the model folder and CSV in the deployment root.
8. Deploy and open the service URL.

## Optional Environment Variables

- `DISABLE_TRANSFORMER=1`: run the retrieval fallback only.
- `CHATBOT_MODEL_DIR`: point to a custom transformer model folder.

## Verification

1. Open `/api/health`.
2. Send a chat request to `/api/chat`.
3. Confirm the response engine matches the expected runtime mode.

## Why This Fix Works

Render can choose a newer default Python release that does not always line up with prebuilt wheels for the transformer and torch stack. Pinning Python 3.11 keeps the deployment path predictable.
