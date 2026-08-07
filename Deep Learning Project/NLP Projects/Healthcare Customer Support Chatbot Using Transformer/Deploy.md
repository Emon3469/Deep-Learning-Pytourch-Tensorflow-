# Healthcare Customer Support Chatbot Deploy Guide

This guide deploys the Flask healthcare chatbot to Render.

## What To Deploy

- Web app: `main.py`
- Required artifacts: `chatbot_model/`, `trained_chatbot/`, `domain_specific_chatbot_data.csv`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Deep Learning Project/NLP Projects/Healthcare Customer Support Chatbot Using Transformer`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `gunicorn main:app`.
6. Keep the model folder and CSV in the deployment root.
7. Deploy and open the service URL.

## Optional Environment Variables

- `DISABLE_TRANSFORMER=1`: run the retrieval fallback only.
- `CHATBOT_MODEL_DIR`: point to a custom transformer model folder.

## Verification

1. Open `/api/health`.
2. Send a chat request to `/api/chat`.
3. Confirm the response engine matches the expected runtime mode.
