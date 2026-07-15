# Healthcare Customer Support Chatbot

Flask chat application for healthcare support questions using a fine-tuned T5 checkpoint with a retrieval fallback built from the provided domain support dataset.

## What This Does

The app provides a chat-style frontend where users can ask healthcare customer-support questions about appointments, medication reminders, vaccine side effects, and related support workflows. The backend first attempts to answer with the local T5 model in `chatbot_model`. If the transformer runtime is unavailable, it retrieves the closest matching support answer from `domain_specific_chatbot_data.csv`.

This is a portfolio support assistant, not a clinical diagnosis tool.

## Tech Stack

- Backend: Flask
- Transformer model: Hugging Face Transformers, PyTorch, safetensors
- Fallback search: scikit-learn TF-IDF
- Data: pandas, NumPy
- Frontend: HTML, CSS, JavaScript

## Model And Data Artifacts

| Path | Purpose |
| --- | --- |
| `chatbot_model/` | Primary fine-tuned T5 model and tokenizer |
| `trained_chatbot/` | Additional exported checkpoint folder |
| `domain_specific_chatbot_data.csv` | Support question/response dataset |
| `HealthCare_Chatbot_for_Domain_Specific_Customer_Support.ipynb` | Training and experimentation notebook |

## Getting Started

Python 3.10 to 3.12 is recommended for transformer deployments.

```bash
cd "Healthcare Customer Support Chatbot Using Transformer"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open `http://localhost:5002`.

For Linux or macOS activation:

```bash
source .venv/bin/activate
```

## API

Health check:

```bash
GET /api/health
```

Chat:

```bash
POST /api/chat
Content-Type: application/json

{
  "message": "How can I schedule an appointment with my doctor?"
}
```

Legacy compatibility endpoint:

```bash
POST /chat
```

## How It Works

1. The backend cleans incoming chat text.
2. It tries to load the local T5 model and tokenizer from `chatbot_model`.
3. If the model loads, it generates a response with beam search.
4. If the model cannot load, it searches the healthcare support rows in the CSV with TF-IDF.
5. The frontend displays the answer and the active engine.

## Deployment Notes

- Keep `chatbot_model/` and `domain_specific_chatbot_data.csv` with the app.
- Use `gunicorn main:app` on Linux hosts.
- Set `DISABLE_TRANSFORMER=1` if you intentionally want to run only the lightweight retrieval fallback.
- Use Git LFS or external storage if your host rejects large `.safetensors` files.

## Future Ideas

- Add conversation memory for multi-turn support flows.
- Add intent analytics for common support questions.
- Add a safety layer that redirects urgent symptoms to professional care.
