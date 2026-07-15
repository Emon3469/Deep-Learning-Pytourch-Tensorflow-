# NLP Projects Portfolio

Four deployable NLP and recommendation projects with Flask backends, browser frontends, local model artifacts, and project-level documentation.

## Projects

| Project | What it does | Entry point | Default port |
| --- | --- | --- | --- |
| `Resume Screening and  App` | Predicts resume job categories from text or uploaded files | `app.py` | `5000` |
| `Product Recommendation With BERT Transformer` | Recommends mobile products from semantic search | `main.py` | `5001` |
| `Healthcare Customer Support Chatbot Using Transformer` | Answers healthcare support questions with T5 or retrieval fallback | `main.py` | `5002` |
| `Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM` | Recommends similar papers and infers subject areas | `main.py` | `5003` |

## Quick Start Pattern

Each project is standalone:

```bash
cd "<project folder>"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

For projects using `main.py`, run:

```bash
python main.py
```

## Deployment Notes

- Read the README inside each project folder before deployment.
- Python 3.10 to 3.12 is recommended for the transformer-based apps.
- Large `.pkl`, `.safetensors`, and `.csv` files are required runtime artifacts. Use Git LFS or external object storage if your hosting provider enforces file-size limits.
- Each app exposes `/api/health` for deployment checks.
