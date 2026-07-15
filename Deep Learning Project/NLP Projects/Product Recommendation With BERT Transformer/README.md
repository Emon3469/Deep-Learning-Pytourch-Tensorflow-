# Mobile Product Recommendation System

Flask web app that recommends mobile phones from a product catalog using precomputed transformer embeddings stored in `product_embeddings.pkl`.

## What This Does

The app accepts a natural-language product search such as "5G AMOLED phone with 128GB storage" and returns the closest matching phones from the catalog. When `sentence-transformers` is available, it uses semantic similarity against the stored embeddings. If the transformer runtime is unavailable, it falls back to TF-IDF search over product names and specifications so the demo remains usable.

## Tech Stack

- Backend: Flask
- Machine learning: sentence-transformers, scikit-learn fallback search
- Data: pandas, NumPy
- Frontend: HTML, CSS, JavaScript

## Model Artifacts

| File | Purpose |
| --- | --- |
| `product_embeddings.pkl` | Product DataFrame with text and vector embeddings |
| `mobile_recommendation_system_dataset.csv` | Source catalog |
| `Product_Recommendation_System_with_Transformers.ipynb` | Training and experimentation notebook |

## Getting Started

Python 3.10 to 3.12 is recommended for transformer deployments.

```bash
cd "Product Recommendation With BERT Transformer"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open `http://localhost:5001`.

For Linux or macOS activation:

```bash
source .venv/bin/activate
```

## API

Health check:

```bash
GET /api/health
```

Recommend products:

```bash
POST /api/recommend
Content-Type: application/json

{
  "query": "5G AMOLED phone 128GB fast processor",
  "top_k": 8
}
```

Legacy compatibility endpoint:

```bash
GET or POST /recommendations
```

## How It Works

1. The backend loads `product_embeddings.pkl` and normalizes all stored vectors.
2. A query is encoded with `all-MiniLM-L6-v2` when `sentence-transformers` is installed.
3. Cosine similarity ranks the catalog by semantic closeness.
4. If the transformer runtime is missing, TF-IDF search ranks products from catalog text.
5. The frontend renders product cards with images, prices, ratings, specs, and match scores.

## Deployment Notes

- Keep `product_embeddings.pkl` beside `main.py`.
- The app can run without GPU acceleration.
- If deploying to Render, Railway, or a similar Linux host, use `gunicorn main:app`.
- Large model/data files may need Git LFS or object storage depending on your host limits.

## Future Ideas

- Add filters for price, brand, RAM, storage, and rating.
- Add user feedback to improve ranking over time.
- Cache the sentence-transformer model during container build for faster cold starts.
