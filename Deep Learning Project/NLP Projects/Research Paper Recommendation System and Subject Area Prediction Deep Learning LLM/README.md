# Research Paper Recommendation And Subject Area Prediction

Flask web app that recommends similar arXiv papers and predicts likely subject areas from local embedding and dataset artifacts.

## What This Does

The app accepts a research title, abstract, or both, then returns similar papers from the arXiv dataset. It also predicts subject areas by aggregating the `terms` labels from the nearest papers. The original Streamlit app referenced a missing `model.h5`; this deployable version uses the model files that are present in the folder.

## Tech Stack

- Backend: Flask
- Recommender: sentence-transformers model loaded from `rec_model.pkl`
- Fallback search: scikit-learn TF-IDF
- Data: pandas, NumPy
- Frontend: HTML, CSS, JavaScript

## Model And Data Artifacts

| Path | Purpose |
| --- | --- |
| `models/rec_model.pkl` | Sentence transformer recommender model |
| `models/embeddings.pkl` | Precomputed paper embeddings |
| `models/sentences.pkl` | Paper titles aligned to embeddings |
| `models/vocab.pkl` | Saved subject vocabulary from experimentation |
| `dataset/arxiv_data_210930-054931.csv` | arXiv titles, abstracts, and subject terms |
| `Research_Paper_recommendation_and_subject_area_prediction.ipynb` | Training and experimentation notebook |

## Getting Started

Python 3.10 to 3.12 is recommended for transformer deployments.

```bash
cd "Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open `http://localhost:5003`.

For Linux or macOS activation:

```bash
source .venv/bin/activate
```

## API

Health check:

```bash
GET /api/health
```

Recommend papers and infer subject areas:

```bash
POST /api/recommend
Content-Type: application/json

{
  "title": "Graph neural networks for privacy preserving machine learning",
  "abstract": "We study robust graph neural networks with differential privacy guarantees.",
  "top_k": 5
}
```

## How It Works

1. The app loads paper metadata from the arXiv CSV and embeddings from `models/embeddings.pkl`.
2. It attempts to load `models/rec_model.pkl` and encode the query semantically.
3. Cosine similarity ranks the nearest papers.
4. If the sentence-transformer runtime is unavailable, TF-IDF search ranks papers from titles and abstracts.
5. Subject areas are inferred by weighting the arXiv `terms` from the top recommendations.

## Deployment Notes

- Keep the `models/` and `dataset/` folders available beside `main.py`.
- Use `gunicorn main:app` on Linux hosts.
- `embeddings.pkl` and `rec_model.pkl` are large enough that Git LFS is recommended.
- The app avoids TensorFlow at runtime because the referenced Keras model file is not included.

## Future Ideas

- Add direct paper links if arXiv IDs are added to the dataset.
- Add filters for subject area and publication year.
- Rebuild the missing subject classifier and compare it with nearest-neighbor subject inference.
