from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "product_embeddings.pkl"
DEFAULT_MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

app = Flask(__name__)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def load_catalog() -> tuple[pd.DataFrame, np.ndarray]:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {EMBEDDINGS_PATH.name}")

    catalog = pd.read_pickle(EMBEDDINGS_PATH).copy()
    required_columns = {"name", "ratings", "price", "imgURL", "corpus", "embeddings"}
    missing = required_columns.difference(catalog.columns)
    if missing:
        raise ValueError(f"Product artifact is missing columns: {', '.join(sorted(missing))}")

    text_column = catalog["text"] if "text" in catalog.columns else ""
    catalog["search_text"] = (
        catalog["name"].fillna("").astype(str)
        + " "
        + catalog["corpus"].fillna("").astype(str)
        + " "
        + pd.Series(text_column, index=catalog.index).fillna("").astype(str)
    ).map(_clean_text)

    embeddings = np.vstack(catalog["embeddings"].map(lambda value: np.asarray(value, dtype=np.float32)))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-12)
    return catalog, normalized_embeddings


@lru_cache(maxsize=1)
def load_sentence_model() -> tuple[Any | None, str | None]:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(DEFAULT_MODEL_NAME), None
    except Exception as exc:  # pragma: no cover - depends on local ML runtime
        return None, str(exc)


@lru_cache(maxsize=1)
def load_lexical_index() -> tuple[TfidfVectorizer, Any]:
    catalog, _ = load_catalog()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)
    matrix = vectorizer.fit_transform(catalog["search_text"])
    return vectorizer, matrix


def _semantic_scores(query: str, embeddings: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    model, error = load_sentence_model()
    if model is None:
        return None, error

    query_embedding = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
    return embeddings @ query_embedding, None


def _lexical_scores(query: str) -> np.ndarray:
    vectorizer, matrix = load_lexical_index()
    query_vector = vectorizer.transform([query])
    scores = matrix @ query_vector.T
    return np.asarray(scores.toarray()).reshape(-1)


def _serialize_product(row: pd.Series, score: float, engine: str) -> dict[str, Any]:
    return {
        "name": _clean_text(row.get("name")),
        "price": _clean_text(row.get("price")),
        "ratings": _to_float(row.get("ratings")),
        "image_url": _clean_text(row.get("imgURL")),
        "specs": _clean_text(row.get("corpus"))[:360],
        "similarity": round(float(score), 4),
        "match_percent": round(max(float(score), 0.0) * 100, 2),
        "engine": engine,
    }


def recommend_products(query: str, top_k: int = 8) -> dict[str, Any]:
    query = _clean_text(query)
    if not query:
        raise ValueError("Search query is required.")

    top_k = min(max(int(top_k), 1), 20)
    catalog, embeddings = load_catalog()

    scores, semantic_error = _semantic_scores(query, embeddings)
    engine = "semantic-transformer"
    if scores is None:
        scores = _lexical_scores(query)
        engine = "tfidf-fallback"

    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = [
        _serialize_product(catalog.iloc[index], float(scores[index]), engine)
        for index in top_indices
    ]

    return {
        "query": query,
        "count": len(recommendations),
        "engine": engine,
        "runtime_note": semantic_error if semantic_error and engine == "tfidf-fallback" else None,
        "recommendations": recommendations,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    try:
        catalog, _ = load_catalog()
        model, error = load_sentence_model()
        return jsonify(
            {
                "status": "ok",
                "products": int(len(catalog)),
                "semantic_model": model is not None,
                "fallback_available": True,
                "runtime_note": error,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503


@app.post("/api/recommend")
def api_recommend() -> Any:
    try:
        payload = request.get_json(silent=True) or request.form
        query = payload.get("query", "")
        top_k = payload.get("top_k", 8)
        return jsonify(recommend_products(query, top_k))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/recommendations", methods=["GET", "POST"])
def legacy_recommendations() -> Any:
    try:
        query = request.values.get("query", "")
        top_k = request.values.get("top_k", 8)
        return jsonify(recommend_products(query, top_k))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
