from __future__ import annotations

import ast
import os
import pickle
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "dataset" / "arxiv_data_210930-054931.csv"

app = Flask(__name__)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path.name}")
    with path.open("rb") as file:
        return pickle.load(file)


def parse_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [term for term in re.split(r"[,; ]+", str(value)) if term]


@lru_cache(maxsize=1)
def load_assets() -> tuple[pd.DataFrame, np.ndarray]:
    sentences = _load_pickle(MODELS_DIR / "sentences.pkl")
    embeddings = np.asarray(_load_pickle(MODELS_DIR / "embeddings.pkl"), dtype=np.float32)
    papers = pd.read_csv(DATA_PATH, usecols=["terms", "titles", "abstracts"]).fillna("")

    limit = min(len(sentences), len(embeddings), len(papers))
    papers = papers.iloc[:limit].copy().reset_index(drop=True)
    papers["titles"] = papers["titles"].where(papers["titles"].astype(bool), pd.Series(sentences[:limit]))
    papers["search_text"] = (papers["titles"].astype(str) + " " + papers["abstracts"].astype(str)).map(_clean_text)

    embeddings = embeddings[:limit]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-12)
    return papers, normalized_embeddings


@lru_cache(maxsize=1)
def load_recommender_model() -> tuple[Any | None, str | None]:
    try:
        model = _load_pickle(MODELS_DIR / "rec_model.pkl")
        if not hasattr(model, "encode"):
            return None, "rec_model.pkl does not expose an encode method."
        return model, None
    except Exception as exc:  # pragma: no cover - depends on sentence-transformers runtime
        return None, str(exc)


@lru_cache(maxsize=1)
def load_lexical_index() -> tuple[TfidfVectorizer, Any]:
    papers, _ = load_assets()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(papers["search_text"])
    return vectorizer, matrix


def _semantic_scores(query: str, embeddings: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    model, error = load_recommender_model()
    if model is None:
        return None, error

    query_embedding = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
    return embeddings @ query_embedding, None


def _lexical_scores(query: str) -> np.ndarray:
    vectorizer, matrix = load_lexical_index()
    query_vector = vectorizer.transform([query])
    scores = matrix @ query_vector.T
    return np.asarray(scores.toarray()).reshape(-1)


def _subject_predictions(papers: pd.DataFrame, indices: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    weighted_terms: defaultdict[str, float] = defaultdict(float)
    for index in indices[:12]:
        weight = max(float(scores[index]), 0.0) + 0.05
        for term in parse_terms(papers.iloc[index]["terms"]):
            weighted_terms[term] += weight

    total = sum(weighted_terms.values()) or 1.0
    ranked = sorted(weighted_terms.items(), key=lambda item: item[1], reverse=True)[:8]
    return [
        {"label": label, "confidence": round((score / total) * 100, 2)}
        for label, score in ranked
    ]


def _paper_result(row: pd.Series, score: float) -> dict[str, Any]:
    return {
        "title": _clean_text(row["titles"]),
        "abstract": _clean_text(row["abstracts"])[:700],
        "terms": parse_terms(row["terms"]),
        "similarity": round(float(score), 4),
        "match_percent": round(max(float(score), 0.0) * 100, 2),
    }


def recommend_papers(title: str, abstract: str = "", top_k: int = 5) -> dict[str, Any]:
    query = _clean_text(f"{title} {abstract}")
    if not query:
        raise ValueError("A paper title or abstract is required.")

    top_k = min(max(int(top_k), 1), 12)
    papers, embeddings = load_assets()

    scores, semantic_error = _semantic_scores(query, embeddings)
    engine = "sentence-transformer"
    if scores is None:
        scores = _lexical_scores(query)
        engine = "tfidf-fallback"

    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = [_paper_result(papers.iloc[index], float(scores[index])) for index in top_indices]

    return {
        "count": len(recommendations),
        "engine": engine,
        "runtime_note": semantic_error if semantic_error and engine == "tfidf-fallback" else None,
        "subjects": _subject_predictions(papers, top_indices, scores),
        "recommendations": recommendations,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    try:
        papers, embeddings = load_assets()
        model, error = load_recommender_model()
        return jsonify(
            {
                "status": "ok",
                "papers": int(len(papers)),
                "embedding_dimensions": int(embeddings.shape[1]),
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
        title = payload.get("title", "")
        abstract = payload.get("abstract", "")
        top_k = payload.get("top_k", 5)
        return jsonify(recommend_papers(title, abstract, top_k))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5003"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
