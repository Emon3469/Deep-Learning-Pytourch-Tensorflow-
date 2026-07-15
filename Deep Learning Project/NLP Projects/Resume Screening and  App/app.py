from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    import docx
except ImportError:  # pragma: no cover - handled at request time
    docx = None

try:
    import PyPDF2
except ImportError:  # pragma: no cover - handled at request time
    PyPDF2 = None


BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_SIZE = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
PUNCTUATION = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


@dataclass(frozen=True)
class ModelBundle:
    classifier: Any
    vectorizer: Any
    encoder: Any


def clean_resume(text: str) -> str:
    """Normalize resume text in the same style used during training."""
    cleaned = re.sub(r"http\S+\s", " ", text)
    cleaned = re.sub(r"\bRT\b|\bcc\b", " ", cleaned)
    cleaned = re.sub(r"#\S+\s", " ", cleaned)
    cleaned = re.sub(r"@\S+", " ", cleaned)
    cleaned = re.sub(r"[%s]" % re.escape(PUNCTUATION), " ", cleaned)
    cleaned = re.sub(r"[^\x00-\x7f]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact is missing: {path.name}")
    with path.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    return ModelBundle(
        classifier=_load_pickle(BASE_DIR / "clf.pkl"),
        vectorizer=_load_pickle(BASE_DIR / "tfidf.pkl"),
        encoder=_load_pickle(BASE_DIR / "encoder.pkl"),
    )


def extract_text_from_pdf(file_storage: Any) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is not installed. Install requirements.txt and try again.")
    reader = PyPDF2.PdfReader(file_storage.stream)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(file_storage: Any) -> str:
    if docx is None:
        raise RuntimeError("python-docx is not installed. Install requirements.txt and try again.")
    document = docx.Document(file_storage.stream)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_txt(file_storage: Any) -> str:
    raw = file_storage.read()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def extract_resume_text(file_storage: Any) -> str:
    extension = Path(file_storage.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Allowed types: {allowed}.")
    if extension == ".pdf":
        return extract_text_from_pdf(file_storage)
    if extension == ".docx":
        return extract_text_from_docx(file_storage)
    return extract_text_from_txt(file_storage)


def _label_name(encoder: Any, value: Any) -> str:
    try:
        return str(encoder.inverse_transform([value])[0])
    except Exception:
        return str(value)


def _rank_labels(bundle: ModelBundle, vectorized: Any, limit: int = 5) -> list[dict[str, Any]]:
    classifier = bundle.classifier
    if not hasattr(classifier, "decision_function"):
        return []

    scores = np.asarray(classifier.decision_function(vectorized)).reshape(-1)
    class_values = np.asarray(getattr(classifier, "classes_", np.arange(len(scores))))
    if len(scores) != len(class_values):
        return []

    order = np.argsort(scores)[::-1][:limit]
    shifted = scores[order] - np.max(scores[order])
    weights = np.exp(shifted)
    probabilities = weights / max(weights.sum(), 1e-12)

    return [
        {
            "label": _label_name(bundle.encoder, class_values[index]),
            "score": float(scores[index]),
            "confidence": round(float(probabilities[position]) * 100, 2),
        }
        for position, index in enumerate(order)
    ]


def predict_resume(text: str) -> dict[str, Any]:
    bundle = get_model_bundle()
    cleaned = clean_resume(text)
    if not cleaned:
        raise ValueError("No readable resume text was provided.")

    vectorized = bundle.vectorizer.transform([cleaned]).toarray()
    predicted_value = bundle.classifier.predict(vectorized)[0]
    ranking = _rank_labels(bundle, vectorized)

    return {
        "category": _label_name(bundle.encoder, predicted_value),
        "ranking": ranking,
        "word_count": len(cleaned.split()),
        "character_count": len(text),
        "preview": text[:900],
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    artifacts = {
        "classifier": (BASE_DIR / "clf.pkl").exists(),
        "tfidf": (BASE_DIR / "tfidf.pkl").exists(),
        "encoder": (BASE_DIR / "encoder.pkl").exists(),
    }
    try:
        bundle = get_model_bundle()
        labels = [str(label) for label in getattr(bundle.encoder, "classes_", [])]
        return jsonify({"status": "ok", "artifacts": artifacts, "label_count": len(labels), "labels": labels})
    except Exception as exc:
        return jsonify({"status": "error", "artifacts": artifacts, "error": str(exc)}), 503


@app.post("/api/predict")
def predict() -> Any:
    try:
        text = ""
        source = "text"

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            text = str(payload.get("text", ""))
        else:
            uploaded_file = request.files.get("resume")
            if uploaded_file and uploaded_file.filename:
                text = extract_resume_text(uploaded_file)
                source = uploaded_file.filename
            else:
                text = request.form.get("text", "")

        result = predict_resume(text)
        result["source"] = source
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
