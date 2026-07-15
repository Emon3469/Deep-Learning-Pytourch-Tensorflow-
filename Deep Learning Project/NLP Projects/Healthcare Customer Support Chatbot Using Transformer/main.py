from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("CHATBOT_MODEL_DIR", BASE_DIR / "chatbot_model"))
DATA_PATH = BASE_DIR / "domain_specific_chatbot_data.csv"

app = Flask(__name__)


@dataclass(frozen=True)
class TransformerBundle:
    tokenizer: Any
    model: Any
    torch: Any
    device: str


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def load_transformer() -> tuple[TransformerBundle | None, str | None]:
    if os.environ.get("DISABLE_TRANSFORMER", "").lower() in {"1", "true", "yes"}:
        return None, "Transformer loading disabled by DISABLE_TRANSFORMER."

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(MODEL_DIR / "tokenizer.json"),
            unk_token="<unk>",
            pad_token="<pad>",
            eos_token="</s>",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return TransformerBundle(tokenizer=tokenizer, model=model, torch=torch, device=device), None
    except Exception as exc:  # pragma: no cover - depends on local deep-learning runtime
        return None, f"Transformer runtime unavailable: {exc}"


@lru_cache(maxsize=1)
def load_retrieval_index() -> tuple[pd.DataFrame, TfidfVectorizer, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing chatbot dataset: {DATA_PATH.name}")

    data = pd.read_csv(DATA_PATH).fillna("")
    if {"query", "response"}.difference(data.columns):
        raise ValueError("Chatbot dataset must contain query and response columns.")

    if "domain" in data.columns:
        healthcare = data[data["domain"].str.lower().eq("healthcare")]
        if not healthcare.empty:
            data = healthcare

    data = data.reset_index(drop=True)
    lookup_text = (data["query"].astype(str) + " " + data.get("intent", "").astype(str)).map(clean_text)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform(lookup_text)
    return data, vectorizer, matrix


def transformer_reply(message: str) -> tuple[str | None, str | None]:
    bundle, error = load_transformer()
    if bundle is None:
        return None, error

    inputs = bundle.tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False,
    )
    inputs = {
        key: value.to(bundle.device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask"}
    }

    with bundle.torch.inference_mode():
        output = bundle.model.generate(
            **inputs,
            max_new_tokens=96,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    response = bundle.tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_text(response), None


def retrieval_reply(message: str) -> dict[str, Any]:
    data, vectorizer, matrix = load_retrieval_index()
    query_vector = vectorizer.transform([message])
    sparse_scores = matrix @ query_vector.T
    scores = np.asarray(sparse_scores.toarray()).reshape(-1)
    index = int(np.argmax(scores))
    row = data.iloc[index]
    return {
        "response": clean_text(row["response"]),
        "matched_query": clean_text(row["query"]),
        "intent": clean_text(row.get("intent", "support")),
        "similarity": round(float(scores[index]), 4),
    }


def answer_message(message: str) -> dict[str, Any]:
    message = clean_text(message)
    if not message:
        raise ValueError("Message is required.")

    generated, runtime_error = transformer_reply(message)
    if generated:
        return {"response": generated, "engine": "fine-tuned-t5", "runtime_note": None}

    fallback = retrieval_reply(message)
    fallback["engine"] = "retrieval-fallback"
    fallback["runtime_note"] = runtime_error
    return fallback


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    try:
        data, _, _ = load_retrieval_index()
        transformer, error = load_transformer()
        return jsonify(
            {
                "status": "ok",
                "transformer_model": transformer is not None,
                "retrieval_rows": int(len(data)),
                "runtime_note": error,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503


@app.post("/api/chat")
def chat() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        result = answer_message(str(payload.get("message", "")))
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/chat")
def legacy_chat() -> Any:
    return chat()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
