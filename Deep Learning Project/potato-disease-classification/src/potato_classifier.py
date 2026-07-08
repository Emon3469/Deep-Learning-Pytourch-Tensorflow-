from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "1.keras"
H5_MODEL_PATH = PROJECT_ROOT / "potatoes.h5"
DEFAULT_TFLITE_PATH = PROJECT_ROOT / "tf-lite-models" / "2.tflite"
IMAGE_SIZE = (256, 256)
CLASS_NAMES = ("Early Blight", "Late Blight", "Healthy")


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_path: str


def read_image(file: bytes | BinaryIO) -> Image.Image:
    if isinstance(file, bytes):
        data = BytesIO(file)
    else:
        data = file

    return Image.open(data).convert("RGB")


def preprocess_image(image: Image.Image, normalize: bool = False) -> np.ndarray:
    array = np.array(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
    if normalize:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def _normalise_probabilities(raw_predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(raw_predictions, dtype=np.float32).reshape(-1)
    if predictions.size != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs, got {predictions.size}."
        )

    total = float(np.sum(predictions))
    if total > 0 and not np.isclose(total, 1.0, atol=1e-3):
        predictions = predictions / total
    return predictions


def _build_result(predictions: np.ndarray, model_path: Path) -> PredictionResult:
    probabilities = _normalise_probabilities(predictions)
    predicted_index = int(np.argmax(probabilities))
    scores = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }

    return PredictionResult(
        predicted_class=CLASS_NAMES[predicted_index],
        confidence=float(probabilities[predicted_index]),
        probabilities=scores,
        model_path=str(model_path),
    )


def load_keras_model(model_path: Path = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for local Keras predictions. "
            "Install the API requirements in a Python version supported by TensorFlow."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return tf.keras.models.load_model(model_path)


def predict_with_keras(
    image: Image.Image,
    model=None,
    model_path: Path = DEFAULT_MODEL_PATH,
    normalize: bool = False,
) -> PredictionResult:
    model = model or load_keras_model(model_path)
    batch = preprocess_image(image, normalize=normalize)
    predictions = model.predict(batch, verbose=0)[0]
    return _build_result(predictions, model_path)


def predict_with_tflite(
    image: Image.Image,
    model_path: Path = DEFAULT_TFLITE_PATH,
    normalize: bool = False,
) -> PredictionResult:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for local TFLite predictions. "
            "Install the API requirements in a Python version supported by TensorFlow."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model file not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    batch = preprocess_image(image, normalize=normalize).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], batch)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details["index"])[0]
    return _build_result(predictions, model_path)
