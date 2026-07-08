from __future__ import annotations

from pathlib import Path
from typing import Any

from services.pydentic import Detection


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_CANDIDATES = [MODEL_DIR / "best.onnx", MODEL_DIR / "best.pt"]
DEFAULT_MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])
CLASS_NAMES = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorbike",
    4: "truck",
}


class MissingInferenceDependency(RuntimeError):
    pass


class TrafficDetector:
    def __init__(
        self,
        model_path: Path | None = None,
        imgsz: int = 640,
        iou: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.imgsz = imgsz
        self.iou = iou
        self.model = None
        self.net = None
        self.backend = None
        self.names = CLASS_NAMES.copy()

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend or "not_loaded",
            "model": str(self.model_path),
            "available_models": [str(path) for path in MODEL_CANDIDATES],
            "classes": self.names,
            "imgsz": self.imgsz,
        }

    def detect_image_file(self, image_path: Path, conf: float) -> tuple[list[Detection], object]:
        cv2 = self._require_cv2()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.detect_frame(frame, conf=conf), frame

    def detect_frame(
        self,
        frame: object,
        conf: float,
        use_tracking: bool = False,
    ) -> list[Detection]:
        self._load_model()
        if self.backend == "opencv-dnn":
            return self._detect_with_onnx(frame, conf=conf)
        if use_tracking:
            try:
                results = self.model.track(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=conf,
                    iou=self.iou,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except Exception:
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=conf,
                    iou=self.iou,
                    verbose=False,
                )
        else:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=conf,
                iou=self.iou,
                verbose=False,
            )
        return self._parse_result(results[0])

    def annotate_frame(self, frame: object, detections: list[Detection]) -> object:
        cv2 = self._require_cv2()
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            label = f"{detection.label} {detection.confidence:.2f}"
            if detection.track_id is not None:
                label = f"{label} #{detection.track_id}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (41, 171, 226), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (41, 171, 226),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _load_model(self) -> None:
        if self.model is not None or self.net is not None:
            return
        if not self.model_path.exists():
            fallback = next((path for path in MODEL_CANDIDATES if path.exists()), None)
            if fallback is None:
                available = ", ".join(str(path) for path in MODEL_CANDIDATES)
                raise FileNotFoundError(f"Model file not found. Looked for: {available}")
            self.model_path = fallback
        if self.model_path.suffix.lower() == ".onnx":
            cv2 = self._require_cv2()
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.backend = "opencv-dnn"
            self.names = CLASS_NAMES.copy()
            return

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise MissingInferenceDependency(
                "Install model dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO(str(self.model_path))
        self.backend = "ultralytics"
        self.names = self._normalize_names(getattr(self.model, "names", CLASS_NAMES))

    def _detect_with_onnx(self, frame: object, conf: float) -> list[Detection]:
        cv2 = self._require_cv2()
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise MissingInferenceDependency(
                "Install NumPy with `pip install -r requirements.txt`."
            ) from exc

        image = frame.copy()
        height, width = image.shape[:2]
        resized, scale, pad_x, pad_y = self._letterbox(image, self.imgsz)
        blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        self.net.setInput(blob)
        outputs = self.net.forward()
        if outputs is None:
            return []

        predictions = np.asarray(outputs)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 64:
            predictions = predictions.transpose(1, 0)

        if predictions.shape[1] < 5:
            return []

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        candidate_boxes: list[list[int]] = []
        candidate_scores: list[float] = []
        candidate_labels: list[int] = []
        for box, score, cls_id in zip(boxes, confidences, class_ids):
            if float(score) < conf:
                continue
            x_center, y_center, box_width, box_height = map(float, box[:4])
            x1 = (x_center - box_width / 2 - pad_x) / scale
            y1 = (y_center - box_height / 2 - pad_y) / scale
            x2 = (x_center + box_width / 2 - pad_x) / scale
            y2 = (y_center + box_height / 2 - pad_y) / scale
            left = max(0, min(int(round(x1)), width - 1))
            top = max(0, min(int(round(y1)), height - 1))
            right = max(0, min(int(round(x2)), width))
            bottom = max(0, min(int(round(y2)), height))
            if right <= left or bottom <= top:
                continue
            candidate_boxes.append([left, top, right - left, bottom - top])
            candidate_scores.append(float(score))
            candidate_labels.append(int(cls_id))

        if not candidate_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_scores, score_threshold=conf, nms_threshold=self.iou)
        if len(indices) == 0:
            return []

        detections: list[Detection] = []
        for index in np.array(indices).flatten():
            x, y, box_width, box_height = candidate_boxes[int(index)]
            detections.append(
                Detection(
                    label=self.names.get(candidate_labels[int(index)], str(candidate_labels[int(index)])),
                    confidence=float(candidate_scores[int(index)]),
                    xyxy=(x, y, x + box_width, y + box_height),
                    track_id=None,
                )
            )
        return detections

    def _letterbox(self, image: object, new_size: int) -> tuple[object, float, float, float]:
        cv2 = self._require_cv2()
        height, width = image.shape[:2]
        scale = min(new_size / height, new_size / width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = self._np_full((new_size, new_size, 3), 114, image.dtype)
        pad_x = (new_size - resized_width) / 2
        pad_y = (new_size - resized_height) / 2
        left = int(round(pad_x))
        top = int(round(pad_y))
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _np_full(shape: tuple[int, int, int], fill_value: int, dtype: Any) -> object:
        import numpy as np

        return np.full(shape, fill_value, dtype=dtype)

    def _parse_result(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(xyxy)

        names = self._normalize_names(getattr(result, "names", self.names))
        detections: list[Detection] = []
        for box, score, cls_id, track_id in zip(xyxy, confs, classes, track_ids):
            detections.append(
                Detection(
                    label=names.get(int(cls_id), str(cls_id)),
                    confidence=float(score),
                    xyxy=tuple(map(int, box)),
                    track_id=None if track_id is None else int(track_id),
                )
            )
        return detections

    @staticmethod
    def _normalize_names(names: object) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return CLASS_NAMES.copy()

    @staticmethod
    def _require_cv2() -> object:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise MissingInferenceDependency(
                "Install OpenCV with `pip install opencv-python` or `pip install -r requirements.txt`."
            ) from exc
        return cv2
