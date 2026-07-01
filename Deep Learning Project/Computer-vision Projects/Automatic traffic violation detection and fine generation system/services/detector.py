from __future__ import annotations

from pathlib import Path

from services.pydentic import Detection


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_CANDIDATES = [MODEL_DIR / "best.pt", MODEL_DIR / "best.onnx"]
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
        self.names = CLASS_NAMES.copy()

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "ultralytics" if self.model is not None else "not_loaded",
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
        if self.model is not None:
            return
        if not self.model_path.exists():
            fallback = next((path for path in MODEL_CANDIDATES if path.exists()), None)
            if fallback is None:
                available = ", ".join(str(path) for path in MODEL_CANDIDATES)
                raise FileNotFoundError(f"Model file not found. Looked for: {available}")
            self.model_path = fallback
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise MissingInferenceDependency(
                "Install model dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO(str(self.model_path))
        self.names = self._normalize_names(getattr(self.model, "names", CLASS_NAMES))

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
