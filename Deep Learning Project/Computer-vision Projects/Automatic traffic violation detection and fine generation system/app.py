from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from database.db import init_db, load_fines, save_fine
from services.detector import MissingInferenceDependency, TrafficDetector
from services.pydentic import Detection, FineRecord


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "Uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
INDEX_HTML_PATH = TEMPLATE_DIR / "index.html"

for directory in (TEMPLATE_DIR, STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(
        title="Traffic Violation Detection API",
        description="FastAPI app for running the trained traffic violation model and generating fines.",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

detector = TrafficDetector()

VIOLATION_RULES: list[tuple[str, str, int]] = [
        ("helmet", "no_helmet", 1000),
        ("signal", "signal_violation", 5000),
        ("speed", "over_speeding", 3000),
        ("parking", "illegal_parking", 1000),
        ("lane", "lane_violation", 1500),
        ("wrong", "wrong_side_driving", 2000),
        ("triple", "triple_riding", 1500),
        ("seatbelt", "seatbelt_violation", 1000),
        ("mobile", "mobile_phone_violation", 1000),
        ("plate", "number_plate_violation", 2000),
        ("smoke", "pollution_violation", 1000),
]


def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def sanitize_filename(filename: str) -> str:
        suffix = Path(filename).suffix.lower() or ".bin"
        stem = Path(filename).stem or "upload"
        cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
        return f"{cleaned}_{uuid.uuid4().hex[:8]}{suffix}"


def upload_file_path(filename: str) -> Path:
        return UPLOAD_DIR / sanitize_filename(filename)


def output_file_path(source_name: str, suffix: str) -> Path:
        return OUTPUT_DIR / f"{Path(source_name).stem}_{suffix}"


def infer_violation(detection: Detection) -> tuple[str, int] | None:
        label = detection.label.lower()
        for keyword, violation_type, fine_amount in VIOLATION_RULES:
                if keyword in label:
                        return violation_type, fine_amount
        return None


def read_plate_text(frame: object, detection: Detection) -> str:
    label = detection.label.lower()
    if "plate" not in label and "license" not in label and "number" not in label:
        return "UNKNOWN"

    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError:
        return "UNKNOWN"

    cv2 = detector._require_cv2()
    x1, y1, x2, y2 = detection.xyxy
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN"

    if crop.shape[0] < 40 or crop.shape[1] < 120:
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
    try:
        results = ocr.ocr(crop, cls=True)
    except AttributeError:
        results = ocr.predict(crop)

    texts: list[str] = []
    if results:
        for line in results:
            if not line:
                continue
            if isinstance(line, list) and len(line) >= 2:
                candidate = line[1]
                if isinstance(candidate, (list, tuple)) and candidate:
                    texts.append(str(candidate[0]))
                    continue
                texts.append(str(candidate))

    plate_text = " ".join(part.strip() for part in texts if str(part).strip())
    return plate_text or "UNKNOWN"


def build_fine_record(
        detection: Detection,
        frame_index: int,
        evidence_path: Path,
        violation_type: str,
        fine_amount: int,
    plate_number: str,
) -> FineRecord:
        return FineRecord(
                timestamp=now_iso(),
                frame_index=frame_index,
                track_id=detection.track_id,
        plate_number=plate_number,
                violation_type=violation_type,
                fine_amount=fine_amount,
                evidence_path=str(evidence_path),
                bbox=detection.xyxy,
        )


def serialise_detections(detections: list[Detection]) -> list[dict[str, object]]:
        return [detection.to_dict() for detection in detections]


def process_image(image_path: Path, conf: float = 0.4) -> dict[str, Any]:
    cv2 = detector._require_cv2()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise HTTPException(status_code=400, detail=f"Could not read image: {image_path.name}")

    detections = detector.detect_frame(frame, conf=conf)
    annotated = detector.annotate_frame(frame, detections)
    annotated_path = output_file_path(image_path.name, "annotated.jpg")
    cv2.imwrite(str(annotated_path), annotated)

    saved_fines: list[dict[str, object]] = []
    seen_signatures: set[str] = set()
    for detection in detections:
        violation = infer_violation(detection)
        if violation is None:
            continue
        violation_type, fine_amount = violation
        signature = f"{violation_type}:{detection.track_id}:{detection.label}"
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        plate_number = read_plate_text(frame, detection)
        record = build_fine_record(detection, 0, annotated_path, violation_type, fine_amount, plate_number)
        record_id = save_fine(record)
        saved_fines.append({**record.to_dict(), "id": record_id})

    return {
        "input_url": f"/uploads/{image_path.name}",
        "annotated_image_url": f"/outputs/{annotated_path.name}",
        "detections": serialise_detections(detections),
        "fines": saved_fines,
        "summary": {
            "detections": len(detections),
            "fines": len(saved_fines),
            "source": "image",
        },
    }


def process_video(video_path: Path, conf: float = 0.4, frame_stride: int = 4) -> dict[str, Any]:
    cv2 = detector._require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail=f"Could not read video: {video_path.name}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise HTTPException(status_code=400, detail="Video dimensions could not be determined.")

    annotated_path = output_file_path(video_path.name, "annotated.mp4")
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_stride = max(1, frame_stride)
    frame_index = 0
    saved_fines: list[dict[str, object]] = []
    seen_signatures: set[str] = set()
    frames: list[dict[str, object]] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                detections = detector.detect_frame(frame, conf=conf)
                annotated = detector.annotate_frame(frame, detections)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "detections": serialise_detections(detections),
                    }
                )
                for detection in detections:
                    violation = infer_violation(detection)
                    if violation is None:
                        continue
                    violation_type, fine_amount = violation
                    signature = f"{violation_type}:{detection.track_id}:{detection.label}"
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    plate_number = read_plate_text(frame, detection)
                    record = build_fine_record(detection, frame_index, annotated_path, violation_type, fine_amount, plate_number)
                    record_id = save_fine(record)
                    saved_fines.append({**record.to_dict(), "id": record_id})
            else:
                annotated = frame

            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "input_url": f"/uploads/{video_path.name}",
        "output_video_url": f"/outputs/{annotated_path.name}",
        "frames": frames,
        "fines": saved_fines,
        "summary": {
            "frames_processed": frame_index,
            "frames_sampled": len(frames),
            "fines": len(saved_fines),
            "source": "video",
        },
    }


def render_index_page() -> str:
        return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.on_event("startup")
def startup() -> None:
        init_db()


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(render_index_page())


@app.get("/health")
def health() -> dict[str, str]:
        return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
        return detector.metadata()


@app.get("/fines")
def list_fines() -> list[dict[str, Any]]:
        return load_fines()


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = 0.4) -> dict[str, Any]:
        upload_path = upload_file_path(file.filename)
        upload_path.write_bytes(await file.read())
        return process_image(upload_path, conf=conf)


@app.post("/detect/video")
async def detect_video(
        file: UploadFile = File(...),
        conf: float = 0.4,
        frame_stride: int = 4,
) -> dict[str, Any]:
        upload_path = upload_file_path(file.filename)
        upload_path.write_bytes(await file.read())
        return process_video(upload_path, conf=conf, frame_stride=frame_stride)