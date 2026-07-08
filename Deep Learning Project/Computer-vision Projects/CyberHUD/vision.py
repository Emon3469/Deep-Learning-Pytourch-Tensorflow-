from __future__ import annotations

import threading
import time
from collections import deque
import importlib
from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional

import cv2

import hud

try:
    mp = importlib.import_module("mediapipe")
except ImportError:
    mp = None
else:
    try:
        import mediapipe.tasks as mp_tasks
        mp_tasks_vision = importlib.import_module("mediapipe.tasks.python.vision")
    except ImportError:
        mp_tasks = None
        mp_tasks_vision = None
    else:
        if not hasattr(mp_tasks_vision, "FaceLandmarker"):
            print("MediaPipe Tasks vision API unavailable; face/hand/pose overlays disabled.")
            mp_tasks = None
            mp_tasks_vision = None

if mp is None:
    mp_tasks = None
    mp_tasks_vision = None

try:
    YOLO = importlib.import_module("ultralytics").YOLO
except Exception as exc:
    print(f"YOLO unavailable: {exc}")
    YOLO = None


MODEL_DIR = Path("models")
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _ensure_model(url: str, filename: str) -> Optional[str]:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / filename
    if path.exists():
        return str(path)

    try:
        print(f"Downloading {filename}...")
        urlretrieve(url, path)
        return str(path)
    except Exception as exc:
        print(f"Could not download {filename}: {exc}")
        return None


class VisionEngine:
    def __init__(self, yolo_model_path="yolov8n.pt", conf=0.45, yolo_every=1, device=None):
        self.lock = threading.Lock()
        self.conf = conf
        self.yolo_every = max(1, int(yolo_every))
        self.device = device

        self.active_modules = {
            "YOLO": True,
            "FACE": True,
            "HAND": True,
            "POSE": True,
        }
        self.latest_jpeg: Optional[bytes] = None
        self.fps_history = deque(maxlen=30)
        self.face_status = "NOT DETECTED"
        self.current_gesture = "NONE"
        self.last_yolo_boxes = []
        self.frame_idx = 0
        self.prev_time = time.time()

        self.yolo_model = None
        if YOLO is not None:
            try:
                print("Loading YOLO model...")
                self.yolo_model = YOLO(yolo_model_path)
                if device:
                    self.yolo_model.to(device)
            except Exception as exc:
                print(f"YOLO model disabled: {exc}")
                self.yolo_model = None

        self.face_landmarker = None
        self.hands_detection = None
        self.pose_detector = None

        if mp is not None and mp_tasks is not None and mp_tasks_vision is not None:
            try:
                face_model = _ensure_model(FACE_MODEL_URL, "face_landmarker.task")
                if face_model is not None:
                    face_options = mp_tasks_vision.FaceLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=face_model),
                        running_mode=mp_tasks_vision.RunningMode.IMAGE,
                        num_faces=1,
                    )
                    self.face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(face_options)
            except Exception as exc:
                print(f"Face landmarker disabled: {exc}")
                self.face_landmarker = None

            try:
                hand_model = _ensure_model(HAND_MODEL_URL, "hand_landmarker.task")
                if hand_model is not None:
                    hand_options = mp_tasks_vision.HandLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model),
                        running_mode=mp_tasks_vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.6,
                        min_hand_presence_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                    self.hands_detection = mp_tasks_vision.HandLandmarker.create_from_options(hand_options)
            except Exception as exc:
                print(f"Hand landmarker disabled: {exc}")
                self.hands_detection = None

            try:
                pose_model = _ensure_model(POSE_MODEL_URL, "pose_landmarker_lite.task")
                if pose_model is not None:
                    pose_options = mp_tasks_vision.PoseLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=pose_model),
                        running_mode=mp_tasks_vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.6,
                        min_pose_presence_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                    self.pose_detector = mp_tasks_vision.PoseLandmarker.create_from_options(pose_options)
            except Exception as exc:
                print(f"Pose landmarker disabled: {exc}")
                self.pose_detector = None

        if self.yolo_model is None:
            self.active_modules["YOLO"] = False
        if self.face_landmarker is None:
            self.active_modules["FACE"] = False
        if self.hands_detection is None:
            self.active_modules["HAND"] = False
        if self.pose_detector is None:
            self.active_modules["POSE"] = False

    def toggle(self, module: str):
        module = module.upper()
        with self.lock:
            if module not in self.active_modules:
                raise ValueError(f"Unknown module: {module}")
            self.active_modules[module] = not self.active_modules[module]
            return self.active_modules[module]

    def get_status(self):
        with self.lock:
            fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0
            return {
                "fps": fps,
                "face_status": self.face_status,
                "gesture": self.current_gesture,
                "active_modules": self.active_modules.copy(),
            }

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def close(self):
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        if self.hands_detection is not None:
            self.hands_detection.close()
        if self.pose_detector is not None:
            self.pose_detector.close()

    def process_frame(self, frame, recording=False):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_image = None
        if mp is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self.lock:
            modules = dict(self.active_modules)
            frame_idx = self.frame_idx
            prev_time = self.prev_time

        now = time.time()
        dt = now - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        face_status = "NOT DETECTED"
        face_detected = False
        gesture = "NONE"

        if self.yolo_model is not None and modules["YOLO"] and frame_idx % self.yolo_every == 0:
            results = self.yolo_model(frame, verbose=False)
            boxes = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names.get(cls_id, str(cls_id))
                    boxes.append((x1, y1, x2, y2, label, conf))
            with self.lock:
                self.last_yolo_boxes = boxes

        if modules["YOLO"]:
            with self.lock:
                boxes = list(self.last_yolo_boxes)
            for x1, y1, x2, y2, label, conf in boxes:
                hud.bracket_box(frame, x1, y1, x2, y2, hud.ORANGE, 16, 2)
                hud.hud_rect(frame, x1, y1, x2, y2, hud.ORANGE, 1)
                hud.put(frame, f"{label} {conf:.2f}", x1, max(y1 - 8, 14), hud.ORANGE, 0.45)

        if self.face_landmarker is not None and modules["FACE"] and mp_image is not None:
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                face_status = "DETECTED"
                face_detected = True
                for landmarks in face_result.face_landmarks:
                    hud.draw_face_overlay(frame, landmarks, w, h)

        if self.hands_detection is not None and modules["HAND"] and mp_image is not None:
            hand_result = self.hands_detection.detect(mp_image)
            if hand_result.hand_landmarks:
                for landmarks in hand_result.hand_landmarks:
                    hud.draw_normalized_landmarks(
                        frame,
                        landmarks,
                        mp_tasks_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        w,
                        h,
                        point_color=hud.CYAN,
                        line_color=hud.CYAN,
                        point_radius=3,
                        line_thick=2,
                    )
                    gesture = hud.detect_hand_gesture(landmarks)

        if self.pose_detector is not None and modules["POSE"] and mp_image is not None:
            pose_result = self.pose_detector.detect(mp_image)
            if pose_result.pose_landmarks:
                for landmarks in pose_result.pose_landmarks:
                    hud.draw_normalized_landmarks(
                        frame,
                        landmarks,
                        mp_tasks_vision.PoseLandmarksConnections.POSE_LANDMARKS,
                        w,
                        h,
                        point_color=hud.GREEN,
                        line_color=hud.GREEN,
                        point_radius=2,
                        line_thick=2,
                    )

        hud.draw_debug_info(frame, fps, face_status, gesture, modules, face_detected, recording=recording)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg = buf.tobytes() if ok else None

        with self.lock:
            self.prev_time = now
            self.frame_idx = frame_idx + 1
            if dt > 0:
                self.fps_history.append(1.0 / dt)
            self.face_status = face_status
            self.current_gesture = gesture
            if jpeg is not None:
                self.latest_jpeg = jpeg

        return frame