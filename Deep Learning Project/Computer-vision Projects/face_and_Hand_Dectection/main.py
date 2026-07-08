from pathlib import Path
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    FaceLandmarksConnections,
    HandLandmarksConnections,
)

PROJECT_DIR = Path(__file__).resolve().parent
FACE_MODEL = PROJECT_DIR / "models" / "face_landmarker.task"
HAND_MODEL = PROJECT_DIR / "models" / "hand_landmarker.task"


def draw_connections(image, landmarks, connections, line_color, point_color):
    height, width = image.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]
    
    for connection in connections:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(image, start, end, line_color, 1, cv2.LINE_AA)
    
    for point in points:
        cv2.circle(image, point, 1, point_color, -1, cv2.LINE_AA)

def require_models():
    missing = [path for path in (FACE_MODEL, HAND_MODEL) if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing MediaPipe model file(s): {names}. "
            f"Place them in: {FACE_MODEL.parent}"
        )

def main():
    require_models()
   
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions or try another "
            "camera index in cv2.VideoCapture()."
        )
    
    previous_time = time.perf_counter()
    start_time = previous_time

    try:
        with(
            vision.FaceLandmarker.create_from_options(face_options) as face_detector,
            vision.HandLandmarker.create_from_options(hand_options) as hand_detector,
        ):
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.resize(frame, (800, 600))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)

                face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

                for face_landmarks in face_result.face_landmarks:
                   draw_connections(
                       frame,
                       face_landmarks,
                       FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                       line_color=(0, 255, 255),
                       point_color=(255, 0, 255),

                   )

                for hand_landmarks in hand_result.hand_landmarks:
                    draw_connections(
                        frame,
                        hand_landmarks,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        line_color=(0, 255, 0),
                        point_color=(255, 0, 255),
                    )

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                previous_time = current_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Facial and Hand Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
