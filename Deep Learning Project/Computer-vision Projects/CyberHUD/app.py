import threading
import time
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from vision import VisionEngine

class HUDState:
    _instance: Optional["HUDState"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, camera=0, width=1280, height=720, yolo_model_path="yolov8n.pt", conf=0.45, yolo_every=1, device=None):
        if self._initialized:
            return
        self._initialized = True
        self.camera_index = camera
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.engine = VisionEngine(
            yolo_model_path=yolo_model_path,
            conf=conf,
            yolo_every=yolo_every,
            device=device,
        )

    def start(self):
        with self.lock:
            if self.running:
                return
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
            self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        self.engine.close()

    def toggle(self, module: str):
        return self.engine.toggle(module)

    def get_status(self):
        return self.engine.get_status()
    
    def get_latest_jpeg(self):
        return self.engine.get_latest_jpeg()
    
    def _capture_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            self.engine.process_frame(frame)

app = FastAPI("CyberHUD: AI Computer Vision HUD API", version="1.0.0")
state = HUDState(camera=0, width=1280, height=720, yolo_model_path="yolov8n.pt")

@app.on_event("startup")
def on_startup():
    state.start()

@app.on_event("shutdown")
def on_shutdown():
    state.stop()

def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = state.get_latest_jpeg()
        if jpeg is None:
            time.sleep(0.03)
            continue
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
               + jpeg + b"\r\n")
        time.sleep(0.01)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
 
 
@app.get("/snapshot")
def snapshot():
    jpeg = state.get_latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")
 
 
@app.get("/status")
def status():
    return state.get_status()
 
 
@app.post("/toggle/{module}")
def toggle_module(module: str):
    try:
        normalized_module = module.upper()
        new_value = state.toggle(normalized_module)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown module '{module}'. "
                                                      f"Use one of: YOLO, FACE, HAND, POSE")
    return {"module": normalized_module, "active": new_value}
 
 
@app.post("/shutdown")
def shutdown():
    state.stop()
    return {"status": "stopped"}
 
 
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Computer Vision HUD</title>
        <style>
            body { background:#06090a; color:#00e5ff; font-family: monospace; text-align:center; }
            img { border: 2px solid #00e5ff; max-width: 95vw; }
            button { background:#0a1a1a; color:#00e5ff; border:1px solid #00e5ff;
                      padding:8px 14px; margin:4px; cursor:pointer; font-family: monospace; }
            button:hover { background:#00e5ff; color:#06090a; }
        </style>
    </head>
    <body>
        <h2>AI COMPUTER VISION HUD</h2>
        <img src="/video_feed" />
        <div>
            <button onclick="toggle('yolo')">Toggle YOLO</button>
            <button onclick="toggle('face')">Toggle FACE</button>
            <button onclick="toggle('hand')">Toggle HAND</button>
            <button onclick="toggle('pose')">Toggle POSE</button>
        </div>
        <pre id="status"></pre>
        <script>
            async function toggle(m) {
                await fetch('/toggle/' + m, { method: 'POST' });
            }
            async function poll() {
                const r = await fetch('/status');
                const j = await r.json();
                document.getElementById('status').textContent = JSON.stringify(j, null, 2);
            }
            setInterval(poll, 1000);
            poll();
        </script>
    </body>
    </html>
    """
