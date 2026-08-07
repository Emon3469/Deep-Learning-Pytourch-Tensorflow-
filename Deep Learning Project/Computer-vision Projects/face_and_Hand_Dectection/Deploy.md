# Face And Hand Detection Deploy Note

This project is a webcam-driven desktop application, not a Render web app.

## Why It Should Not Go On Render

- It reads frames from a local webcam.
- It opens an OpenCV window for live visualization.
- Render cannot provide the required interactive camera and GUI environment.

## Local Run

1. Install the project dependencies.
2. Place the required MediaPipe model files in `models/`.
3. Run `python main.py` from this folder.
4. Press `q` to quit the OpenCV window.

## If You Need A Web Demo

Convert the camera pipeline into an upload-based or browser-streaming app before trying to host it on Render.
