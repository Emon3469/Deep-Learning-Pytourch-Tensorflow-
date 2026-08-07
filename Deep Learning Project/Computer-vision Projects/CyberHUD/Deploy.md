# CyberHUD Deploy Note

This project is a webcam-driven desktop application, not a Render web app.

## Why It Should Not Go On Render

- It depends on local camera access through OpenCV.
- It opens an interactive desktop window with `cv2.imshow()`.
- Render does not provide webcam hardware or a desktop GUI session.

## Local Run

1. Install the project dependencies.
2. Run `python main.py` from this folder.
3. Make sure your camera permissions are enabled.
4. Adjust the `--camera` argument if the default device index does not work.

## If You Need A Web Demo

Rebuild the project as a browser app with an upload flow or a live-streaming backend before deploying it to Render.
