import argparse
import os
from datetime import datetime

import cv2
from vision import VisionEngine

def parse_args():
    p = argparse.ArgumentParser(description="AI Computer Vision HUD")
    p.add_argument("--camera",     type=int,   default=0,           help="Camera index")
    p.add_argument("--width",      type=int,   default=1280,        help="Capture width")
    p.add_argument("--height",     type=int,   default=720,         help="Capture height")
    p.add_argument("--yolo-model", type=str,   default="yolov8n.pt",help="YOLO model")
    p.add_argument("--conf",       type=float, default=0.45,        help="YOLO confidence")
    p.add_argument("--yolo-every", type=int,   default=1,           help="Run YOLO every N frames")
    p.add_argument("--device",     type=str,   default=None,        help="Force device, e.g. cuda:0")
    return p.parse_args()
def main():
    args = parse_args()

    engine = VisionEngine(
        yolo_model_path=args.yolo_model,
        conf=args.conf,
        yolo_every=args.yolo_every,
        device=args.device,
    )

    cap = cv2.VideoCapture(args.camera)
    if args.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera index {args.camera}. "
            "Try a different --camera index, or check OS camera permissions."
        )

    video_writer = None
    recording = False

    os.makedirs("captures", exist_ok=True)
    print("Running.  Press  Q  in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame grab failed, stopping.")
                break

            h, w = frame.shape[:2]

            frame = engine.process_frame(frame, recording=recording)

            if recording and video_writer is not None:
                video_writer.write(frame)

            cv2.imshow("AI Computer Vision HUD", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("1"): engine.toggle("YOLO")
            elif key == ord("2"): engine.toggle("FACE")
            elif key == ord("3"): engine.toggle("HAND")
            elif key == ord("4"): engine.toggle("POSE")
            elif key == ord("s"):
                fname = f"captures/shot_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(fname, frame)
                print(f"Saved screenshot: {fname}")
            elif key == ord("r"):
                if not recording:
                    fname    = f"captures/rec_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(fname, fourcc, 20.0, (w, h))
                    recording = True
                    print(f"Recording started: {fname}")
                else:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("Recording stopped.")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        engine.close()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()