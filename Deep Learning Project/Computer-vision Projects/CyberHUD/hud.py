import random
import cv2

CYAN      = (255, 230,   0)
GREEN     = (  0, 255,   0)
RED       = (  0,   0, 255)
ORANGE    = (  0, 165, 255)
WHITE     = (255, 255, 255)
GRAY      = ( 90,  90,  90)
DARK_CYAN = (180, 160,   0)
LIME      = (  0, 255, 128)
PANEL_BG  = (  0,  20,  10)

def put(img, text, x, y, color=GREEN, scale=0.55, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)
 
 
def hud_rect(img, x1, y1, x2, y2, color=CYAN, thick=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

def filled_rect(img, x1, y1, x2, y2, color, alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def bracket_box(img, x1, y1, x2, y2, color=CYAN, arm=18, thick=2):
    for cx, cy, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + sx * arm, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * arm), color, thick, cv2.LINE_AA)


def draw_landmarks(img, landmarks, connections, point_color=CYAN, line_color=None, point_radius=2, line_thick=1):
    if line_color is None:
        line_color = point_color

    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        p1 = (int(start.x), int(start.y)) if start.x > 1 or start.y > 1 else None
        p2 = (int(end.x), int(end.y)) if end.x > 1 or end.y > 1 else None
        if p1 is None or p2 is None:
            continue
        cv2.line(img, p1, p2, line_color, line_thick, cv2.LINE_AA)

    for landmark in landmarks:
        x = int(landmark.x)
        y = int(landmark.y)
        if x <= 1 and y <= 1:
            continue
        cv2.circle(img, (x, y), point_radius, point_color, -1, cv2.LINE_AA)


def draw_normalized_landmarks(img, landmarks, connections, width, height, point_color=CYAN, line_color=None, point_radius=2, line_thick=1):
    if line_color is None:
        line_color = point_color

    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))
        cv2.line(img, p1, p2, line_color, line_thick, cv2.LINE_AA)

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(img, (x, y), point_radius, point_color, -1, cv2.LINE_AA)

def detect_hand_gesture(lm):
    landmarks = getattr(lm, "landmark", lm)
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    extended = sum(1 for t, p in zip(tip_ids, pip_ids) if landmarks[t].y < landmarks[p].y)
    thumb_ext = abs(landmarks[4].x - landmarks[2].x) > 0.07
    total = extended + (1 if thumb_ext else 0)
    if total >= 5:
        return "OPEN_PALM"
    if total <= 1:
        return "FIST"
    if total == 2 and extended == 1:
        return "POINTING"
    return "PARTIAL_HAND"

_NN_NODES = None

def _build_nn_nodes(h):
    random.seed(42)
    layers = [4, 5, 4, 3]
    nodes = []
    x_positions = [30, 70, 110, 145]
    for li, (nx, count) in enumerate(zip(x_positions, layers)):
        y_start = int(h * 0.25)
        y_gap = int(h * 0.10)
        offset = int((max(layers) - count) * y_gap / 2)
        for i in range(count):
            nodes.append((nx, y_start + offset + i * y_gap, li))
    return nodes

def draw_nn_graph(img, face_detected):
    global _NN_NODES
    h, w = img.shape[:2]
    if _NN_NODES is None:
        _NN_NODES = _build_nn_nodes(h)
    
    color = CYAN if face_detected else DARK_CYAN
    layers = {}
    for (x, y, li) in _NN_NODES:
        layers.setdefault(li, []).append((x, y))
    
    sorted_keys = sorted(layers.keys())
    for i in range(len(sorted_keys) - 1):
        for x1, y1 in layers[sorted_keys[i]]:
            for x2, y2 in layers[sorted_keys[i + 1]]:
                cv2.line(img, (x1, y1), (x2, y2), (60, 160, 160), 1, cv2.LINE_AA)

    for (x, y, li) in _NN_NODES:
        cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 7, color, 1, cv2.LINE_AA) 

def draw_scan_lines(img):
    h, w = img.shape[:2]
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (20, 60, 60), 1)
 
    arm = 110
    for cx, cy, sx, sy in [(18, 18, 1, 1), (w - 18, 18, -1, 1),
                            (18, h - 18, 1, -1), (w - 18, h - 18, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + sx * arm, cy), CYAN, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * arm), CYAN, 2, cv2.LINE_AA)

def draw_face_overlay(img, landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    if not xs or not ys:
        return

    x = max(min(xs) - 8, 0)
    y = max(min(ys) - 8, 0)
    x2 = min(max(xs) + 8, w - 1)
    y2 = min(max(ys) + 8, h - 1)
    bw = x2 - x
    bh = y2 - y
 
    bracket_box(img, x, y, x2, y2, RED, 20, 2)
    hud_rect(img, x, y, x2, y2, RED, 1)
 
    lbl = "FACIAL RECOGNITION ACTIVE"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    lx = x + (bw - tw) // 2
    ly = y - 10
    put(img, lbl, lx, ly, RED, 0.45, 1)
 
    for idx in (1, 33, 61, 199, 263, 291):
        if idx < len(landmarks):
            kx = int(landmarks[idx].x * w)
            ky = int(landmarks[idx].y * h)
            hud_rect(img, kx - 6, ky - 6, kx + 6, ky + 6, CYAN, 1)
 
    cx, cy = x + bw // 2, y + bh // 2
    cv2.circle(img, (cx, cy), 4, WHITE, -1)
    cv2.line(img, (cx - 18, cy), (cx + 18, cy), WHITE, 1)
    cv2.line(img, (cx, cy - 18), (cx, cy + 18), WHITE, 1)
 

def draw_system_panel(img, fps, face_status, gesture, active_modules, frame_idx=0):
    h, w = img.shape[:2]
    pw, ph = 300, 210
    x1 = w - pw - 15
    y1 = 30
    x2 = w - 15
    y2 = y1 + ph
 
    filled_rect(img, x1, y1, x2, y2, PANEL_BG, 0.55)
    hud_rect(img, x1, y1, x2, y2, GREEN, 1)
 
    put(img, "ARP PROCESS", x1 + 10, y1 + 18, GREEN, 0.45, 1)
    put(img, f"FPS: {fps:.1f}", x1 + 150, y1 + 18, LIME, 0.5, 1)
 
    hud_rect(img, x1, y1 + 22, x2, y1 + 23, GREEN, 1)
 
    scan_lines = [
        "HEALING LIMBS ACTIVE",
        f"BIOMETRIC SCAN {'ON' if face_status != 'NOT DETECTED' else 'OFF'}",
        "BIOMETRIC SCAN IN",
        f"PROGRESS: {'100%' if face_status != 'NOT DETECTED' else '0%'} DONE IN",
        "PROTOCOL SCAN TO",
        f"STATUS: {'OPERATIONAL' if face_status != 'NOT DETECTED' else 'STANDBY'}",
    ]
    for i, line in enumerate(scan_lines):
        put(img, line, x1 + 10, y1 + 42 + i * 20, CYAN, 0.38, 1)
 
    hud_rect(img, x1, y1 + 168, x2, y1 + 169, GRAY, 1)
 
    mods = " ".join(k for k, v in active_modules.items() if v) or "NONE"
    put(img, f"ACTIVE: {mods}", x1 + 10, y1 + 182, CYAN, 0.37, 1)
    put(img, "[1]YOLO [2]FACE [3]HAND [4]POSE", x1 + 10, y1 + 198, GRAY, 0.35, 1)
 

def draw_progress_bars(img, face_detected):
    h, w = img.shape[:2]
    bar_w = 260
    bar_h = 14
    x1 = w - bar_w - 20
    labels = ["cyber evolution:", "evolution level:"]
    values = [100.0, 100.0] if face_detected else [0.0, 0.0]
    y_base = h - 55
 
    for i, (lbl, val) in enumerate(zip(labels, values)):
        y = y_base + i * 30
        put(img, f"{lbl}  {val:.1f}% complete", x1, y - 3, ORANGE, 0.42, 1)
        hud_rect(img, x1, y, x1 + bar_w, y + bar_h, ORANGE, 1)
        fill = int(bar_w * val / 100)
        if fill > 0:
            filled_rect(img, x1, y, x1 + fill, y + bar_h, ORANGE, 0.65)
 

def draw_help_panel(img):
    h, w = img.shape[:2]
    lines = [
        "Show your hand gestures:",
        "\u2022 Open Palm \u2013 Circuit overlays",
        "\u2022 Fist \u2013 Evolution progress",
        "\u2022 Pinch \u2013 Face scanning",
        "  Press 'q' to quit",
    ]
    pw = 230
    lh = 18
    ph = lh * len(lines) + 14
    cx = w // 2
    x1 = cx - pw // 2
    y1 = h - ph - 12
    x2 = cx + pw // 2
    y2 = h - 12
 
    filled_rect(img, x1, y1, x2, y2, (0, 0, 0), 0.45)
    hud_rect(img, x1, y1, x2, y2, GRAY, 1)
 
    for i, line in enumerate(lines):
        clr = CYAN if i == 0 else WHITE
        put(img, line, x1 + 8, y1 + 14 + i * lh, clr, 0.37, 1)
 
def draw_right_decorations(img):
    h, w = img.shape[:2]
    arm = 14
    positions = [(w - 40, h // 3), (w - 40, h // 2), (w - 40, 2 * h // 3)]
    for cx, cy in positions:
        cv2.line(img, (cx - arm, cy), (cx + arm, cy), CYAN, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - arm), (cx, cy + arm), CYAN, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, CYAN, 1, cv2.LINE_AA)
 
def draw_bottom_status(img, gesture, face_status):
    h = img.shape[0]
    biometric_txt = "BIOMETRIC: FACE DETECTED" if face_status != "NOT DETECTED" else "BIOMETRIC: SCANNING..."
    put(img, f"GESTURE: {gesture}", 20, h - 38, CYAN, 0.55, 1)
    put(img, biometric_txt, 20, h - 14, CYAN, 0.55, 1)

def draw_debug_info(frame, fps, face_status, gesture, active_modules, face_detected, recording=False):
    h, w = frame.shape[:2]
    draw_scan_lines(frame)
    draw_nn_graph(frame, face_detected)
    draw_system_panel(frame, fps, face_status, gesture, active_modules)
    draw_progress_bars(frame, face_detected)
    draw_help_panel(frame)
    draw_right_decorations(frame)
    draw_bottom_status(frame, gesture, face_status)
    if recording:
        cv2.circle(frame, (w - 28, h - 28), 8, RED, -1)
        put(frame, "REC", w - 55, h - 22, RED, 0.45)

    return frame


draw_full_hud = draw_debug_info
 