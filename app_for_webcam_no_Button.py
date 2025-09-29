import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# كاميرا IP
CAM_URL = "http://192.168.1.3:8080/video"
cap = cv2.VideoCapture(CAM_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

effect_mode = 1
hue_offset = 0
frame_count = 0

# أزرار عرض على الشاشة فقط
buttons = [
    {"label": "0", "color": (150,150,150)},
    {"label": "1", "color": (0,0,255)},
    {"label": "2", "color": (0,255,255)},
    {"label": "3", "color": (0,255,0)},
    {"label": "4", "color": (255,0,0)},
    {"label": "5", "color": (200,0,200)},
]

def clamp_point(pt, w, h):
    x, y = pt
    return (max(0, min(w-1, int(x))), max(0, min(h-1, int(y))))

def apply_effect_masked(frame, mask, mode, hue_offset):
    effect_frame = frame.copy()
    h, w = mask.shape
    ys, xs = np.where(mask == 255)
    if ys.size == 0:
        return frame

    if mode == 1:  # Invert
        full = cv2.bitwise_not(frame)
    elif mode == 2:  # Blur
        blurred = cv2.GaussianBlur(frame, (15,15), 0)
        full = cv2.addWeighted(blurred, 0.8, frame, 0.2, 0)
    elif mode == 3:  # Edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        full = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif mode == 4:  # Sepia
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])
        full = np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)
    elif mode == 5:  # Color Shift
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[...,0] = (hsv[...,0] + hue_offset) % 180
        full = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    else:
        return frame

    effect_frame[ys, xs] = full[ys, xs]
    return effect_frame

def generate_frames():
    global effect_mode, hue_offset, frame_count
    while True:
        frame_count += 1
        # تعالج كل إطار ثاني فقط لتسريع الأداء
        if frame_count % 2 != 0:
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        out = frame.copy()

        # رسم الأزرار (رقمية صغيرة على الشاشة)
        start_y = 10
        for btn in buttons:
            x = w - 30
            y = start_y
            cv2.putText(out, btn["label"], (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, btn["color"], 2)
            start_y += 25

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            try:
                l_thumb = results.multi_hand_landmarks[0].landmark[4]
                l_index = results.multi_hand_landmarks[0].landmark[8]
                r_thumb = results.multi_hand_landmarks[1].landmark[4]
                r_index = results.multi_hand_landmarks[1].landmark[8]

                p1 = clamp_point((l_index.x * w, l_index.y * h), w, h)
                p2 = clamp_point((l_thumb.x * w, l_thumb.y * h), w, h)
                p3 = clamp_point((r_thumb.x * w, r_thumb.y * h), w, h)
                p4 = clamp_point((r_index.x * w, r_index.y * h), w, h)

                poly = np.array([p1,p2,p3,p4], dtype=np.int32)
                if cv2.contourArea(poly) > 300:
                    mask = np.zeros((h,w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    hue_offset = (hue_offset + 2) % 180
                    out = apply_effect_masked(out, mask, effect_mode, hue_offset)
                    cv2.polylines(out, [poly], True, (0,255,0), 2)
            except:
                pass

        _, buffer = cv2.imencode('.jpg', out)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

from flask import render_template

@app.route('/')
def index():
    return render_template('index_v2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
