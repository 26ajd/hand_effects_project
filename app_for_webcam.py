import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

# MediaPipe Hands (أخف إعدادات عشان يكون سريع)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,  # أسرع
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 🎥 رابط الكاميرا (غير IP لو جهازك مختلف)
CAM_URL = "http://192.168.1.3:8080/video"
cap = cv2.VideoCapture(CAM_URL)

# قلل الدقة عشان السرعة
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# تأثير افتراضي
effect_mode = 1
hue_offset = 0

def clamp_point(pt, w, h):
    x, y = pt
    return (max(0, min(w-1, int(x))), max(0, min(h-1, int(y))))

def apply_effect_masked(frame, mask, mode, hue_offset):
    effect_frame = frame.copy()
    if mask is None or np.count_nonzero(mask) == 0:
        return frame

    # تحديد مستطيل الحدود للقناع لتقليل المعالجة
    ys, xs = np.where(mask == 255)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    roi = frame[y_min:y_max+1, x_min:x_max+1]
    mask_roi = mask[y_min:y_max+1, x_min:x_max+1]

    # تطبيق التأثير على ROI فقط
    if mode == 1:  # Invert
        full = cv2.bitwise_not(roi)
    elif mode == 2:  # Blur
        full = cv2.blur(roi, (15,15))  # أسرع من Gaussian كبير
    elif mode == 3:  # Edges
        edges = cv2.Canny(roi, 50, 150)
        full = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif mode == 4:  # Sepia
        kernel = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])
        full = np.clip(cv2.transform(roi, kernel),0,255).astype(np.uint8)
    elif mode == 5:  # Color Shift
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[...,0] = (hsv[...,0] + hue_offset) % 180
        full = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    else:
        full = roi

    # دمج التأثير مع المنطقة الأصلية
    roi[mask_roi==255] = full[mask_roi==255]
    effect_frame[y_min:y_max+1, x_min:x_max+1] = roi
    return effect_frame


def generate_frames():
    global effect_mode, hue_offset
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ لم يتم التقاط إطار من كاميرا IP")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        out = frame.copy()

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

                poly = np.array([p1, p2, p3, p4], dtype=np.int32)
                if cv2.contourArea(poly) > 500:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    hue_offset = (hue_offset + 2) % 180
                    out = apply_effect_masked(out, mask, effect_mode, hue_offset)
                    cv2.polylines(out, [poly], True, (0, 255, 0), 2)
            except Exception as e:
                print("Hand landmark error:", e)

        _, buffer = cv2.imencode('.jpg', out)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_mode/<int:mode>')
def set_mode(mode):
    global effect_mode
    if 0 <= mode <= 5:
        effect_mode = mode
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


