import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

cap = cv2.VideoCapture(0)

effect_mode = 1
hue = 0

def hsv_to_bgr(hue):
    hsv = np.uint8([[[hue % 180, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def generate_frames():
    global effect_mode, hue
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape
        hue += 2

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            p1 = (int(hand1.landmark[4].x * w), int(hand1.landmark[4].y * h))
            p2 = (int(hand1.landmark[8].x * w), int(hand1.landmark[8].y * h))
            p3 = (int(hand2.landmark[4].x * w), int(hand2.landmark[4].y * h))
            p4 = (int(hand2.landmark[8].x * w), int(hand2.landmark[8].y * h))

            x1 = min(p1[0], p2[0], p3[0], p4[0])
            y1 = min(p1[1], p2[1], p3[1], p4[1])
            x2 = max(p1[0], p2[0], p3[0], p4[0])
            y2 = max(p1[1], p2[1], p3[1], p4[1])

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]

                if effect_mode == 1:
                    roi = cv2.bitwise_not(roi)
                elif effect_mode == 2:
                    roi = cv2.GaussianBlur(roi, (35, 35), 0)
                elif effect_mode == 3:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                elif effect_mode == 4:
                    kernel = np.array([[0.272, 0.534, 0.131],
                                       [0.349, 0.686, 0.168],
                                       [0.393, 0.769, 0.189]])
                    roi = cv2.transform(roi, kernel)
                    roi = np.clip(roi, 0, 255)
                elif effect_mode == 5:
                    overlay = np.full_like(roi, hsv_to_bgr(hue))
                    roi = cv2.addWeighted(roi, 0.5, overlay, 0.5, 0)

                frame[y1:y2, x1:x2] = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, hand2, mp_hands.HAND_CONNECTIONS)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_mode/<int:mode>')
def set_mode(mode):
    global effect_mode
    effect_mode = mode
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
