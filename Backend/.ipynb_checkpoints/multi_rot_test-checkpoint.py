import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from camera_source import get_camera_source

# =========================================
# ⚙️ CONFIGURATION
# =========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CONFIDENCE = 0.4
SMOOTH_WINDOW = 5
ROI_DISPLAY = True
FPS = 30
BUFFER_SIZE = int(FPS * 15)
BPM_SMOOTHING_WINDOW = 10

# =========================================
# 🧠 BANDPASS FILTER
# =========================================
def bandpass_filter(data, low=0.75, high=3, fs=FPS, order=4):
    nyq = 0.5 * fs
    lowcut, highcut = low / nyq, high / nyq
    b, a = butter(order, [lowcut, highcut], btype="band")
    if len(data) <= max(len(a), len(b)) * 3:
        return data
    return filtfilt(b, a, data)

# =========================================
# 📦 MODEL LOAD
# =========================================
model_path = "models/yolov8n-face-lindevs.onnx"
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
model = YOLO(model_path)

# =========================================
# 🎥 CAMERA INITIALIZATION
# =========================================
cap = get_camera_source()
if not cap:
    print("[WARN] Using fallback webcam source.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("[INFO] ✅ Camera initialized successfully at 640x480")

# =========================================
# 🧩 CAMERA HEALTH CHECKER
# =========================================
def check_camera_health(cap):
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("[ERROR] ❌ Lost camera feed. Switching to fallback webcam...")
        cap.release()
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Detect frozen / black frame
    if np.mean(frame) < 5:
        print("[WARN] ⚠️ Black frame detected – switching to webcam.")
        cap.release()
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap

# =========================================
# 💾 BUFFERS
# =========================================
forehead_buffer = deque(maxlen=BUFFER_SIZE)
left_eye_buffer = deque(maxlen=BUFFER_SIZE)
right_eye_buffer = deque(maxlen=BUFFER_SIZE)
bpm_history = deque(maxlen=BPM_SMOOTHING_WINDOW)

# =========================================
# 🚀 MAIN LOOP
# =========================================
while True:
    cap = check_camera_health(cap)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed!")
        break

    results = model(frame, conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        cv2.putText(frame, "No Face Detected", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Heart Rate Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # --- Face selection ---
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[areas.argmax()].astype(int)
    face_w, face_h = x2 - x1, y2 - y1
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    # --- ROIs ---
    fh_y1, fh_y2 = y1 + int(0.10 * face_h), y1 + int(0.25 * face_h)
    fh_x1, fh_x2 = x1 + int(0.25 * face_w), x1 + int(0.75 * face_w)
    forehead_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]

    le_y1, le_y2 = y1 + int(0.45 * face_h), y1 + int(0.57 * face_h)
    le_x1, le_x2 = x1 + int(0.18 * face_w), x1 + int(0.38 * face_w)
    left_roi = frame[le_y1:le_y2, le_x1:le_x2]

    re_y1, re_y2 = y1 + int(0.45 * face_h), y1 + int(0.57 * face_h)
    re_x1, re_x2 = x1 + int(0.62 * face_w), x1 + int(0.82 * face_w)
    right_roi = frame[re_y1:re_y2, re_x1:re_x2]

    if ROI_DISPLAY:
        cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (le_x1, le_y1), (le_x2, le_y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (re_x1, re_y1), (re_x2, re_y2), (0, 255, 255), 2)

    # --- Green signal ---
    def mean_green(region): return region[:, :, 1].mean() if region.size else 0
    forehead_buffer.append(mean_green(forehead_roi))
    left_eye_buffer.append(mean_green(left_roi))
    right_eye_buffer.append(mean_green(right_roi))

    bpm_display = 0
    if len(forehead_buffer) >= SMOOTH_WINDOW:
        combined = (0.6 * np.array(forehead_buffer)
                   + 0.2 * np.array(left_eye_buffer)
                   + 0.2 * np.array(right_eye_buffer))
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        signal = np.convolve(combined, kernel, mode="valid")

        if len(signal) >= BUFFER_SIZE - SMOOTH_WINDOW + 1:
            signal = (signal - np.mean(signal)) / np.std(signal)
            signal = bandpass_filter(signal, fs=FPS)
            N = len(signal)
            freq = fftfreq(N, 1 / FPS)[:N // 2]
            fft_values = np.abs(fft(signal))[:N // 2]
            idx = (freq >= 0.75) & (freq <= 3)
            if np.any(idx):
                dominant_freq = freq[idx][np.argmax(fft_values[idx])]
                bpm = int(dominant_freq * 60)
                bpm_history.append(bpm)
                bpm_display = int(np.mean(bpm_history))

    # --- Display ---
    if len(forehead_buffer) < BUFFER_SIZE:
        cv2.putText(frame, "Collecting Signal...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Heart Rate: {bpm_display} BPM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Heart Rate Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
