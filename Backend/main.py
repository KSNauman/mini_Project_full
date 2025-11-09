import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# =====================================
# ⚙️ CONFIGURATIONS
# =====================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FPS = 30                    # frames per second
BUFFER_SIZE = 300           # ≈10 seconds buffer
CONFIDENCE = 0.4            # YOLO confidence threshold
ROI_HEIGHT_RATIO = 0.55     # top 55% of face (forehead + under-eye)
ROI_WIDTH_PADDING = 0.15    # 15% padding from left/right
SMOOTH_WINDOW = 5           # moving average window for green signal

# =====================================
# 🧠 BANDPASS FILTER
# =====================================
def bandpass_filter(data, low=0.75, high=3, fs=FPS, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    if len(data) < max(len(a), len(b)) * 3:
        return data
    return filtfilt(b, a, data)

# =====================================
# 📦 MODEL LOAD
# =====================================
model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8n-face-lindevs.onnx")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
model = YOLO(model_path)

# =====================================
# 🚀 FRAME + BPM GENERATOR (for Flask)
# =====================================
def get_frame_and_bpm():
    """
    Continuously captures frames, detects face ROI,
    extracts rPPG signal from the green channel, and yields (frame, bpm).
    Designed for live streaming in Flask frontend.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access webcam")

    green_buffer = deque(maxlen=BUFFER_SIZE)
    bpm = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE)
        if len(results[0].boxes) == 0:
            # no face detected
            cv2.putText(frame, "No Face Detected", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            yield frame, bpm
            continue

        # take largest detected face
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[areas.argmax()].astype(int)

        # Define ROI (forehead + under-eye)
        roi_x1 = x1 + int((x2 - x1) * ROI_WIDTH_PADDING)
        roi_x2 = x2 - int((x2 - x1) * ROI_WIDTH_PADDING)
        roi_y1 = y1
        roi_y2 = y1 + int((y2 - y1) * ROI_HEIGHT_RATIO)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Green channel extraction
        green_channel = roi[:, :, 1]
        mean_green = green_channel.mean() if roi.size > 0 else 0
        green_buffer.append(mean_green)

        # Draw ROI rectangle
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

        # Compute BPM once enough buffer is filled
        if len(green_buffer) >= SMOOTH_WINDOW:
            signal = np.array(green_buffer)
            kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            signal = np.convolve(signal, kernel, mode='valid')

            if len(signal) >= BUFFER_SIZE - SMOOTH_WINDOW + 1:
                signal = signal - np.mean(signal)
                signal = signal / np.std(signal)
                signal = bandpass_filter(signal)

                N = len(signal)
                T = 1 / FPS
                freq = fftfreq(N, T)[:N // 2]
                fft_values = np.abs(fft(signal))[:N // 2]

                idx = (freq >= 0.75) & (freq <= 3)
                if np.any(idx):
                    dominant_freq = freq[idx][np.argmax(fft_values[idx])]
                    bpm = int(dominant_freq * 60)

        # Overlay text on frame
        cv2.putText(frame, f"Heart Rate: {bpm} BPM", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        yield frame, bpm

    cap.release()
