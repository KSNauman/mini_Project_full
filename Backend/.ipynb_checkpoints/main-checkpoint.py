import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# -------------------------------
# Configurations
# -------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FPS = 30                    # frames per second
BUFFER_SIZE = 300           # approx 10 seconds buffer
CONFIDENCE = 0.4            # YOLO confidence threshold
ROI_HEIGHT_RATIO = 0.55     # top 55% of face (forehead + under-eye)
ROI_WIDTH_PADDING = 0.15    # 15% padding from left/right
SMOOTH_WINDOW = 5           # moving average window for green signal

# -------------------------------
# Bandpass filter for HR
# -------------------------------
def bandpass_filter(data, low=0.75, high=3, fs=FPS, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("models/yolov8n-face-lindevs.onnx")

# Initialize buffer for green channel
green_buffer = deque(maxlen=BUFFER_SIZE)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # YOLO face detection
    # -------------------------------
    results = model(frame, conf=CONFIDENCE)
    if len(results[0].boxes) == 0:
        cv2.imshow("Heart Rate Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Take the first detected face
    box = results[0].boxes.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
    x1, y1, x2, y2 = box.astype(int)

    # -------------------------------
    # Define ROI (forehead + under-eye)
    # -------------------------------
    roi_x1 = x1 + int((x2 - x1) * ROI_WIDTH_PADDING)
    roi_x2 = x2 - int((x2 - x1) * ROI_WIDTH_PADDING)
    roi_y1 = y1
    roi_y2 = y1 + int((y2 - y1) * ROI_HEIGHT_RATIO)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # -------------------------------
    # Green channel extraction
    # -------------------------------
    green_channel = roi[:, :, 1]
    mean_green = green_channel.mean()
    green_buffer.append(mean_green)

    # -------------------------------
    # Signal smoothing
    # -------------------------------
    signal = np.array(green_buffer)
    if len(signal) >= SMOOTH_WINDOW:
        # moving average smoothing
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        signal = np.convolve(signal, kernel, mode='valid')

    # -------------------------------
    # Signal processing and BPM
    # -------------------------------
    bpm = 0
    if len(signal) == BUFFER_SIZE - SMOOTH_WINDOW + 1:
        signal = signal - np.mean(signal)        # detrend
        signal = signal / np.std(signal)         # normalize

        # Optional: bandpass filter
        signal = bandpass_filter(signal)

        N = len(signal)
        T = 1 / FPS
        freq = fftfreq(N, T)[:N//2]
        fft_values = np.abs(fft(signal))[:N//2]

        # Heart rate frequency range: 0.75Hz - 3Hz (45 - 180 BPM)
        idx = (freq >= 0.75) & (freq <= 3)
        if np.any(idx):
            dominant_freq = freq[idx][np.argmax(fft_values[idx])]
            bpm = int(dominant_freq * 60)

    # Display BPM
    cv2.putText(frame, f"Heart Rate: {bpm} BPM", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Heart Rate Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
