import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from camera_source import get_camera_source

# =========================================
# ⚙️ CONFIGURATION
# =========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CONFIDENCE = 0.4
FPS = 30
WINDOW_SEC = 8                     # rolling window in seconds
BUFFER_SIZE = int(FPS * WINDOW_SEC)
SMOOTH_WINDOW = 5
BPM_SMOOTHING_WINDOW = 5

# If your camera supports it, you can try locking exposure to reduce flicker noise
TRY_LOCK_EXPOSURE = True

# =========================================
# 🧠 BANDPASS FILTER
# =========================================
def bandpass_filter(data, low=0.8, high=3.0, fs=FPS, order=4):
    if len(data) < 8:
        return data
    nyq = 0.5 * fs
    lowcut, highcut = low / nyq, high / nyq
    b, a = butter(order, [lowcut, highcut], btype="band")
    # filtfilt needs enough samples
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
# 🧩 CAMERA HEALTH CHECKER
# =========================================
def check_camera_health(cap):
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("[ERROR] ❌ Lost camera feed. Switching to fallback webcam...")
        try:
            cap.release()
        except Exception:
            pass
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if np.mean(frame) < 5:
        print("[WARN] ⚠️ Black frame detected – switching to webcam.")
        try:
            cap.release()
        except Exception:
            pass
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap

# =========================================
# 🚀 MAIN GENERATOR FUNCTION
# (accepts optional capture `cap`)
# =========================================
def get_frame_and_bpm(show_overlay=True, cap=None):
    print("[INFO] Initializing camera for frame + BPM generator...")

    # If an existing capture is provided, use it (shared across threads)
    if cap is None:
        cap = get_camera_source()
        if not cap:
            print("[WARN] Using fallback webcam source.")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        print("[INFO] ✅ Using provided camera capture object.")

    # Try to lock exposure if possible (may depend on camera driver)
    if TRY_LOCK_EXPOSURE:
        try:
            # The exact values vary by camera/OS; this is harmless if unsupported
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        except Exception:
            pass

    forehead_buffer = deque(maxlen=BUFFER_SIZE)
    left_eye_buffer = deque(maxlen=BUFFER_SIZE)
    right_eye_buffer = deque(maxlen=BUFFER_SIZE)
    bpm_history = deque(maxlen=BPM_SMOOTHING_WINDOW)

    # helper: compute mean green safely
    def mean_green(region):
        if region is None or region.size == 0:
            return 0.0
        # convert to float for stability
        return float(np.mean(region[:, :, 1]))

    # helper: simple motion detection on ROI to skip bad frames
    def roi_motion(prev_roi, cur_roi):
        if prev_roi is None or cur_roi is None:
            return 0.0
        try:
            prev = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
            cur = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
            # resize to small dims for speed and robustness
            prev = cv2.resize(prev, (32, 32))
            cur = cv2.resize(cur, (32, 32))
            diff = np.abs(prev.astype(np.int16) - cur.astype(np.int16))
            return np.mean(diff) / 255.0
        except Exception:
            return 0.0

    prev_forehead = None

    while True:
        cap = check_camera_health(cap)
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed!")
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)
        # guard in case model returns unexpected
        try:
            boxes = results[0].boxes.xyxy.cpu().numpy()
        except Exception:
            boxes = np.array([])

        # =============================
        # 🧍 FACE DETECTION
        # =============================
        if len(boxes) == 0:
            if show_overlay:
                cv2.putText(frame, "No Face Detected", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # still yield frame, but keep buffers (so user doesn't lose accumulated signal)
            yield frame, 0
            continue

        # --- Face selection (largest) ---
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

        # =============================
        # 💚 APPEND GREEN CHANNELS (critical)
        # =============================
        # motion check: if forehead moved a lot, skip appending to reduce artifact
        motion = roi_motion(prev_forehead, forehead_roi)
        prev_forehead = forehead_roi.copy() if forehead_roi is not None and forehead_roi.size else None

        # If motion is high, skip appending this frame
        if motion < 0.06:  # tuned threshold; adjust if too strict
            forehead_g = mean_green(forehead_roi)
            left_g = mean_green(left_roi)
            right_g = mean_green(right_roi)

            forehead_buffer.append(forehead_g)
            left_eye_buffer.append(left_g)
            right_eye_buffer.append(right_g)
        else:
            # still append last value to keep timing, or skip completely to avoid noise
            if len(forehead_buffer) > 0:
                forehead_buffer.append(forehead_buffer[-1])
                left_eye_buffer.append(left_eye_buffer[-1])
                right_eye_buffer.append(right_eye_buffer[-1])

        # =============================
        # 💚 GREEN SIGNAL + BPM COMPUTATION (STABLE)
        # =============================
        bpm_display = 0
        # ensure we have some data (start after ~4 sec)
        if len(forehead_buffer) > int(FPS * 4):
            combined = (0.6 * np.array(forehead_buffer)
                        + 0.2 * np.array(left_eye_buffer)
                        + 0.2 * np.array(right_eye_buffer))

            # detrend + normalize
            combined = combined - np.mean(combined)
            std = np.std(combined)
            if std > 1e-6:
                combined = combined / std

            # small clipping to reduce extreme outliers
            combined = np.clip(combined, -3.0, 3.0)

            # optional: color-ratio correction (useful in some lighting)
            try:
                if len(left_eye_buffer) == len(forehead_buffer) == len(right_eye_buffer):
                    # use ratio forehead/right as a coarse illumination-invariant feature
                    red_ratio = (np.array(forehead_buffer) /
                                 (np.array(right_eye_buffer) + 1e-6))
                    red_ratio = red_ratio - np.mean(red_ratio)
                    combined = 0.75 * combined + 0.25 * red_ratio
            except Exception:
                pass

            # band-pass filter (physiological range)
            filtered = bandpass_filter(combined, fs=FPS)

            # rolling segment for FFT (last WINDOW_SEC seconds or shorter if start)
            seg_len = min(len(filtered), BUFFER_SIZE)
            segment = filtered[-seg_len:]
            if len(segment) >= FPS * 4:  # require at least 4 s for robust freq estimate
                N = len(segment)
                freq = fftfreq(N, 1 / FPS)[:N // 2]
                fft_values = np.abs(fft(segment))[:N // 2]

                # physiological band: 0.8 - 3.0 Hz -> 48 - 180 BPM
                idx = (freq >= 0.8) & (freq <= 3.0)
                if np.any(idx):
                    freq_band = freq[idx]
                    fft_band = fft_values[idx]

                    # smooth the spectrum a bit
                    if len(fft_band) > 3:
                        fft_band = np.convolve(fft_band, np.ones(3) / 3, mode="same")

                    # get dominant frequency
                    try:
                        dominant_freq = freq_band[np.argmax(fft_band)]
                        bpm_raw = float(dominant_freq * 60.0)

                        # a simple exponential smoothing (like 1D Kalman)
                        if len(bpm_history) == 0:
                            bpm_est = bpm_raw
                        else:
                            bpm_est = 0.9 * float(bpm_history[-1]) + 0.1 * bpm_raw

                        bpm_history.append(bpm_est)
                        # final displayed BPM = mean of last few estimates
                        bpm_display = int(np.round(np.mean(list(bpm_history))))
                    except Exception:
                        pass

        # =============================
        # 🖼️ OVERLAYS (Only if enabled)
        # =============================
        if show_overlay:
            # Draw ROI rectangles (safeguard indices)
            try:
                cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (255, 0, 0), 2)
                cv2.rectangle(frame, (le_x1, le_y1), (le_x2, le_y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (re_x1, re_y1), (re_x2, re_y2), (0, 255, 255), 2)
            except Exception:
                pass

            # Status texts
            if len(forehead_buffer) < BUFFER_SIZE:
                cv2.putText(frame, "Collecting Signal...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Heart Rate: {bpm_display} BPM", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        yield frame, bpm_display

    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()

# =========================================
# 🧪 DEBUG MODE (Standalone)
# =========================================
if __name__ == "__main__":
    for f, bpm in get_frame_and_bpm(show_overlay=True):
        cv2.imshow("Debug Heart Rate Detector", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
