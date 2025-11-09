import cv2
import time

def get_camera_source():
    print("[INFO] Checking for connected cameras...")

    preferred_indexes = [1, 2]  # try likely DroidCam/Iriun indexes first

    for idx in preferred_indexes:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        time.sleep(0.3)  # wait a bit for initialization
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[INFO] ✅ Found active external camera on index {idx}")
            return cap
        cap.release()

    print("[WARN] ⚙️ No external camera detected. Returning None.")
    return None
