from flask import Flask, render_template, Response, request
import threading, time, cv2
from multi_rot_test import get_frame_and_bpm
import sys

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

bpm = 0
frame_bytes = None
mode = None
lock = threading.Lock()

video_running = False
mirror_active = False
stop_mirror = False

# Shared camera capture object (so both video thread and mirror reuse same feed)
camera_cap = None

# ============================================
# 🎥 BACKGROUND THREAD (Web Feed)
# ============================================
def video_thread():
    global bpm, frame_bytes, video_running, camera_cap
    video_running = True
    print("[INFO] Background video thread started...")

    # Initialize camera once
    if camera_cap is None:
        try:
            from multi_rot_test import get_camera_source
            camera_cap = get_camera_source()
        except Exception:
            camera_cap = None
        if not camera_cap:
            print("[WARN] get_camera_source() failed — using fallback webcam.")
            camera_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ✅ Keep one persistent generator alive
    generator = get_frame_and_bpm(cap=camera_cap)

    try:
        for frame, bpm_val in generator:
            if not video_running:
                print("[INFO] Background video thread stopping...")
                break
            bpm = bpm_val
            _, buffer = cv2.imencode('.jpg', frame)
            with lock:
                frame_bytes = buffer.tobytes()
            # Give time for Flask response but not too slow
            time.sleep(0.03)
    except Exception as e:
        print("[ERROR] Exception in video_thread:", e)
    finally:
        print("[INFO] Background video thread stopped.")


# ============================================
# 🪞 MIRROR MODE (Safe Switch)
# ============================================
def mirror_window():
    global mirror_active, video_running, stop_mirror, camera_cap
    if mirror_active:
        print("[WARN] Mirror already active.")
        return

    mirror_active = True
    # Stop web feed so mirror can take the same camera safely
    video_running = False
    stop_mirror = False
    print("[INFO] 🪞 Mirror mode activated — switching to native window.")

    WINDOW_X, WINDOW_Y, WINDOW_W, WINDOW_H = 432, 180, 640, 480
    cv2.namedWindow("Mirror View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mirror View", WINDOW_W, WINDOW_H)
    cv2.moveWindow("Mirror View", WINDOW_X, WINDOW_Y)

    # Ensure we have a camera_cap to pass to generator
    if camera_cap is None:
        try:
            cap = get_frame_and_bpm.__globals__['get_camera_source']()
        except Exception:
            cap = None
        if not cap:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera_cap = cap

    try:
        for frame, bpm_val in get_frame_and_bpm(show_overlay=False, cap=camera_cap):
            if stop_mirror:
                break
            cv2.imshow("Mirror View", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27, ord('m')]:  # 'q' or ESC or 'm'
                stop_mirror = True
                break
    except Exception as e:
        print("[ERROR] Exception in mirror_window:", e, file=sys.stderr)
    finally:
        cv2.destroyAllWindows()
        mirror_active = False
        print("[INFO] 🪞 Mirror mode closed.")
        # Restart background thread so web feed resumes
        threading.Thread(target=video_thread, daemon=True).start()

@app.route('/mirror_stop', methods=['POST'])
def mirror_stop():
    global stop_mirror
    stop_mirror = True
    print("[INFO] Mirror stop signal received.")
    return "Mirror stopped."

# ============================================
# 🌐 ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html', current_year="2025")

@app.route('/start', methods=['POST'])
def start():
    global mode
    mode = request.form.get('mode')
    threading.Thread(target=video_thread, daemon=True).start()
    return render_template('monitor.html', mode=mode)

@app.route('/video_feed')
def video_feed():
    def gen():
        global frame_bytes
        while True:
            with lock:
                if frame_bytes is None:
                    continue
                fb = frame_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fb + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mirror', methods=['POST'])
def mirror_mode():
    threading.Thread(target=mirror_window, daemon=True).start()
    return "Mirror mode started."

@app.route('/bpm')
def get_bpm():
    return str(bpm)

# ============================================
# 🚀 RUN
# ============================================
if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
