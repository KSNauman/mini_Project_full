from flask import Flask, render_template, Response, request
import threading, time, cv2
from multi_rot_test import get_frame_and_bpm

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

bpm = 0
frame_bytes = None
mode = None
lock = threading.Lock()

video_running = False
mirror_active = False


# ============================================
# 🎥 BACKGROUND THREAD (Web Feed)
# ============================================
def video_thread():
    global bpm, frame_bytes, video_running
    video_running = True
    print("[INFO] Background video thread started...")

    for frame, bpm_val in get_frame_and_bpm():
        if not video_running:
            print("[INFO] Background video thread stopping...")
            break
        bpm = bpm_val
        _, buffer = cv2.imencode('.jpg', frame)
        with lock:
            frame_bytes = buffer.tobytes()
        time.sleep(0.02)

    print("[INFO] Background video thread stopped.")


# ============================================
# 🪞 MIRROR MODE (Safe Switch)
# ============================================
dmirror_active = False
stop_mirror = False

def mirror_window():
    global mirror_active, video_running, stop_mirror
    if mirror_active:
        print("[WARN] Mirror already active.")
        return

    mirror_active = True
    video_running = False
    stop_mirror = False
    print("[INFO] 🪞 Mirror mode activated — switching to native window.")

    WINDOW_X, WINDOW_Y, WINDOW_W, WINDOW_H = 432, 180, 640, 480
    cv2.namedWindow("Mirror View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mirror View", WINDOW_W, WINDOW_H)
    cv2.moveWindow("Mirror View", WINDOW_X, WINDOW_Y)

    for frame, bpm_val in get_frame_and_bpm(show_overlay=False):
        if stop_mirror:
            break
        cv2.imshow("Mirror View", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27, ord('m')]:  # 'q' or ESC or 'm'
            stop_mirror = True
            break

    cv2.destroyAllWindows()
    mirror_active = False
    print("[INFO] 🪞 Mirror mode closed.")
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
    return render_template('index.html')

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
