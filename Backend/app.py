from flask import Flask, render_template, Response, request
import threading, time, cv2
from multi_rot_test import get_frame_and_bpm
import numpy as np

# Flask setup
app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

bpm = 0
frame_bytes = None
mode = None

# Background thread to keep capturing frames and BPM
def video_thread():
    global bpm, frame_bytes
    for frame, bpm_val in get_frame_and_bpm():
        bpm = bpm_val
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        time.sleep(0.03)  # ~30 FPS

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
        for frame, bpm_val in get_frame_and_bpm():
            # ✅ Encode frame for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # ✅ Yield as multipart HTTP stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/bpm')
def get_bpm():
    return str(bpm)

if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
