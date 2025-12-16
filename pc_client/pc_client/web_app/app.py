import cv2
import sys
import os
import time
import threading
from flask import Flask, render_template, Response, jsonify, request

# Add the project root to path
current_dir = os.path.dirname(os.path.abspath(__file__)) # web_app
parent_dir = os.path.dirname(current_dir) # pc_client (inner)
outer_pc_client = os.path.dirname(parent_dir) # pc_client (outer)
project_root = os.path.dirname(outer_pc_client) # Mini QWEN

if project_root not in sys.path:
    sys.path.append(project_root)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vlm.model import VLMModel

app = Flask(__name__)

# Configuration - Default to local webcam
DEFAULT_SOURCE = 0

# Global variables
model = None
video_stream = None
lock = threading.Lock()
last_analysis = {"time": 0, "result": "Waiting for machine...", "status": "waiting"}
analysis_history = []
ANALYSIS_INTERVAL = 7  # Analyze every 7 seconds

class VideoStream:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    # Reconnect if stream lost
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.source)
            else:
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

def get_model():
    global model
    if model is None:
        model = VLMModel()
    return model

def get_video_stream():
    global video_stream
    if video_stream is None:
        video_stream = VideoStream(DEFAULT_SOURCE)
    return video_stream

def change_video_source(new_source):
    global video_stream
    if video_stream is not None:
        video_stream.stop()
    
    # Convert to int if it's a digit (for webcam index)
    if str(new_source).isdigit():
        new_source = int(new_source)
        
    video_stream = VideoStream(new_source)
    return True

def generate_frames():
    stream = get_video_stream()
    while True:
        frame = stream.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/analyze', methods=['POST'])
def analyze_frame():
    stream = get_video_stream()
    frame = stream.get_frame()
    
    if frame is None:
        return jsonify({'error': 'No frame available from stream'}), 503
        
    vlm = get_model()
    
    # Run analysis
    try:
        prediction = vlm.analyze(frame)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_latest_analysis', methods=['GET'])
def get_latest_analysis():
    """Get the latest analysis result without triggering new analysis"""
    global last_analysis
    return jsonify(last_analysis)

@app.route('/api/get_history', methods=['GET'])
def get_history():
    """Get analysis history"""
    global analysis_history
    return jsonify({'history': analysis_history[-5:]})  # Last 5 results

def continuous_analysis():
    """Background thread that continuously analyzes frames"""
    global last_analysis, analysis_history
    
    while True:
        try:
            current_time = time.time()
            
            # Check if enough time has passed
            if current_time - last_analysis["time"] >= ANALYSIS_INTERVAL:
                stream = get_video_stream()
                frame = stream.get_frame()
                
                if frame is not None:
                    vlm = get_model()
                    
                    # Run analysis
                    prediction = vlm.analyze(frame)
                    
                    # Update last analysis
                    last_analysis = {
                        "time": current_time,
                        "result": prediction,
                        "status": "analyzed",
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    
                    # Add to history
                    analysis_history.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "result": prediction[:100] + "..." if len(prediction) > 100 else prediction
                    })
                    
                    # Keep only last 10 results
                    if len(analysis_history) > 10:
                        analysis_history.pop(0)
                else:
                    last_analysis["status"] = "no_frame"
                    last_analysis["result"] = "No video frame available"
                    
        except Exception as e:
            print(f"[Analysis Error] {e}")
            last_analysis["status"] = "error"
            last_analysis["result"] = f"Error: {str(e)}"
        
        time.sleep(1)  # Check every second

@app.route('/api/change_source', methods=['POST'])
def change_source():
    try:
        data = request.get_json()
        source = data.get('source')
        
        if source is None:
            return jsonify({'error': 'Source parameter is required'}), 400
        
        change_video_source(source)
        return jsonify({'status': 'success', 'message': f'Switched to source: {source}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start continuous analysis thread
    analysis_thread = threading.Thread(target=continuous_analysis, daemon=True)
    analysis_thread.start()
    print("[Server] Continuous analysis thread started")
    
    # Initialize model on startup (optional, can be lazy loaded)
    # get_model()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
