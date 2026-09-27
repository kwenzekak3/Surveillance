# Main application - Complete version with all detectors

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import time
import logging
import os
from datetime import datetime

from config import Config
from motion_detector import MotionDetector
from behavior_analyzer import BehaviorAnalyzer
from object_detector import ObjectDetector
from decision_fusion import DecisionFusion

# Print OpenCV version for debugging
print(f"OpenCV version: {cv2.__version__}")

# Initialize Flask
app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60)

# Initialize all detectors
motion_detector = MotionDetector(threshold=Config.MOTION_THRESHOLD)
behavior_analyzer = BehaviorAnalyzer()
object_detector = ObjectDetector()
decision_fusion = DecisionFusion()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/security.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Statistics
stats = {
    'frames_processed': 0,
    'detections': 0,
    'alerts': 0,
    'start_time': time.time(),
    'accuracy': 0.0,
    'accuracy_status': 'Starting...'
}

# ========== ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    accuracy, status = decision_fusion.get_accuracy()
    stats['accuracy'] = accuracy
    stats['accuracy_status'] = status
    
    return jsonify({
        'frames': stats['frames_processed'],
        'detections': stats['detections'],
        'alerts': stats['alerts'],
        'uptime': time.time() - stats['start_time'],
        'accuracy': round(accuracy, 1),
        'accuracy_status': status,
        'threshold': round(decision_fusion.threshold * 100, 1)
    })

@app.route('/api/feedback', methods=['POST'])
def provide_feedback():
    """User feedback endpoint"""
    data = request.json
    is_correct = data.get('correct', True)
    decision_fusion.provide_feedback(is_correct)
    return jsonify({'status': 'Feedback received'})

# ========== WEBSOCKET EVENTS ==========

@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")
    emit('connection_ack', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    """Process incoming video frame"""
    try:
        start_time = time.time()
        
        # Decode frame
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        stats['frames_processed'] += 1
        
        # === RUN ALL DETECTORS ===
        
        # 1. Motion Detection
        motion_detected, motion_conf = motion_detector.detect(frame)
        motion_result = {'confidence': motion_conf} if motion_detected else None
        
        # 2. Behavior Analysis
        behavior_alerts = behavior_analyzer.analyze(frame, [])
        behavior_result = {
            'confidence': 0.70 if behavior_alerts else 0.0,
            'alerts': behavior_alerts
        } if behavior_alerts else None
        
        # 3. Object Detection (NEW!)
        object_conf, object_detections = object_detector.detect(frame)
        object_result = {
            'confidence': object_conf,
            'detections': object_detections
        } if object_conf > 0.3 else None
        
        # === DECISION FUSION ===
        decision, confidence, details = decision_fusion.decide(
            motion_result, 
            None,  # No face recognition
            behavior_result, 
            object_result
        )
        
        # === HANDLE DECISION ===
        if decision:
            stats['detections'] += 1
            stats['alerts'] += 1
            
            alert_data = {
                'type': '🚨 INTRUSION DETECTED',
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'motion': motion_detected,
                'behavior': len(behavior_alerts) if behavior_alerts else 0,
                'objects': len(object_detections) if object_detections else 0,
                'details': details
            }
            
            emit('intrusion_alert', alert_data, broadcast=True)
            logging.warning(f"ALERT: Intrusion detected with {confidence:.0%} confidence")
            logging.info(f"  Details: {details}")
            
            # Draw alert on frame
            cv2.putText(frame, f"🚨 INTRUSION: {confidence:.0%}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 3)
            
            # Show detection details
            y_pos = 60
            for key, value in details.items():
                if key != 'boost' and value > 0.1:
                    cv2.putText(frame, f"{key}: {value:.0%}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (255, 255, 0), 1)
                    y_pos += 25
        else:
            # Show monitoring status
            cv2.putText(frame, f"Monitoring...", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
        
        # Show motion status
        if motion_detected:
            cv2.putText(frame, "🟡 Motion Detected", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 255, 255), 1)
        
        # Show object detection count
        if object_result and object_detections:
            cv2.putText(frame, f"👤 Objects: {len(object_detections)}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 255, 255), 1)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        
        # Get accuracy
        accuracy, status = decision_fusion.get_accuracy()
        
        # Send processed frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('frame_update', {
            'image': frame_b64,
            'fps': round(fps, 1),
            'processing_time': round(processing_time * 1000, 1),
            'motion': motion_detected,
            'alerts': stats['alerts'],
            'accuracy': round(accuracy, 1),
            'accuracy_status': status,
            'threshold': round(decision_fusion.threshold * 100, 1)
        })
        
    except Exception as e:
        logging.error(f"Frame error: {str(e)}")
        print(f"❌ Frame error: {e}")

# ========== RUN ==========

if __name__ == '__main__':
    print("=" * 60)
    print("🔒 SURVEILLANCE INTRUSION DETECTION SYSTEM")
    print("=" * 60)
    print(f"🌐 Server: http://127.0.0.1:5000")
    print("")
    print("📊 Detection Modules:")
    print("  ✅ Motion Detection (65-70%)")
    print("  ✅ Behavior Analysis (65-70%)")
    print("  ✅ Object Detection (NEW! 70-75%)")
    print("  ✅ Decision Fusion (75%+ TARGET!)")
    print("")
    print("🎯 Target: 75% Accuracy")
    print("💡 Features:")
    print("   • People detection")
    print("   • Face detection")
    print("   • Motion tracking")
    print("   • Behavior analysis")
    print("   • Smart decision fusion")
    print("")
    print("Press CTRL+C to stop")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)