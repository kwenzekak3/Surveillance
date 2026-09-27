// Main JavaScript - Updated for 75% accuracy

let socket;
let stream = null;
let isStreaming = false;
let isMuted = false;
let alertTimeout = null;
let frameInterval = null;
let currentAlert = null;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const confidenceFill = document.getElementById('confidence-fill');
const alertDetails = document.getElementById('alert-details');

function initSocket() {
    socket = io({
        transports: ['websocket', 'polling'],
        reconnectionAttempts: 5
    });

    socket.on('connect', () => {
        addLog('Connected to server', 'system');
        updateStatus('online');
    });

    socket.on('disconnect', () => {
        addLog('Disconnected', 'system');
        updateStatus('offline');
    });

    socket.on('intrusion_alert', (data) => {
        const confidence = (data.confidence * 100).toFixed(1);
        currentAlert = data;
        
        // Show alert
        showAlert(confidence);
        
        // Build details string
        let details = `Confidence: ${confidence}% | `;
        details += `Motion: ${data.motion ? 'Yes' : 'No'} | `;
        details += `Objects: ${data.objects || 0}`;
        if (data.details) {
            details += ` | Details: ${JSON.stringify(data.details)}`;
        }
        
        addLog(`🚨 INTRUSION DETECTED: ${confidence}%`, 'detection');
        updateAlertCount();
        
        // Show feedback buttons
        document.getElementById('feedbackCorrect').style.display = 'inline-block';
        document.getElementById('feedbackWrong').style.display = 'inline-block';
    });

    socket.on('frame_update', (data) => {
        document.getElementById('fps').textContent = data.fps;
        document.getElementById('latency').textContent = data.processing_time + 'ms';
        document.getElementById('accuracy-stat').textContent = data.accuracy + '%';
        document.getElementById('accuracy-status').textContent = data.accuracy_status;
        document.getElementById('threshold-display').textContent = `Threshold: ${data.threshold}%`;
        
        // Update threshold slider
        const slider = document.getElementById('threshold');
        if (slider && data.threshold) {
            slider.value = data.threshold;
            document.getElementById('threshold-value').textContent = data.threshold + '%';
        }
    });

    socket.on('connection_ack', () => {
        addLog('Connection acknowledged', 'system');
    });
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 640, height: 480 },
        audio: false
    }).then(streamObj => {
        stream = streamObj;
        video.srcObject = stream;
        video.play();
        isStreaming = true;
        
        document.getElementById('toggleCamera').textContent = '⏹ Stop';
        document.getElementById('toggleCamera').className = 'btn btn-danger';
        addLog('Camera started', 'system');
        sendFrames();
    }).catch(error => {
        addLog('Camera access denied', 'system');
        alert('Please allow camera access.');
    });
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    video.srcObject = null;
    isStreaming = false;
    
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    
    document.getElementById('toggleCamera').textContent = '▶ Start Camera';
    document.getElementById('toggleCamera').className = 'btn btn-primary';
    addLog('Camera stopped', 'system');
}

function sendFrames() {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    
    frameInterval = setInterval(() => {
        if (!isStreaming || !video.readyState === video.HAVE_ENOUGH_DATA) return;
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
        
        if (socket && socket.connected) {
            socket.emit('frame', { image: base64 });
        }
    }, 100);
}

function showAlert(confidence) {
    overlay.style.display = 'flex';
    confidenceFill.style.width = confidence + '%';
    alertDetails.textContent = `Confidence: ${confidence}%`;
    
    if (alertTimeout) clearTimeout(alertTimeout);
    if (!isMuted) {
        alertTimeout = setTimeout(() => {
            overlay.style.display = 'none';
        }, 5000);
    }
}

function updateStatus(status) {
    const el = document.getElementById('status');
    el.textContent = status === 'online' ? '● Online' : '● Offline';
    el.className = status;
}

function updateAlertCount() {
    const el = document.getElementById('alert-count');
    const count = parseInt(el.textContent.split(': ')[1]) + 1;
    el.textContent = `Alerts: ${count}`;
}

function addLog(message, type = 'system') {
    const container = document.getElementById('log-container');
    const entry = document.createElement('p');
    const timestamp = new Date().toLocaleTimeString();
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
    
    if (container.children.length > 100) {
        container.removeChild(container.children[0]);
    }
}

// Feedback functions
function sendFeedback(correct) {
    if (!currentAlert) return;
    
    fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correct: correct })
    });
    
    addLog(`Feedback: ${correct ? '✅ Correct' : '❌ Wrong'}`, 'system');
    
    document.getElementById('feedbackCorrect').style.display = 'none';
    document.getElementById('feedbackWrong').style.display = 'none';
    currentAlert = null;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', initSocket);

document.getElementById('toggleCamera').addEventListener('click', () => {
    isStreaming ? stopCamera() : startCamera();
});

document.getElementById('toggleAlert').addEventListener('click', (e) => {
    isMuted = !isMuted;
    e.target.textContent = isMuted ? '🔊 Unmute' : '🔇 Mute';
    e.target.className = isMuted ? 'btn btn-success' : 'btn btn-danger';
});

document.getElementById('threshold').addEventListener('input', (e) => {
    document.getElementById('threshold-value').textContent = e.target.value + '%';
});

document.getElementById('feedbackCorrect').addEventListener('click', () => {
    sendFeedback(true);
});

document.getElementById('feedbackWrong').addEventListener('click', () => {
    sendFeedback(false);
});

// Cleanup
window.addEventListener('beforeunload', () => {
    if (frameInterval) clearInterval(frameInterval);
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (socket) socket.disconnect();
});