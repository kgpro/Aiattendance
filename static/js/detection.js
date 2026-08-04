// Face Detection page JavaScript

class FaceDetectionApp {
    constructor() {
        this.socket = null;
        this.isProcessing = false;
        this.canvas = document.getElementById('processed-video');
        this.ctx = this.canvas.getContext('2d');
        this.startBtn = document.getElementById('start-camera');
        this.stopBtn = document.getElementById('stop-camera');
        this.resultsContainer = document.getElementById('detection-results');
        this.attendanceBody = document.getElementById('attendance-table-body');
        this.connectionStatus = document.getElementById('connection-status');
        this.cameraPlaceholder = document.getElementById('camera-placeholder');
        this.frameCounter = 0;
        this.stream = null;

        // Stats
        this.totalStudentsEl = document.getElementById('total-students');
        this.todayAttendanceEl = document.getElementById('today-attendance');
        this.currentTimeEl = document.getElementById('current-time');
        this.detectionSpeedEl = document.getElementById('detection-speed');

        this.startBtn.addEventListener('click', () => this.startDetection());
        this.stopBtn.addEventListener('click', () => this.stopDetection());

        this.updateCurrentTime();
        setInterval(() => this.updateCurrentTime(), 1000);
        this.fetchStats();
    }

    updateCurrentTime() {
        const now = new Date();
        if (this.currentTimeEl) {
            this.currentTimeEl.textContent = now.toLocaleTimeString();
        }
    }

    async fetchStats() {
        try {
            const resp = await fetch('/api/attendance/stats/');
            const data = await resp.json();
            if (this.totalStudentsEl) this.totalStudentsEl.textContent = data.total_students || 0;
            if (this.todayAttendanceEl) this.todayAttendanceEl.textContent = (data.attendance_rate || 0) + '%';
        } catch (e) {
            console.error('Stats error:', e);
        }
    }

    updateConnectionStatus(status, msg) {
        this.connectionStatus.className = 'connection-status';
        if (status === 'connected') {
            this.connectionStatus.classList.add('status-connected');
            this.connectionStatus.innerHTML = `<i class="fas fa-plug"></i> Connected`;
        } else if (status === 'connecting') {
            this.connectionStatus.classList.add('status-connecting');
            this.connectionStatus.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Connecting...`;
        } else {
            this.connectionStatus.classList.add('status-disconnected');
            this.connectionStatus.innerHTML = `<i class="fas fa-plug"></i> Disconnected`;
        }
    }

    async startDetection() {
        if (this.isProcessing) return;
        try {
            this.updateConnectionStatus('connecting');

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width:480, height:360, facingMode: 'user' }
            });
            this.stream = stream;
            const track = stream.getVideoTracks()[0];
            const settings = track.getSettings();
            this.canvas.width = settings.width || 640;
            this.canvas.height = settings.height || 480;

            await this.connectWebSocket();

            this.isProcessing = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            if (this.cameraPlaceholder) this.cameraPlaceholder.style.display = 'none';

            this.processFrames(stream);
        } catch (err) {
            console.error('Start detection error:', err);
            this.updateConnectionStatus('disconnected');
            this.showError('Camera error: ' + err.message);
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/face-detection/`;
            this.socket = new WebSocket(wsUrl);

            this.socket.onopen = () => {
                this.updateConnectionStatus('connected');
                resolve();
            };
            this.socket.onmessage = (e) => this.handleMessage(e.data);
            this.socket.onclose = () => {
                this.updateConnectionStatus('disconnected');
                if (this.isProcessing) this.stopDetection();
            };
            this.socket.onerror = (err) => reject(err);
        });
    }

    async processFrames(stream) {
        const video = document.createElement('video');
        video.srcObject = stream;
        await video.play();

        const process = async () => {
            if (!this.isProcessing || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
                requestAnimationFrame(process);
                return;
            }

            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            tempCtx.drawImage(video, 0, 0);
            tempCanvas.toBlob((blob) => {
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(blob);
                }
            }, 'image/jpeg', 0.8);

            // Update speed (approximate)
            const now = performance.now();
            if (this._lastFrameTime) {
                const diff = now - this._lastFrameTime;
                if (this.detectionSpeedEl) this.detectionSpeedEl.textContent = Math.round(diff) + 'ms';
            }
            this._lastFrameTime = now;

            this.frameCounter++;
            setTimeout(() => requestAnimationFrame(process), 100);
        };
        process();
    }

    handleMessage(data) {
        try {
            const msg = JSON.parse(data);
            if (msg.type === 'detection_results') {
                this.displayFrame(msg.frame);
                this.updateDetectionResults(msg.detections);
            } else if (msg.type === 'recognition_results') {
                this.addAttendanceRow(msg.recognition);
                this.updateRecognitionInResults(msg.recognition);
            } else if (msg.type === 'error') {
                this.showError(msg.message);
            }
        } catch (e) {
            console.error('Message parse error:', e);
        }
    }

    displayFrame(base64Frame) {
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        };
        img.src = 'data:image/jpeg;base64,' + base64Frame;
    }

    updateDetectionResults(detections) {
        this.resultsContainer.innerHTML = '';
        if (!detections || detections.length === 0) {
            this.resultsContainer.innerHTML = '<div class="detection-result">No faces detected</div>';
            return;
        }
        detections.forEach(det => {
            const div = document.createElement('div');
            div.className = 'detection-result unknown';
            div.dataset.bbox = JSON.stringify(det.bbox);
            div.innerHTML = `
                <div class="result-header">
                    <span class="result-title">Face detected</span>
                    <span class="result-time">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="result-details">
                    Confidence: <span class="value">${(det.confidence * 100).toFixed(1)}%</span>
                </div>
            `;
            this.resultsContainer.appendChild(div);
        });
    }

    updateRecognitionInResults(recognition) {
        // Find the result with matching bbox (approximate)
        const items = this.resultsContainer.querySelectorAll('.detection-result');
        const bbox = recognition.bbox;
        if (!bbox) return;
        const bboxStr = JSON.stringify(bbox);
        items.forEach(el => {
            if (el.dataset.bbox === bboxStr) {
                el.classList.remove('unknown');
                el.classList.add('recognized');
                const title = el.querySelector('.result-title');
                if (title) title.textContent = `✅ ${recognition.person_name}`;
                const details = el.querySelector('.result-details');
                if (details) {
                    details.innerHTML = `
                        Confidence: <span class="value">${(recognition.confidence * 100).toFixed(1)}%</span>
                        &nbsp;| Distance: <span class="value">${recognition.distance.toFixed(4)}</span>
                    `;
                }
            }
        });
    }

    addAttendanceRow(recognition) {
        const row = document.createElement('tr');
        const now = new Date().toLocaleTimeString();
        row.innerHTML = `
            <td>${recognition.student_id}</td>
            <td>${recognition.person_name}</td>
            <td>${now}</td>
            <td><span class="status-badge status-present">Present</span></td>
            <td>${(recognition.confidence * 100).toFixed(1)}%</td>
        `;
        this.attendanceBody.prepend(row);
    }

    showError(msg) {
        this.resultsContainer.innerHTML = `<div class="detection-result error">⚠️ ${msg}</div>`;
    }

    stopDetection() {
        this.isProcessing = false;
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        if (this.cameraPlaceholder) this.cameraPlaceholder.style.display = 'flex';
        this.updateConnectionStatus('disconnected');
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = 'white';
        this.ctx.font = '20px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Detection stopped', this.canvas.width/2, this.canvas.height/2);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new FaceDetectionApp();
});