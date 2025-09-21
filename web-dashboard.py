#!/usr/bin/env python3
"""
SmartRig AI Tuner Pro - Web Dashboard
Browser-based monitoring interface using Flask (no API keys required)
Access from any device on your network!
"""

import os
import sys
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file
import psutil
import numpy as np
from collections import deque

# Try to import GPUtil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartrig-tuner-2025'

# Global variables for data sharing
current_metrics = {}
metrics_history = deque(maxlen=60)  # Last 60 seconds
alerts = deque(maxlen=20)

# HTML Template with embedded CSS and JavaScript
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartRig AI Tuner - Web Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            color: #333;
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-title {
            font-size: 18px;
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: #f3f4f6;
            border-radius: 8px;
        }
        
        .metric-label {
            color: #6b7280;
            font-size: 14px;
        }
        
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .progress-fill.warning {
            background: linear-gradient(90deg, #f59e0b, #d97706);
        }
        
        .progress-fill.danger {
            background: linear-gradient(90deg, #ef4444, #dc2626);
        }
        
        .chart-container {
            width: 100%;
            height: 200px;
            margin-top: 15px;
        }
        
        canvas {
            width: 100% !important;
            height: 100% !important;
        }
        
        .alert {
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .alert-info {
            background: #dbeafe;
            color: #1e40af;
        }
        
        .alert-warning {
            background: #fed7aa;
            color: #92400e;
        }
        
        .alert-danger {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: #6366f1;
            color: white;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #4f46e5;
        }
        
        button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-box {
            text-align: center;
            padding: 15px;
            background: #f9fafb;
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        
        .stat-label {
            font-size: 12px;
            color: #6b7280;
            margin-top: 5px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                <span class="status-indicator"></span>
                SmartRig AI Tuner - Web Dashboard
            </h1>
        </header>
        
        <div class="dashboard">
            <!-- CPU Card -->
            <div class="card">
                <h2 class="card-title">🖥️ CPU Performance</h2>
                <div class="metric">
                    <span class="metric-label">Usage</span>
                    <span class="metric-value" id="cpu-usage">--</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="cpu-progress"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Temperature</span>
                    <span class="metric-value" id="cpu-temp">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Frequency</span>
                    <span class="metric-value" id="cpu-freq">--</span>
                </div>
                <div class="chart-container">
                    <canvas id="cpu-chart"></canvas>
                </div>
            </div>
            
            <!-- GPU Card -->
            <div class="card">
                <h2 class="card-title">🎮 GPU Performance</h2>
                <div class="metric">
                    <span class="metric-label">Usage</span>
                    <span class="metric-value" id="gpu-usage">--</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="gpu-progress"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Temperature</span>
                    <span class="metric-value" id="gpu-temp">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">VRAM</span>
                    <span class="metric-value" id="gpu-memory">--</span>
                </div>
                <div class="chart-container">
                    <canvas id="gpu-chart"></canvas>
                </div>
            </div>
            
            <!-- Memory Card -->
            <div class="card">
                <h2 class="card-title">💾 Memory</h2>
                <div class="metric">
                    <span class="metric-label">RAM Usage</span>
                    <span class="metric-value" id="ram-usage">--</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="ram-progress"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Used / Total</span>
                    <span class="metric-value" id="ram-details">--</span>
                </div>
                <div class="chart-container">
                    <canvas id="ram-chart"></canvas>
                </div>
            </div>
            
            <!-- AI Predictions Card -->
            <div class="card">
                <h2 class="card-title">🤖 AI Predictions</h2>
                <div class="metric">
                    <span class="metric-label">Performance Score</span>
                    <span class="metric-value" id="perf-score">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Thermal Risk</span>
                    <span class="metric-value" id="thermal-risk">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Profile</span>
                    <span class="metric-value" id="active-profile">Balanced</span>
                </div>
                <div class="controls">
                    <button onclick="optimizeNow()">Optimize Now</button>
                    <button onclick="toggleAutoTune()" id="auto-tune-btn">Enable Auto-Tune</button>
                </div>
            </div>
            
            <!-- System Stats Card -->
            <div class="card full-width">
                <h2 class="card-title">📊 System Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="avg-cpu">--</div>
                        <div class="stat-label">Avg CPU</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="avg-gpu">--</div>
                        <div class="stat-label">Avg GPU</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="max-temp">--</div>
                        <div class="stat-label">Max Temp</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="uptime">--</div>
                        <div class="stat-label">Uptime</div>
                    </div>
                </div>
            </div>
            
            <!-- Alerts Card -->
            <div class="card full-width">
                <h2 class="card-title">🔔 Recent Alerts</h2>
                <div id="alerts-container">
                    <div class="alert alert-info">System monitoring active</div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script>
        // Initialize charts
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { display: false }
                },
                x: {
                    display: false
                }
            },
            plugins: {
                legend: { display: false }
            }
        };
        
        const cpuChart = new Chart(document.getElementById('cpu-chart'), {
            type: 'line',
            data: {
                labels: Array(60).fill(''),
                datasets: [{
                    data: Array(60).fill(0),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: chartOptions
        });
        
        const gpuChart = new Chart(document.getElementById('gpu-chart'), {
            type: 'line',
            data: {
                labels: Array(60).fill(''),
                datasets: [{
                    data: Array(60).fill(0),
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: chartOptions
        });
        
        const ramChart = new Chart(document.getElementById('ram-chart'), {
            type: 'line',
            data: {
                labels: Array(60).fill(''),
                datasets: [{
                    data: Array(60).fill(0),
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: chartOptions
        });
        
        let autoTuneEnabled = false;
        
        // Update functions
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update CPU
                    document.getElementById('cpu-usage').textContent = data.cpu.usage.toFixed(1) + '%';
                    document.getElementById('cpu-temp').textContent = data.cpu.temperature.toFixed(0) + '°C';
                    document.getElementById('cpu-freq').textContent = data.cpu.frequency.toFixed(0) + ' MHz';
                    
                    const cpuProgress = document.getElementById('cpu-progress');
                    cpuProgress.style.width = data.cpu.usage + '%';
                    cpuProgress.className = 'progress-fill';
                    if (data.cpu.usage > 80) cpuProgress.classList.add('danger');
                    else if (data.cpu.usage > 60) cpuProgress.classList.add('warning');
                    
                    // Update GPU
                    document.getElementById('gpu-usage').textContent = data.gpu.usage.toFixed(1) + '%';
                    document.getElementById('gpu-temp').textContent = data.gpu.temperature.toFixed(0) + '°C';
                    document.getElementById('gpu-memory').textContent = data.gpu.memory.toFixed(1) + '%';
                    
                    const gpuProgress = document.getElementById('gpu-progress');
                    gpuProgress.style.width = data.gpu.usage + '%';
                    gpuProgress.className = 'progress-fill';
                    if (data.gpu.usage > 85) gpuProgress.classList.add('danger');
                    else if (data.gpu.usage > 70) gpuProgress.classList.add('warning');
                    
                    // Update RAM
                    document.getElementById('ram-usage').textContent = data.memory.usage.toFixed(1) + '%';
                    document.getElementById('ram-details').textContent = 
                        data.memory.used.toFixed(1) + ' / ' + data.memory.total.toFixed(1) + ' GB';
                    
                    const ramProgress = document.getElementById('ram-progress');
                    ramProgress.style.width = data.memory.usage + '%';
                    ramProgress.className = 'progress-fill';
                    if (data.memory.usage > 90) ramProgress.classList.add('danger');
                    else if (data.memory.usage > 75) ramProgress.classList.add('warning');
                    
                    // Update charts
                    cpuChart.data.datasets[0].data.push(data.cpu.usage);
                    cpuChart.data.datasets[0].data.shift();
                    cpuChart.update();
                    
                    gpuChart.data.datasets[0].data.push(data.gpu.usage);
                    gpuChart.data.datasets[0].data.shift();
                    gpuChart.update();
                    
                    ramChart.data.datasets[0].data.push(data.memory.usage);
                    ramChart.data.datasets[0].data.shift();
                    ramChart.update();
                });
        }
        
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('avg-cpu').textContent = data.avg_cpu.toFixed(1) + '%';
                    document.getElementById('avg-gpu').textContent = data.avg_gpu.toFixed(1) + '%';
                    document.getElementById('max-temp').textContent = data.max_temp.toFixed(0) + '°C';
                    document.getElementById('uptime').textContent = data.uptime;
                });
        }
        
        function updateAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('alerts-container');
                    container.innerHTML = '';
                    
                    data.alerts.forEach(alert => {
                        const div = document.createElement('div');
                        div.className = 'alert alert-' + alert.level;
                        div.textContent = alert.message;
                        container.appendChild(div);
                    });
                });
        }
        
        function updatePredictions() {
            fetch('/api/predictions')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('perf-score').textContent = data.performance.toFixed(0);
                    document.getElementById('thermal-risk').textContent = data.thermal_risk;
                    document.getElementById('active-profile').textContent = data.profile;
                });
        }
        
        function optimizeNow() {
            fetch('/api/optimize', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                });
        }
        
        function toggleAutoTune() {
            autoTuneEnabled = !autoTuneEnabled;
            fetch('/api/auto-tune', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: autoTuneEnabled })
            })
            .then(response => response.json())
            .then(data => {
                const btn = document.getElementById('auto-tune-btn');
                btn.textContent = autoTuneEnabled ? 'Disable Auto-Tune' : 'Enable Auto-Tune';
            });
        }
        
        // Update intervals
        setInterval(updateMetrics, 1000);
        setInterval(updateStats, 5000);
        setInterval(updateAlerts, 10000);
        setInterval(updatePredictions, 3000);
        
        // Initial updates
        updateMetrics();
        updateStats();
        updateAlerts();
        updatePredictions();
    </script>
</body>
</html>
'''

class SystemMonitor:
    """Simplified system monitor for web dashboard"""
    
    def get_metrics(self):
        """Get current system metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        
        # GPU
        gpu_data = {'usage': 0, 'memory': 0, 'temperature': 0, 'name': 'N/A'}
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_data = {
                        'usage': gpu.load * 100,
                        'memory': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature,
                        'name': gpu.name
                    }
            except:
                pass
        
        # Memory
        mem = psutil.virtual_memory()
        
        # Temperature (simplified)
        cpu_temp = 45.0  # Default
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if 'cpu' in entry.label.lower():
                            cpu_temp = entry.current
                            break
        except:
            pass
        
        return {
            'cpu': {
                'usage': cpu_percent,
                'temperature': cpu_temp,
                'frequency': cpu_freq.current if cpu_freq else 0
            },
            'gpu': gpu_data,
            'memory': {
                'usage': mem.percent,
                'total': mem.total / (1024**3),
                'used': mem.used / (1024**3)
            }
        }

# Initialize monitor
monitor = SystemMonitor()
start_time = datetime.now()

# Routes
@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current system metrics"""
    metrics = monitor.get_metrics()
    metrics_history.append(metrics)
    return jsonify(metrics)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    uptime = datetime.now() - start_time
    hours = int(uptime.total_seconds() // 3600)
    minutes = int((uptime.total_seconds() % 3600) // 60)
    
    # Calculate averages from history
    if metrics_history:
        avg_cpu = np.mean([m['cpu']['usage'] for m in metrics_history])
        avg_gpu = np.mean([m['gpu']['usage'] for m in metrics_history])
        max_temp = max([m['cpu']['temperature'] for m in metrics_history] + 
                      [m['gpu']['temperature'] for m in metrics_history])
    else:
        avg_cpu = avg_gpu = max_temp = 0
    
    return jsonify({
        'avg_cpu': avg_cpu,
        'avg_gpu': avg_gpu,
        'max_temp': max_temp,
        'uptime': f"{hours}h {minutes}m"
    })

@app.route('/api/predictions')
def get_predictions():
    """Get AI predictions (simulated)"""
    # Simulate predictions based on current metrics
    if metrics_history:
        current = metrics_history[-1]
        perf_score = (100 - current['cpu']['usage']) * 0.4 + \
                    (100 - current['gpu']['usage']) * 0.4 + \
                    (100 - current['memory']['usage']) * 0.2
        
        max_temp = max(current['cpu']['temperature'], current['gpu']['temperature'])
        if max_temp > 80:
            thermal_risk = 'High'
        elif max_temp > 70:
            thermal_risk = 'Medium'
        else:
            thermal_risk = 'Low'
    else:
        perf_score = 50
        thermal_risk = 'Unknown'
    
    return jsonify({
        'performance': perf_score,
        'thermal_risk': thermal_risk,
        'profile': 'Balanced'
    })

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    alert_list = []
    
    if metrics_history:
        current = metrics_history[-1]
        
        # Check for high temps
        if current['cpu']['temperature'] > 80:
            alert_list.append({
                'level': 'danger',
                'message': f"CPU temperature critical: {current['cpu']['temperature']:.0f}°C"
            })
        
        # Check for high usage
        if current['cpu']['usage'] > 90:
            alert_list.append({
                'level': 'warning',
                'message': f"CPU usage high: {current['cpu']['usage']:.1f}%"
            })
        
        if current['memory']['usage'] > 90:
            alert_list.append({
                'level': 'warning',
                'message': f"Memory usage critical: {current['memory']['usage']:.1f}%"
            })
    
    if not alert_list:
        alert_list.append({
            'level': 'info',
            'message': 'System operating normally'
        })
    
    return jsonify({'alerts': alert_list})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Trigger optimization"""
    # This would trigger actual optimization in the main app
    return jsonify({'success': True, 'message': 'Optimization triggered'})

@app.route('/api/auto-tune', methods=['POST'])
def toggle_auto_tune():
    """Toggle auto-tune"""
    enabled = request.json.get('enabled', False)
    return jsonify({'success': True, 'enabled': enabled})

def run_web_server(host='0.0.0.0', port=5000):
    """Run the web server"""
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║     SmartRig AI Tuner - Web Dashboard               ║
    ╠══════════════════════════════════════════════════════╣
    ║  Access the dashboard from any device:              ║
    ║                                                      ║
    ║  Local:    http://localhost:{port}                     ║
    ║  Network:  http://{get_local_ip()}:{port}              ║
    ║                                                      ║
    ║  Press Ctrl+C to stop the server                    ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=False)

def get_local_ip():
    """Get local IP address"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    # Check if Flask is installed
    try:
        import flask
    except ImportError:
        print("Flask not installed. Installing...")
        os.system(f"{sys.executable} -m pip install flask")
        import flask
    
    # Run the web server
    run_web_server()