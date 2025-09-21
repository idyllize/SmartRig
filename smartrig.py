#!/usr/bin/env python3
"""
SmartRig AI Tuner Pro - Advanced AI-Powered PC Optimization Tool
No API keys required - All processing done locally
Author: SmartRig Development Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import sqlite3
import threading
import subprocess
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import psutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Try to import GPUtil, handle gracefully if not available
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not found. GPU monitoring will be limited.")

# Constants
APP_NAME = "SmartRig AI Tuner Pro"
VERSION = "1.0.0"
DATA_DIR = Path.home() / ".smartrig_tuner"
DB_PATH = DATA_DIR / "performance.db"
MODEL_PATH = DATA_DIR / "ml_models"
PROFILES_PATH = DATA_DIR / "profiles"
LOGS_PATH = DATA_DIR / "logs"

# Create necessary directories
for path in [DATA_DIR, MODEL_PATH, PROFILES_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

class SystemMonitor:
    """Advanced system monitoring with predictive capabilities"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=120)  # 2 minutes of data
        self.gpu_history = deque(maxlen=120)
        self.ram_history = deque(maxlen=120)
        self.temp_history = deque(maxlen=120)
        self.running_games = []
        self.current_profile = "Balanced"
        
    def get_cpu_info(self) -> Dict:
        """Get detailed CPU information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_freq = psutil.cpu_freq()
            cpu_temp = self._get_cpu_temperature()
            
            return {
                'usage': np.mean(cpu_percent),
                'per_core': cpu_percent,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'frequency_max': cpu_freq.max if cpu_freq else 0,
                'temperature': cpu_temp,
                'core_count': psutil.cpu_count(logical=False),
                'thread_count': psutil.cpu_count(logical=True)
            }
        except Exception as e:
            return {'usage': 0, 'per_core': [], 'frequency': 0, 'temperature': 0}
    
    def get_gpu_info(self) -> Dict:
        """Get GPU information if available"""
        if not GPU_AVAILABLE:
            return {'usage': 0, 'memory': 0, 'temperature': 0, 'name': 'N/A'}
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                return {
                    'usage': gpu.load * 100,
                    'memory': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed
                }
        except:
            pass
        
        # Fallback for NVIDIA GPUs using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name',
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'usage': float(values[0]),
                    'memory_used': float(values[1]),
                    'memory_total': float(values[2]),
                    'temperature': float(values[3]),
                    'name': values[4],
                    'memory': (float(values[1]) / float(values[2])) * 100
                }
        except:
            pass
        
        return {'usage': 0, 'memory': 0, 'temperature': 0, 'name': 'N/A'}
    
    def get_memory_info(self) -> Dict:
        """Get RAM information"""
        mem = psutil.virtual_memory()
        return {
            'usage': mem.percent,
            'total': mem.total / (1024**3),  # GB
            'available': mem.available / (1024**3),
            'used': mem.used / (1024**3)
        }
    
    def detect_running_games(self) -> List[Dict]:
        """Detect currently running games"""
        games = []
        game_processes = ['game', 'steam', 'epic', 'origin', 'uplay', 'battle.net',
                         'minecraft', 'fortnite', 'valorant', 'csgo', 'dota2',
                         'leagueoflegends', 'overwatch', 'apex', 'pubg', 'gta',
                         'rocketleague', 'destiny2', 'warzone', 'fifa', 'nba2k']
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                name = proc.info['name'].lower()
                if any(game in name for game in game_processes) or name.endswith('.exe'):
                    # Check if it's using significant resources (likely a game)
                    if proc.info['cpu_percent'] > 10 or proc.info['memory_percent'] > 5:
                        games.append({
                            'name': proc.info['name'],
                            'pid': proc.info['pid'],
                            'cpu': proc.info['cpu_percent'],
                            'memory': proc.info['memory_percent']
                        })
            except:
                continue
        
        return games
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (platform-specific)"""
        try:
            # Try sensors (Linux)
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if 'cpu' in entry.label.lower() or 'core' in entry.label.lower():
                            return entry.current
            
            # Try Windows WMI
            if sys.platform == 'win32':
                try:
                    import wmi
                    c = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                    sensors = c.Sensor()
                    for sensor in sensors:
                        if sensor.SensorType == 'Temperature' and 'CPU' in sensor.Name:
                            return sensor.Value
                except:
                    pass
        except:
            pass
        
        return 45.0  # Default safe temperature
    
    def update_history(self):
        """Update monitoring history"""
        cpu_info = self.get_cpu_info()
        gpu_info = self.get_gpu_info()
        mem_info = self.get_memory_info()
        
        self.cpu_history.append(cpu_info['usage'])
        self.gpu_history.append(gpu_info['usage'])
        self.ram_history.append(mem_info['usage'])
        self.temp_history.append(max(cpu_info['temperature'], gpu_info['temperature']))
        
        return {
            'cpu': cpu_info,
            'gpu': gpu_info,
            'memory': mem_info,
            'timestamp': datetime.now()
        }

class MLPredictor:
    """Machine Learning model for performance prediction"""
    
    def __init__(self):
        self.performance_model = None
        self.thermal_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: List[Dict]) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        for point in data:
            features.append([
                point['cpu']['usage'],
                point['cpu']['frequency'],
                point['cpu']['temperature'],
                point['gpu']['usage'],
                point['gpu']['memory'],
                point['gpu']['temperature'],
                point['memory']['usage'],
                datetime.now().hour,  # Time of day
                datetime.now().weekday()  # Day of week
            ])
        return np.array(features)
    
    def train_models(self, training_data: pd.DataFrame):
        """Train ML models on collected data"""
        if len(training_data) < 100:
            return False
        
        try:
            # Prepare features
            feature_cols = ['cpu_usage', 'cpu_freq', 'cpu_temp', 'gpu_usage',
                          'gpu_memory', 'gpu_temp', 'ram_usage', 'hour', 'weekday']
            
            X = training_data[feature_cols].values
            
            # Train performance prediction model
            y_perf = training_data['performance_score'].values
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_perf, test_size=0.2, random_state=42
            )
            
            self.performance_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.performance_model.fit(X_train, y_train)
            
            # Train thermal prediction model
            y_thermal = training_data['max_temp'].values
            self.thermal_model = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42
            )
            self.thermal_model.fit(X_train, y_thermal)
            
            # Train anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.anomaly_detector.fit(X_scaled)
            
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def predict_performance(self, current_metrics: Dict) -> Dict:
        """Predict performance and thermal behavior"""
        if not self.is_trained:
            return {'performance': 50, 'thermal_risk': 'Low', 'anomaly': False}
        
        try:
            features = np.array([[
                current_metrics['cpu']['usage'],
                current_metrics['cpu']['frequency'],
                current_metrics['cpu']['temperature'],
                current_metrics['gpu']['usage'],
                current_metrics['gpu']['memory'],
                current_metrics['gpu']['temperature'],
                current_metrics['memory']['usage'],
                datetime.now().hour,
                datetime.now().weekday()
            ]])
            
            features_scaled = self.scaler.transform(features)
            
            perf_score = self.performance_model.predict(features_scaled)[0]
            thermal_pred = self.thermal_model.predict(features_scaled)[0]
            anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            
            # Determine thermal risk
            if thermal_pred > 85:
                thermal_risk = 'Critical'
            elif thermal_pred > 75:
                thermal_risk = 'High'
            elif thermal_pred > 65:
                thermal_risk = 'Medium'
            else:
                thermal_risk = 'Low'
            
            return {
                'performance': perf_score,
                'thermal_risk': thermal_risk,
                'predicted_temp': thermal_pred,
                'anomaly': anomaly
            }
            
        except Exception as e:
            return {'performance': 50, 'thermal_risk': 'Unknown', 'anomaly': False}
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            with open(MODEL_PATH / 'performance_model.pkl', 'wb') as f:
                pickle.dump(self.performance_model, f)
            with open(MODEL_PATH / 'thermal_model.pkl', 'wb') as f:
                pickle.dump(self.thermal_model, f)
            with open(MODEL_PATH / 'anomaly_detector.pkl', 'wb') as f:
                pickle.dump(self.anomaly_detector, f)
            with open(MODEL_PATH / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            with open(MODEL_PATH / 'performance_model.pkl', 'rb') as f:
                self.performance_model = pickle.load(f)
            with open(MODEL_PATH / 'thermal_model.pkl', 'rb') as f:
                self.thermal_model = pickle.load(f)
            with open(MODEL_PATH / 'anomaly_detector.pkl', 'rb') as f:
                self.anomaly_detector = pickle.load(f)
            with open(MODEL_PATH / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            return True
        except:
            return False

class ProfileManager:
    """Manages hardware profiles and optimizations"""
    
    def __init__(self):
        self.profiles = self._load_default_profiles()
        self.active_profile = 'Balanced'
        self.custom_profiles = self._load_custom_profiles()
        
    def _load_default_profiles(self) -> Dict:
        """Load default performance profiles"""
        return {
            'Power Saver': {
                'cpu_governor': 'powersave',
                'gpu_power_limit': 70,
                'fan_curve': 'quiet',
                'target_fps': 30,
                'description': 'Minimize power consumption and heat'
            },
            'Balanced': {
                'cpu_governor': 'balanced',
                'gpu_power_limit': 85,
                'fan_curve': 'balanced',
                'target_fps': 60,
                'description': 'Balance between performance and efficiency'
            },
            'Performance': {
                'cpu_governor': 'performance',
                'gpu_power_limit': 100,
                'fan_curve': 'performance',
                'target_fps': 144,
                'description': 'Maximum performance, higher temps'
            },
            'Ultra': {
                'cpu_governor': 'performance',
                'gpu_power_limit': 110,
                'fan_curve': 'aggressive',
                'target_fps': 240,
                'description': 'Extreme performance for competitive gaming'
            }
        }
    
    def _load_custom_profiles(self) -> Dict:
        """Load user-created profiles"""
        custom = {}
        for profile_file in PROFILES_PATH.glob('*.json'):
            try:
                with open(profile_file, 'r') as f:
                    profile_name = profile_file.stem
                    custom[profile_name] = json.load(f)
            except:
                continue
        return custom
    
    def create_game_profile(self, game_name: str, metrics: Dict) -> Dict:
        """Create optimized profile for specific game"""
        profile = {
            'name': f'{game_name}_optimized',
            'game': game_name,
            'created': datetime.now().isoformat(),
            'settings': {}
        }
        
        # Analyze metrics and create optimized settings
        avg_cpu = np.mean([m['cpu']['usage'] for m in metrics[-100:]])
        avg_gpu = np.mean([m['gpu']['usage'] for m in metrics[-100:]])
        max_temp = max([m['cpu']['temperature'] for m in metrics[-100:]])
        
        # Determine bottleneck
        if avg_cpu > avg_gpu + 20:
            profile['settings']['priority'] = 'cpu'
            profile['settings']['cpu_boost'] = True
            profile['settings']['gpu_power_limit'] = 90
        elif avg_gpu > avg_cpu + 20:
            profile['settings']['priority'] = 'gpu'
            profile['settings']['cpu_boost'] = False
            profile['settings']['gpu_power_limit'] = 110
        else:
            profile['settings']['priority'] = 'balanced'
            profile['settings']['cpu_boost'] = True
            profile['settings']['gpu_power_limit'] = 100
        
        # Thermal management
        if max_temp > 80:
            profile['settings']['fan_curve'] = 'aggressive'
            profile['settings']['undervolt'] = -50  # mV
        elif max_temp > 70:
            profile['settings']['fan_curve'] = 'performance'
            profile['settings']['undervolt'] = -30
        else:
            profile['settings']['fan_curve'] = 'balanced'
            profile['settings']['undervolt'] = 0
        
        return profile
    
    def apply_profile(self, profile_name: str) -> bool:
        """Apply a performance profile (simulated for safety)"""
        try:
            if profile_name in self.profiles:
                profile = self.profiles[profile_name]
            elif profile_name in self.custom_profiles:
                profile = self.custom_profiles[profile_name]
            else:
                return False
            
            # NOTE: Actual hardware control would require admin privileges
            # and platform-specific tools. This is a safe simulation.
            
            # Simulate applying settings
            print(f"Applying profile: {profile_name}")
            print(f"Settings: {profile}")
            
            # On Windows, could use:
            # - powercfg for power plans
            # - nvidia-smi for GPU settings
            # - Throttlestop API for CPU
            
            # On Linux, could use:
            # - cpufreq-set for CPU governor
            # - nvidia-settings for GPU
            
            self.active_profile = profile_name
            return True
            
        except Exception as e:
            print(f"Error applying profile: {e}")
            return False
    
    def export_profile(self, profile_name: str, filepath: str):
        """Export profile for sharing"""
        if profile_name in self.custom_profiles:
            profile = self.custom_profiles[profile_name]
            with open(filepath, 'w') as f:
                json.dump(profile, f, indent=2)
    
    def import_profile(self, filepath: str) -> bool:
        """Import a profile from file"""
        try:
            with open(filepath, 'r') as f:
                profile = json.load(f)
            
            name = Path(filepath).stem
            self.custom_profiles[name] = profile
            
            # Save to profiles directory
            with open(PROFILES_PATH / f"{name}.json", 'w') as f:
                json.dump(profile, f, indent=2)
            
            return True
        except:
            return False

class DatabaseManager:
    """Manages SQLite database for performance data"""
    
    def __init__(self):
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                cpu_freq REAL,
                cpu_temp REAL,
                gpu_usage REAL,
                gpu_memory REAL,
                gpu_temp REAL,
                ram_usage REAL,
                performance_score REAL,
                max_temp REAL,
                profile TEXT,
                game TEXT,
                hour INTEGER,
                weekday INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action TEXT,
                before_state TEXT,
                after_state TEXT,
                result TEXT
            )
        ''')
        
        self.conn.commit()
    
    def log_metrics(self, metrics: Dict, game: str = None):
        """Log performance metrics to database"""
        cursor = self.conn.cursor()
        
        # Calculate performance score (simplified)
        perf_score = (100 - metrics['cpu']['usage']) * 0.4 + \
                    (100 - metrics['gpu']['usage']) * 0.4 + \
                    (100 - metrics['memory']['usage']) * 0.2
        
        cursor.execute('''
            INSERT INTO performance_logs 
            (cpu_usage, cpu_freq, cpu_temp, gpu_usage, gpu_memory, gpu_temp,
             ram_usage, performance_score, max_temp, profile, game, hour, weekday)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['cpu']['usage'],
            metrics['cpu']['frequency'],
            metrics['cpu']['temperature'],
            metrics['gpu']['usage'],
            metrics['gpu']['memory'],
            metrics['gpu']['temperature'],
            metrics['memory']['usage'],
            perf_score,
            max(metrics['cpu']['temperature'], metrics['gpu']['temperature']),
            'Balanced',  # Current profile
            game,
            datetime.now().hour,
            datetime.now().weekday()
        ))
        
        self.conn.commit()
    
    def get_training_data(self, hours: int = 24) -> pd.DataFrame:
        """Get training data from database"""
        query = '''
            SELECT * FROM performance_logs 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        cursor = self.conn.cursor()
        
        # Last 24 hours stats
        cursor.execute('''
            SELECT 
                AVG(cpu_usage) as avg_cpu,
                AVG(gpu_usage) as avg_gpu,
                MAX(cpu_temp) as max_cpu_temp,
                MAX(gpu_temp) as max_gpu_temp,
                COUNT(*) as total_records
            FROM performance_logs
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        
        stats = cursor.fetchone()
        
        return {
            'avg_cpu': stats[0] or 0,
            'avg_gpu': stats[1] or 0,
            'max_cpu_temp': stats[2] or 0,
            'max_gpu_temp': stats[3] or 0,
            'total_records': stats[4] or 0
        }

class SmartRigGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.monitor = SystemMonitor()
        self.predictor = MLPredictor()
        self.profile_manager = ProfileManager()
        self.db_manager = DatabaseManager()
        
        # Load ML models if available
        self.predictor.load_models()
        
        # GUI state
        self.monitoring = False
        self.auto_tune = False
        self.current_metrics = {}
        self.metrics_history = deque(maxlen=1000)
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start GUI update loop
        self.update_display()
    
    def setup_styles(self):
        """Configure GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        bg_color = '#1e1e1e'
        fg_color = '#ffffff'
        accent_color = '#00a8ff'
        
        self.root.configure(bg=bg_color)
        
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'),
                       background=bg_color, foreground=accent_color)
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'),
                       background=bg_color, foreground=fg_color)
        style.configure('Info.TLabel', font=('Arial', 10),
                       background=bg_color, foreground=fg_color)
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text=APP_NAME, style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Dashboard tab
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.create_dashboard()
        
        # Profiles tab
        self.profiles_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.profiles_frame, text="Profiles")
        self.create_profiles_tab()
        
        # Analytics tab
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="Analytics")
        self.create_analytics_tab()
        
        # AI Predictions tab
        self.ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ai_frame, text="AI Predictions")
        self.create_ai_tab()
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self.create_settings_tab()
    
    def create_dashboard(self):
        """Create dashboard with real-time monitoring"""
        # Metrics display frame
        metrics_frame = ttk.LabelFrame(self.dashboard_frame, text="System Metrics")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # CPU metrics
        cpu_frame = ttk.Frame(metrics_frame)
        cpu_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(cpu_frame, text="CPU:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.cpu_label = ttk.Label(cpu_frame, text="0%", style='Info.TLabel')
        self.cpu_label.pack(side=tk.LEFT, padx=10)
        
        self.cpu_temp_label = ttk.Label(cpu_frame, text="0°C", style='Info.TLabel')
        self.cpu_temp_label.pack(side=tk.LEFT, padx=10)
        
        self.cpu_freq_label = ttk.Label(cpu_frame, text="0 MHz", style='Info.TLabel')
        self.cpu_freq_label.pack(side=tk.LEFT, padx=10)
        
        # GPU metrics
        gpu_frame = ttk.Frame(metrics_frame)
        gpu_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(gpu_frame, text="GPU:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.gpu_label = ttk.Label(gpu_frame, text="0%", style='Info.TLabel')
        self.gpu_label.pack(side=tk.LEFT, padx=10)
        
        self.gpu_temp_label = ttk.Label(gpu_frame, text="0°C", style='Info.TLabel')
        self.gpu_temp_label.pack(side=tk.LEFT, padx=10)
        
        self.gpu_mem_label = ttk.Label(gpu_frame, text="0% VRAM", style='Info.TLabel')
        self.gpu_mem_label.pack(side=tk.LEFT, padx=10)
        
        # RAM metrics
        ram_frame = ttk.Frame(metrics_frame)
        ram_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ram_frame, text="RAM:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.ram_label = ttk.Label(ram_frame, text="0%", style='Info.TLabel')
        self.ram_label.pack(side=tk.LEFT, padx=10)
        
        self.ram_used_label = ttk.Label(ram_frame, text="0 GB / 0 GB", style='Info.TLabel')
        self.ram_used_label.pack(side=tk.LEFT, padx=10)
        
        # Graphs frame
        graphs_frame = ttk.LabelFrame(self.dashboard_frame, text="Performance Graphs")
        graphs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 4), facecolor='#1e1e1e')
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#2e2e2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        self.ax1.set_title('CPU Usage', color='white')
        self.ax2.set_title('GPU Usage', color='white')
        self.ax3.set_title('Temperature', color='white')
        
        self.ax1.set_ylim(0, 100)
        self.ax2.set_ylim(0, 100)
        self.ax3.set_ylim(0, 100)
        
        self.canvas = FigureCanvasTkAgg(self.fig, graphs_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(self.dashboard_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_tune_btn = ttk.Button(control_frame, text="Enable Auto-Tune",
                                        command=self.toggle_auto_tune)
        self.auto_tune_btn.pack(side=tk.LEFT, padx=5)
        
        self.optimize_btn = ttk.Button(control_frame, text="Optimize Now",
                                       command=self.optimize_now)
        self.optimize_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(control_frame, text="Train AI Model",
                                    command=self.train_model)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_label = ttk.Label(control_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=5)
    
    def create_profiles_tab(self):
        """Create profiles management tab"""
        # Profile list
        list_frame = ttk.LabelFrame(self.profiles_frame, text="Available Profiles")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.profile_listbox = tk.Listbox(list_frame, bg='#2e2e2e', fg='white',
                                          selectmode=tk.SINGLE)
        self.profile_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Populate profiles
        for profile_name in self.profile_manager.profiles:
            self.profile_listbox.insert(tk.END, profile_name)
        for profile_name in self.profile_manager.custom_profiles:
            self.profile_listbox.insert(tk.END, f"[Custom] {profile_name}")
        
        # Profile details
        details_frame = ttk.LabelFrame(self.profiles_frame, text="Profile Details")
        details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.profile_details = tk.Text(details_frame, bg='#2e2e2e', fg='white',
                                       height=20, width=40)
        self.profile_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Profile controls
        controls_frame = ttk.Frame(self.profiles_frame)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Apply Profile",
                  command=self.apply_selected_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Create Game Profile",
                  command=self.create_game_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Import Profile",
                  command=self.import_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Profile",
                  command=self.export_profile).pack(side=tk.LEFT, padx=5)
        
        # Bind selection event
        self.profile_listbox.bind('<<ListboxSelect>>', self.on_profile_select)
    
    def create_analytics_tab(self):
        """Create analytics tab"""
        # Stats frame
        stats_frame = ttk.LabelFrame(self.analytics_frame, text="24-Hour Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, bg='#2e2e2e', fg='white',
                                 height=10, width=80)
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Session log frame
        log_frame = ttk.LabelFrame(self.analytics_frame, text="Session Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, bg='#2e2e2e', fg='white')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Refresh button
        ttk.Button(self.analytics_frame, text="Refresh Analytics",
                  command=self.refresh_analytics).pack(pady=5)
    
    def create_ai_tab(self):
        """Create AI predictions tab"""
        # Predictions frame
        pred_frame = ttk.LabelFrame(self.ai_frame, text="AI Predictions")
        pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Current predictions
        current_frame = ttk.Frame(pred_frame)
        current_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(current_frame, text="Performance Score:",
                 style='Heading.TLabel').pack(side=tk.LEFT)
        self.perf_score_label = ttk.Label(current_frame, text="--",
                                          style='Info.TLabel')
        self.perf_score_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(current_frame, text="Thermal Risk:",
                 style='Heading.TLabel').pack(side=tk.LEFT, padx=10)
        self.thermal_risk_label = ttk.Label(current_frame, text="--",
                                           style='Info.TLabel')
        self.thermal_risk_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(current_frame, text="Anomaly Detected:",
                 style='Heading.TLabel').pack(side=tk.LEFT, padx=10)
        self.anomaly_label = ttk.Label(current_frame, text="--",
                                      style='Info.TLabel')
        self.anomaly_label.pack(side=tk.LEFT, padx=10)
        
        # Recommendations
        rec_frame = ttk.LabelFrame(pred_frame, text="AI Recommendations")
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.recommendations_text = tk.Text(rec_frame, bg='#2e2e2e', fg='white',
                                           height=15, width=80)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Model info
        model_frame = ttk.Frame(self.ai_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_status_label = ttk.Label(model_frame,
                                           text="Model Status: Not Trained",
                                           style='Info.TLabel')
        self.model_status_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(model_frame, text="Retrain Model",
                  command=self.train_model).pack(side=tk.RIGHT, padx=5)
    
    def create_settings_tab(self):
        """Create settings tab"""
        # General settings
        general_frame = ttk.LabelFrame(self.settings_frame, text="General Settings")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_start_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(general_frame, text="Start with Windows",
                       variable=self.auto_start_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.minimize_tray_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(general_frame, text="Minimize to system tray",
                       variable=self.minimize_tray_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # Safety settings
        safety_frame = ttk.LabelFrame(self.settings_frame, text="Safety Settings")
        safety_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(safety_frame, text="Max CPU Temp (°C):",
                 style='Info.TLabel').pack(side=tk.LEFT, padx=10, pady=5)
        self.max_cpu_temp = tk.Scale(safety_frame, from_=60, to=95, orient=tk.HORIZONTAL,
                                     bg='#2e2e2e', fg='white')
        self.max_cpu_temp.set(85)
        self.max_cpu_temp.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(safety_frame, text="Max GPU Temp (°C):",
                 style='Info.TLabel').pack(side=tk.LEFT, padx=10, pady=5)
        self.max_gpu_temp = tk.Scale(safety_frame, from_=60, to=90, orient=tk.HORIZONTAL,
                                     bg='#2e2e2e', fg='white')
        self.max_gpu_temp.set(83)
        self.max_gpu_temp.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Data settings
        data_frame = ttk.LabelFrame(self.settings_frame, text="Data Management")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(data_frame, text="Clear Performance Logs",
                  command=self.clear_logs).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_frame, text="Export Data",
                  command=self.export_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_frame, text="Backup Settings",
                  command=self.backup_settings).pack(side=tk.LEFT, padx=5, pady=5)
        
        # About
        about_frame = ttk.LabelFrame(self.settings_frame, text="About")
        about_frame.pack(fill=tk.X, padx=5, pady=5)
        
        about_text = f"{APP_NAME} v{VERSION}\n\n"
        about_text += "Advanced AI-powered PC optimization tool\n"
        about_text += "No API keys required - All processing done locally\n\n"
        about_text += "Features:\n"
        about_text += "• Real-time system monitoring\n"
        about_text += "• ML-based performance prediction\n"
        about_text += "• Automatic game detection\n"
        about_text += "• Thermal throttle prevention\n"
        about_text += "• Custom profile creation"
        
        about_label = ttk.Label(about_frame, text=about_text, style='Info.TLabel')
        about_label.pack(padx=10, pady=10)
    
    def monitoring_loop(self):
        """Background monitoring thread"""
        while self.monitoring:
            try:
                # Update metrics
                metrics = self.monitor.update_history()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Log to database
                games = self.monitor.detect_running_games()
                game_name = games[0]['name'] if games else None
                self.db_manager.log_metrics(metrics, game_name)
                
                # Auto-tune if enabled
                if self.auto_tune:
                    self.auto_optimize(metrics)
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def update_display(self):
        """Update GUI displays"""
        if self.current_metrics:
            # Update metric labels
            cpu = self.current_metrics['cpu']
            gpu = self.current_metrics['gpu']
            memory = self.current_metrics['memory']
            
            self.cpu_label.config(text=f"{cpu['usage']:.1f}%")
            self.cpu_temp_label.config(text=f"{cpu['temperature']:.0f}°C")
            self.cpu_freq_label.config(text=f"{cpu['frequency']:.0f} MHz")
            
            self.gpu_label.config(text=f"{gpu['usage']:.1f}%")
            self.gpu_temp_label.config(text=f"{gpu['temperature']:.0f}°C")
            self.gpu_mem_label.config(text=f"{gpu['memory']:.1f}% VRAM")
            
            self.ram_label.config(text=f"{memory['usage']:.1f}%")
            self.ram_used_label.config(text=f"{memory['used']:.1f} GB / {memory['total']:.1f} GB")
            
            # Update graphs
            if len(self.monitor.cpu_history) > 1:
                self.ax1.clear()
                self.ax1.plot(list(self.monitor.cpu_history), 'c-', linewidth=2)
                self.ax1.set_title('CPU Usage (%)', color='white')
                self.ax1.set_ylim(0, 100)
                self.ax1.set_facecolor('#2e2e2e')
                self.ax1.tick_params(colors='white')
                
                self.ax2.clear()
                self.ax2.plot(list(self.monitor.gpu_history), 'g-', linewidth=2)
                self.ax2.set_title('GPU Usage (%)', color='white')
                self.ax2.set_ylim(0, 100)
                self.ax2.set_facecolor('#2e2e2e')
                self.ax2.tick_params(colors='white')
                
                self.ax3.clear()
                self.ax3.plot(list(self.monitor.temp_history), 'r-', linewidth=2)
                self.ax3.set_title('Max Temperature (°C)', color='white')
                self.ax3.set_ylim(0, 100)
                self.ax3.set_facecolor('#2e2e2e')
                self.ax3.tick_params(colors='white')
                
                self.canvas.draw()
            
            # Update AI predictions
            if self.predictor.is_trained:
                predictions = self.predictor.predict_performance(self.current_metrics)
                self.perf_score_label.config(text=f"{predictions['performance']:.1f}")
                self.thermal_risk_label.config(text=predictions['thermal_risk'])
                self.anomaly_label.config(text="Yes" if predictions['anomaly'] else "No")
                
                # Update recommendations
                self.update_recommendations(predictions)
        
        # Schedule next update
        self.root.after(1000, self.update_display)
    
    def update_recommendations(self, predictions):
        """Update AI recommendations"""
        recs = []
        
        if predictions['thermal_risk'] in ['High', 'Critical']:
            recs.append("⚠️ High thermal risk detected!")
            recs.append("• Consider increasing fan speed")
            recs.append("• Apply undervolt profile")
            recs.append("• Reduce power limits temporarily")
        
        if predictions['performance'] < 30:
            recs.append("📉 Low performance detected")
            recs.append("• Close background applications")
            recs.append("• Check for thermal throttling")
            recs.append("• Consider Performance profile")
        
        if predictions['anomaly']:
            recs.append("🔍 Anomaly detected in system behavior")
            recs.append("• Check for driver updates")
            recs.append("• Scan for malware")
            recs.append("• Review recent system changes")
        
        if not recs:
            recs.append("✅ System running optimally")
            recs.append("• No immediate actions needed")
        
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(1.0, "\n".join(recs))
    
    def toggle_auto_tune(self):
        """Toggle auto-tune feature"""
        self.auto_tune = not self.auto_tune
        if self.auto_tune:
            self.auto_tune_btn.config(text="Disable Auto-Tune")
            self.status_label.config(text="Auto-Tune Enabled")
        else:
            self.auto_tune_btn.config(text="Enable Auto-Tune")
            self.status_label.config(text="Auto-Tune Disabled")
    
    def auto_optimize(self, metrics):
        """Automatic optimization based on current metrics"""
        # Simple rule-based optimization
        cpu_usage = metrics['cpu']['usage']
        gpu_usage = metrics['gpu']['usage']
        max_temp = max(metrics['cpu']['temperature'], metrics['gpu']['temperature'])
        
        # Thermal protection
        if max_temp > self.max_cpu_temp.get():
            if self.profile_manager.active_profile != 'Power Saver':
                self.profile_manager.apply_profile('Power Saver')
                self.status_label.config(text="Thermal protection activated")
        
        # Performance boost
        elif cpu_usage > 80 or gpu_usage > 80:
            if self.profile_manager.active_profile != 'Performance':
                self.profile_manager.apply_profile('Performance')
                self.status_label.config(text="Performance mode activated")
        
        # Balanced mode
        elif cpu_usage < 50 and gpu_usage < 50:
            if self.profile_manager.active_profile != 'Balanced':
                self.profile_manager.apply_profile('Balanced')
                self.status_label.config(text="Balanced mode activated")
    
    def optimize_now(self):
        """Manual optimization trigger"""
        if not self.current_metrics:
            messagebox.showwarning("No Data", "No metrics available for optimization")
            return
        
        # Get predictions if model is trained
        if self.predictor.is_trained:
            predictions = self.predictor.predict_performance(self.current_metrics)
            
            # Apply optimizations based on predictions
            if predictions['thermal_risk'] in ['High', 'Critical']:
                self.profile_manager.apply_profile('Power Saver')
                messagebox.showinfo("Optimization", "Applied Power Saver profile for thermal management")
            elif predictions['performance'] < 40:
                self.profile_manager.apply_profile('Performance')
                messagebox.showinfo("Optimization", "Applied Performance profile for better performance")
            else:
                messagebox.showinfo("Optimization", "System is already optimized")
        else:
            # Simple optimization without ML
            self.auto_optimize(self.current_metrics)
            messagebox.showinfo("Optimization", "Applied rule-based optimization")
    
    def train_model(self):
        """Train ML model with collected data"""
        self.status_label.config(text="Training AI model...")
        
        # Get training data
        training_data = self.db_manager.get_training_data(hours=24)
        
        if len(training_data) < 100:
            messagebox.showwarning("Insufficient Data",
                                  f"Need at least 100 data points. Current: {len(training_data)}")
            return
        
        # Train in background thread
        def train():
            success = self.predictor.train_models(training_data)
            if success:
                self.model_status_label.config(text="Model Status: Trained")
                self.status_label.config(text="AI model trained successfully")
                messagebox.showinfo("Success", "AI model trained successfully!")
            else:
                self.status_label.config(text="Training failed")
                messagebox.showerror("Error", "Failed to train AI model")
        
        threading.Thread(target=train, daemon=True).start()
    
    def on_profile_select(self, event):
        """Handle profile selection"""
        selection = self.profile_listbox.curselection()
        if selection:
            profile_name = self.profile_listbox.get(selection[0])
            profile_name = profile_name.replace("[Custom] ", "")
            
            # Get profile details
            if profile_name in self.profile_manager.profiles:
                profile = self.profile_manager.profiles[profile_name]
            else:
                profile = self.profile_manager.custom_profiles.get(profile_name, {})
            
            # Display details
            self.profile_details.delete(1.0, tk.END)
            self.profile_details.insert(1.0, json.dumps(profile, indent=2))
    
    def apply_selected_profile(self):
        """Apply selected profile"""
        selection = self.profile_listbox.curselection()
        if selection:
            profile_name = self.profile_listbox.get(selection[0])
            profile_name = profile_name.replace("[Custom] ", "")
            
            if self.profile_manager.apply_profile(profile_name):
                self.status_label.config(text=f"Applied profile: {profile_name}")
                messagebox.showinfo("Success", f"Profile '{profile_name}' applied successfully")
            else:
                messagebox.showerror("Error", "Failed to apply profile")
    
    def create_game_profile(self):
        """Create optimized profile for current game"""
        games = self.monitor.detect_running_games()
        if not games:
            messagebox.showwarning("No Game", "No game detected. Please start a game first.")
            return
        
        game_name = games[0]['name']
        
        # Need sufficient metrics history
        if len(self.metrics_history) < 100:
            messagebox.showwarning("Insufficient Data",
                                  "Need more metrics data. Keep the game running for a few minutes.")
            return
        
        # Create profile
        profile = self.profile_manager.create_game_profile(game_name, list(self.metrics_history))
        
        # Save profile
        profile_path = PROFILES_PATH / f"{game_name}_optimized.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        # Add to custom profiles
        self.profile_manager.custom_profiles[f"{game_name}_optimized"] = profile
        
        # Update listbox
        self.profile_listbox.insert(tk.END, f"[Custom] {game_name}_optimized")
        
        messagebox.showinfo("Success", f"Created optimized profile for {game_name}")
    
    def import_profile(self):
        """Import profile from file"""
        filepath = filedialog.askopenfilename(
            title="Import Profile",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath and self.profile_manager.import_profile(filepath):
            # Update listbox
            profile_name = Path(filepath).stem
            self.profile_listbox.insert(tk.END, f"[Custom] {profile_name}")
            messagebox.showinfo("Success", "Profile imported successfully")
        else:
            messagebox.showerror("Error", "Failed to import profile")
    
    def export_profile(self):
        """Export selected profile"""
        selection = self.profile_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a profile to export")
            return
        
        profile_name = self.profile_listbox.get(selection[0])
        if not profile_name.startswith("[Custom]"):
            messagebox.showwarning("Export Error", "Can only export custom profiles")
            return
        
        profile_name = profile_name.replace("[Custom] ", "")
        
        filepath = filedialog.asksaveasfilename(
            title="Export Profile",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            self.profile_manager.export_profile(profile_name, filepath)
            messagebox.showinfo("Success", "Profile exported successfully")
    
    def refresh_analytics(self):
        """Refresh analytics display"""
        stats = self.db_manager.get_statistics()
        
        stats_text = f"=== 24-Hour Statistics ===\n\n"
        stats_text += f"Average CPU Usage: {stats['avg_cpu']:.1f}%\n"
        stats_text += f"Average GPU Usage: {stats['avg_gpu']:.1f}%\n"
        stats_text += f"Max CPU Temperature: {stats['max_cpu_temp']:.1f}°C\n"
        stats_text += f"Max GPU Temperature: {stats['max_gpu_temp']:.1f}°C\n"
        stats_text += f"Total Data Points: {stats['total_records']}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        
        # Update session log
        log_text = f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_text += f"Current profile: {self.profile_manager.active_profile}\n"
        log_text += f"Auto-tune: {'Enabled' if self.auto_tune else 'Disabled'}\n"
        log_text += f"AI Model: {'Trained' if self.predictor.is_trained else 'Not trained'}\n\n"
        
        games = self.monitor.detect_running_games()
        if games:
            log_text += "Running games:\n"
            for game in games:
                log_text += f"  - {game['name']} (CPU: {game['cpu']:.1f}%, RAM: {game['memory']:.1f}%)\n"
        
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(1.0, log_text)
    
    def clear_logs(self):
        """Clear performance logs"""
        if messagebox.askyesno("Confirm", "Clear all performance logs? This cannot be undone."):
            cursor = self.db_manager.conn.cursor()
            cursor.execute("DELETE FROM performance_logs")
            self.db_manager.conn.commit()
            self.status_label.config(text="Performance logs cleared")
            messagebox.showinfo("Success", "Performance logs cleared")
    
    def export_data(self):
        """Export performance data"""
        filepath = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            data = self.db_manager.get_training_data(hours=24*30)  # Last 30 days
            data.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Data exported to {filepath}")
    
    def backup_settings(self):
        """Backup all settings and profiles"""
        backup_dir = filedialog.askdirectory(title="Select Backup Directory")
        
        if backup_dir:
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(backup_dir) / f"smartrig_backup_{timestamp}"
            
            try:
                shutil.copytree(DATA_DIR, backup_path)
                messagebox.showinfo("Success", f"Backup created at {backup_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Backup failed: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        self.monitoring = False
        self.root.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()
    app = SmartRigGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    print(f"Starting {APP_NAME} v{VERSION}")
    print("No API keys required - All processing done locally")
    print("-" * 50)
    main()