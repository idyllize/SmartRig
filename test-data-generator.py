#!/usr/bin/env python3
"""
SmartRig AI Tuner Pro - Test Data Generator
Generates realistic synthetic data for testing ML models without waiting for real data collection
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
import math

class SyntheticDataGenerator:
    """Generate realistic PC performance data"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or (Path.home() / ".smartrig_tuner" / "performance.db")
        self.games = [
            {"name": "Cyberpunk2077.exe", "cpu_load": 65, "gpu_load": 90, "ram_gb": 12},
            {"name": "CSGO.exe", "cpu_load": 45, "gpu_load": 70, "ram_gb": 4},
            {"name": "Minecraft.exe", "cpu_load": 35, "gpu_load": 40, "ram_gb": 6},
            {"name": "MSFS2020.exe", "cpu_load": 80, "gpu_load": 95, "ram_gb": 16},
            {"name": "Valorant.exe", "cpu_load": 40, "gpu_load": 60, "ram_gb": 4},
            {"name": "GTA5.exe", "cpu_load": 60, "gpu_load": 85, "ram_gb": 8},
            {"name": "RDR2.exe", "cpu_load": 70, "gpu_load": 92, "ram_gb": 10},
            {"name": "Fortnite.exe", "cpu_load": 50, "gpu_load": 75, "ram_gb": 6},
        ]
        
        self.profiles = ["Power Saver", "Balanced", "Performance", "Ultra"]
        
    def generate_usage_pattern(self, base_load, time_hour, add_noise=True):
        """Generate realistic usage pattern based on time of day"""
        # Time-based multiplier (lower at night, higher during day)
        time_multiplier = 0.7 + 0.3 * math.sin((time_hour - 6) * math.pi / 12)
        
        # Add realistic fluctuations
        if add_noise:
            noise = np.random.normal(0, 5)
            spike = np.random.choice([0, 0, 0, 0, 15, 25], p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
        else:
            noise = 0
            spike = 0
        
        value = base_load * time_multiplier + noise + spike
        return max(5, min(100, value))  # Clamp between 5-100
    
    def generate_temperature(self, load, ambient=22, cooling_quality=0.7):
        """Generate realistic temperature based on load"""
        # Base temperature calculation
        base_temp = ambient + (load * 0.6)
        
        # Cooling effectiveness (0.5 = poor, 1.0 = excellent)
        cooled_temp = base_temp * (2 - cooling_quality)
        
        # Add some thermal inertia/lag
        noise = np.random.normal(0, 2)
        
        return max(30, min(95, cooled_temp + noise))
    
    def generate_gaming_session(self, game, duration_minutes, start_time):
        """Generate data for a gaming session"""
        data_points = []
        current_time = start_time
        
        # Ramp-up phase (game starting)
        for i in range(min(5, duration_minutes)):
            cpu_usage = self.generate_usage_pattern(
                game["cpu_load"] * (0.3 + 0.7 * i / 5), 
                current_time.hour
            )
            gpu_usage = self.generate_usage_pattern(
                game["gpu_load"] * (0.2 + 0.8 * i / 5),
                current_time.hour
            )
            
            data_points.append(self.create_data_point(
                current_time, cpu_usage, gpu_usage,
                game["ram_gb"], game["name"]
            ))
            current_time += timedelta(minutes=1)
        
        # Main gaming phase
        remaining = duration_minutes - 5
        for i in range(max(0, remaining)):
            # Occasional loading screens (low GPU)
            if random.random() < 0.05:
                cpu_usage = self.generate_usage_pattern(80, current_time.hour)
                gpu_usage = self.generate_usage_pattern(20, current_time.hour)
            else:
                cpu_usage = self.generate_usage_pattern(game["cpu_load"], current_time.hour)
                gpu_usage = self.generate_usage_pattern(game["gpu_load"], current_time.hour)
            
            data_points.append(self.create_data_point(
                current_time, cpu_usage, gpu_usage,
                game["ram_gb"], game["name"]
            ))
            current_time += timedelta(minutes=1)
        
        return data_points
    
    def generate_idle_period(self, duration_minutes, start_time):
        """Generate data for idle/browsing period"""
        data_points = []
        current_time = start_time
        
        for _ in range(duration_minutes):
            # Random browsing/idle activities
            activity = random.choice(["idle", "browsing", "video", "music"])
            
            if activity == "idle":
                cpu_usage = self.generate_usage_pattern(5, current_time.hour)
                gpu_usage = self.generate_usage_pattern(2, current_time.hour)
                ram_gb = 3
            elif activity == "browsing":
                cpu_usage = self.generate_usage_pattern(15, current_time.hour)
                gpu_usage = self.generate_usage_pattern(10, current_time.hour)
                ram_gb = 4
            elif activity == "video":
                cpu_usage = self.generate_usage_pattern(25, current_time.hour)
                gpu_usage = self.generate_usage_pattern(30, current_time.hour)
                ram_gb = 5
            else:  # music
                cpu_usage = self.generate_usage_pattern(10, current_time.hour)
                gpu_usage = self.generate_usage_pattern(5, current_time.hour)
                ram_gb = 3
            
            data_points.append(self.create_data_point(
                current_time, cpu_usage, gpu_usage, ram_gb, None
            ))
            current_time += timedelta(minutes=1)
        
        return data_points
    
    def create_data_point(self, timestamp, cpu_usage, gpu_usage, ram_gb, game):
        """Create a single data point"""
        cpu_temp = self.generate_temperature(cpu_usage)
        gpu_temp = self.generate_temperature(gpu_usage, ambient=25)
        
        # CPU frequency scales with usage
        cpu_freq = 2000 + (cpu_usage / 100) * 3000 + np.random.normal(0, 100)
        
        # RAM usage
        total_ram = 32  # GB
        ram_usage = (ram_gb / total_ram) * 100 + np.random.normal(0, 3)
        
        # Performance score (inverse of usage)
        perf_score = (100 - cpu_usage) * 0.4 + (100 - gpu_usage) * 0.4 + (100 - ram_usage) * 0.2
        
        # Determine profile based on temps and usage
        max_temp = max(cpu_temp, gpu_temp)
        if max_temp > 80 or (cpu_usage > 85 and gpu_usage > 85):
            profile = "Ultra"
        elif cpu_usage > 60 or gpu_usage > 70:
            profile = "Performance"
        elif cpu_usage < 30 and gpu_usage < 30:
            profile = "Power Saver"
        else:
            profile = "Balanced"
        
        return {
            "timestamp": timestamp,
            "cpu_usage": cpu_usage,
            "cpu_freq": cpu_freq,
            "cpu_temp": cpu_temp,
            "gpu_usage": gpu_usage,
            "gpu_memory": gpu_usage * 0.8 + np.random.normal(0, 5),  # Correlate with usage
            "gpu_temp": gpu_temp,
            "ram_usage": ram_usage,
            "performance_score": perf_score,
            "max_temp": max_temp,
            "profile": profile,
            "game": game,
            "hour": timestamp.hour,
            "weekday": timestamp.weekday()
        }
    
    def generate_day_pattern(self, date):
        """Generate a realistic daily usage pattern"""
        all_data = []
        current_time = datetime.combine(date, datetime.min.time().replace(hour=8))
        
        # Morning routine (8 AM - 12 PM)
        all_data.extend(self.generate_idle_period(
            random.randint(180, 240), current_time
        ))
        current_time += timedelta(hours=4)
        
        # Lunch break gaming (12 PM - 1 PM)
        if random.random() < 0.3:  # 30% chance of lunch gaming
            game = random.choice(self.games[:4])  # Lighter games
            all_data.extend(self.generate_gaming_session(
                game, random.randint(30, 60), current_time
            ))
            current_time += timedelta(hours=1)
        else:
            all_data.extend(self.generate_idle_period(60, current_time))
            current_time += timedelta(hours=1)
        
        # Afternoon work (1 PM - 5 PM)
        all_data.extend(self.generate_idle_period(
            random.randint(180, 240), current_time
        ))
        current_time += timedelta(hours=4)
        
        # Evening gaming (5 PM - 11 PM)
        gaming_sessions = random.randint(1, 3)
        for _ in range(gaming_sessions):
            if random.random() < 0.7:  # 70% chance of gaming in evening
                game = random.choice(self.games)
                session_length = random.randint(45, 180)
                all_data.extend(self.generate_gaming_session(
                    game, session_length, current_time
                ))
                current_time += timedelta(minutes=session_length)
                
                # Break between games
                break_length = random.randint(10, 30)
                all_data.extend(self.generate_idle_period(break_length, current_time))
                current_time += timedelta(minutes=break_length)
            else:
                all_data.extend(self.generate_idle_period(60, current_time))
                current_time += timedelta(hours=1)
        
        return all_data
    
    def generate_anomalies(self, data_points, anomaly_rate=0.02):
        """Add realistic anomalies to the data"""
        num_anomalies = int(len(data_points) * anomaly_rate)
        anomaly_indices = random.sample(range(len(data_points)), num_anomalies)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice(["thermal", "usage_spike", "throttle"])
            
            if anomaly_type == "thermal":
                # Sudden temperature spike
                data_points[idx]["cpu_temp"] = min(95, data_points[idx]["cpu_temp"] + 20)
                data_points[idx]["gpu_temp"] = min(92, data_points[idx]["gpu_temp"] + 15)
            elif anomaly_type == "usage_spike":
                # Unusual usage pattern
                data_points[idx]["cpu_usage"] = min(100, data_points[idx]["cpu_usage"] + 30)
                data_points[idx]["gpu_usage"] = max(0, data_points[idx]["gpu_usage"] - 30)
            else:  # throttle
                # Thermal throttling simulation
                data_points[idx]["cpu_freq"] = data_points[idx]["cpu_freq"] * 0.6
                data_points[idx]["performance_score"] = data_points[idx]["performance_score"] * 0.5
        
        return data_points
    
    def save_to_database(self, data_points):
        """Save generated data to SQLite database"""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create table if not exists
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
        
        # Insert data
        for point in data_points:
            cursor.execute('''
                INSERT INTO performance_logs 
                (timestamp, cpu_usage, cpu_freq, cpu_temp, gpu_usage, gpu_memory, 
                 gpu_temp, ram_usage, performance_score, max_temp, profile, game, hour, weekday)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                point["timestamp"],
                point["cpu_usage"],
                point["cpu_freq"],
                point["cpu_temp"],
                point["gpu_usage"],
                point["gpu_memory"],
                point["gpu_temp"],
                point["ram_usage"],
                point["performance_score"],
                point["max_temp"],
                point["profile"],
                point["game"],
                point["hour"],
                point["weekday"]
            ))
        
        conn.commit()
        conn.close()
    
    def generate_dataset(self, days=7):
        """Generate a complete dataset for specified number of days"""
        all_data = []
        start_date = datetime.now() - timedelta(days=days)
        
        print(f"Generating {days} days of synthetic data...")
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            print(f"  Generating day {day + 1}/{days}: {current_date.date()}")
            
            day_data = self.generate_day_pattern(current_date.date())
            all_data.extend(day_data)
        
        # Add some anomalies
        all_data = self.generate_anomalies(all_data)
        
        print(f"Generated {len(all_data)} data points")
        return all_data
    
    def export_to_csv(self, data_points, filename="synthetic_data.csv"):
        """Export data to CSV for analysis"""
        df = pd.DataFrame(data_points)
        df.to_csv(filename, index=False)
        print(f"Exported to {filename}")
    
    def generate_sample_profiles(self):
        """Generate sample game profiles"""
        profiles_dir = Path.home() / ".smartrig_tuner" / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        for game in self.games[:3]:  # Generate for first 3 games
            profile = {
                "profile_info": {
                    "name": f"{game['name'].replace('.exe', '')} Optimized",
                    "game": game["name"],
                    "created": datetime.now().isoformat(),
                    "auto_generated": True
                },
                "performance_targets": {
                    "target_fps": 60 if game["gpu_load"] > 80 else 144,
                    "min_acceptable_fps": 45 if game["gpu_load"] > 80 else 60
                },
                "cpu_settings": {
                    "governor": "performance" if game["cpu_load"] > 60 else "balanced",
                    "boost_enabled": game["cpu_load"] > 50
                },
                "gpu_settings": {
                    "power_limit_percent": min(110, 80 + game["gpu_load"] // 3),
                    "fan_curve": "aggressive" if game["gpu_load"] > 85 else "balanced"
                },
                "memory_settings": {
                    "priority": "high" if game["ram_gb"] > 8 else "normal"
                }
            }
            
            profile_path = profiles_dir / f"{game['name'].replace('.exe', '')}_profile.json"
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            print(f"Created profile: {profile_path.name}")

def main():
    """Main function"""
    print("=" * 60)
    print("   SmartRig AI Tuner - Test Data Generator")
    print("=" * 60)
    print()
    
    generator = SyntheticDataGenerator()
    
    # Ask user for options
    print("Options:")
    print("1. Generate test data (recommended for first-time users)")
    print("2. Generate minimal data (for quick testing)")
    print("3. Generate extensive data (for ML model training)")
    print("4. Export existing data to CSV")
    print("5. Generate sample profiles")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == "1":
        # Standard test data - 3 days
        data = generator.generate_dataset(days=3)
        generator.save_to_database(data)
        print("\n✅ Test data generated successfully!")
        print("   You can now train the AI model in the application.")
        
    elif choice == "2":
        # Minimal data - 1 day
        data = generator.generate_dataset(days=1)
        generator.save_to_database(data)
        print("\n✅ Minimal test data generated!")
        
    elif choice == "3":
        # Extensive data - 14 days
        data = generator.generate_dataset(days=14)
        generator.save_to_database(data)
        generator.export_to_csv(data, "training_data.csv")
        print("\n✅ Extensive training data generated!")
        print("   Exported to training_data.csv for analysis")
        
    elif choice == "4":
        # Export existing data
        conn = sqlite3.connect(str(generator.db_path))
        df = pd.read_sql_query("SELECT * FROM performance_logs", conn)
        conn.close()
        
        if len(df) > 0:
            filename = f"exported_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"\n✅ Exported {len(df)} records to {filename}")
        else:
            print("\n❌ No data found in database")
        
    elif choice == "5":
        # Generate sample profiles
        generator.generate_sample_profiles()
        print("\n✅ Sample profiles generated!")
    
    else:
        print("\n❌ Invalid option")
        return
    
    print("\nData statistics:")
    conn = sqlite3.connect(str(generator.db_path))
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM performance_logs")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT game) FROM performance_logs WHERE game IS NOT NULL")
    unique_games = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(cpu_usage), AVG(gpu_usage), MAX(cpu_temp), MAX(gpu_temp) FROM performance_logs")
    stats = cursor.fetchone()
    
    conn.close()
    
    print(f"  Total records: {total_records}")
    print(f"  Unique games: {unique_games}")
    if stats[0]:
        print(f"  Avg CPU usage: {stats[0]:.1f}%")
        print(f"  Avg GPU usage: {stats[1]:.1f}%")
        print(f"  Max CPU temp: {stats[2]:.1f}°C")
        print(f"  Max GPU temp: {stats[3]:.1f}°C")

if __name__ == "__main__":
    main()