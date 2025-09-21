# SmartRig AI Tuner Pro 🚀

Advanced AI-powered PC optimization tool with local machine learning - **Zero API keys required!**

## 🌟 Features

### Core Capabilities
- **Real-Time System Monitoring**: Track CPU, GPU, RAM usage with millisecond precision
- **Local AI/ML Models**: Train personalized optimization models on YOUR hardware patterns
- **Predictive Thermal Management**: AI predicts throttling 30-60 seconds before it happens
- **Automatic Game Detection**: Identifies running games and applies custom profiles
- **Safety-First Design**: Multi-layered protection prevents dangerous overclocks
- **Community Profiles**: Import/export optimization profiles (no cloud required)

### Advanced Features
- **Performance Score Prediction**: ML model rates your system's current efficiency
- **Anomaly Detection**: Identifies unusual system behavior that could indicate issues
- **Frame Time Analysis**: Focuses on consistency, not just raw FPS
- **Session-Based Learning**: AI improves recommendations based on your usage patterns
- **Thermal Risk Simulation**: Predicts temperature spikes before they occur
- **Bottleneck Analysis**: Identifies whether CPU or GPU is limiting performance

## 📋 Requirements

- **Python**: 3.8 or higher
- **OS**: Windows 10/11, Linux, macOS (limited GPU support on Mac)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for app + data

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone or download the script
# Navigate to the directory

# Install required packages
pip install psutil numpy pandas scikit-learn matplotlib

# Optional but recommended for GPU monitoring
pip install gputil

# For Windows users (better temperature monitoring)
pip install wmi
```

### 2. Run the Application

```bash
python smartrig_tuner.py
```

### 3. First-Time Setup

1. **Let it collect data**: Run for 5-10 minutes while using your PC normally
2. **Train the AI**: Click "Train AI Model" after ~100 data points
3. **Create profiles**: Run games and create custom profiles
4. **Enable Auto-Tune**: Let AI optimize in real-time

## 🎮 Usage Guide

### Dashboard Tab
- **Live Metrics**: View real-time CPU/GPU/RAM usage
- **Performance Graphs**: Visual trends over last 2 minutes
- **Quick Actions**:
  - `Enable Auto-Tune`: AI automatically adjusts settings
  - `Optimize Now`: Manual optimization trigger
  - `Train AI Model`: Update ML models with recent data

### Profiles Tab
- **Default Profiles**:
  - Power Saver: Minimize heat and power usage
  - Balanced: Everyday computing
  - Performance: Gaming and heavy workloads
  - Ultra: Maximum performance (monitor temps!)
- **Game Profiles**: Create custom profiles per game
- **Import/Export**: Share profiles with the community

### Analytics Tab
- View 24-hour statistics
- Session logs and performance history
- Export data for analysis

### AI Predictions Tab
- **Performance Score**: Current system efficiency (0-100)
- **Thermal Risk**: Low/Medium/High/Critical
- **Anomaly Detection**: Unusual behavior alerts
- **Recommendations**: AI-generated optimization tips

### Settings Tab
- **Safety Limits**: Set max temperatures
- **Data Management**: Clear logs, export data
- **Backup**: Save all settings and profiles

## 🧠 How the AI Works

### 1. Data Collection
The app continuously monitors:
- Hardware usage (CPU, GPU, RAM)
- Temperatures and frequencies
- Running processes and games
- Time patterns (hour, day of week)

### 2. Local ML Training
Uses **scikit-learn** models (no cloud/API needed):
- **Random Forest**: Predicts performance scores
- **Thermal Model**: Forecasts temperature spikes  
- **Isolation Forest**: Detects anomalies

### 3. Optimization Logic
```python
if thermal_risk == "High":
    apply_profile("Power Saver")
elif performance_score < 40:
    apply_profile("Performance")
elif anomaly_detected:
    notify_user("Check system")
```

## 🔧 Advanced Configuration

### Custom Game Profiles
1. Start your game
2. Play for 5+ minutes
3. Click "Create Game Profile"
4. Profile auto-applies when game launches

### Manual Model Training
```python
# Collect 24 hours of data
# Click "Train AI Model"
# Models saved to ~/.smartrig_tuner/ml_models/
```

### Export Performance Data
```python
# Analytics Tab > Export Data
# Creates CSV with all metrics
# Use for custom analysis in Excel/Python
```

## 🛡️ Safety Features

- **Temperature Limits**: Configurable max temps (default: 85°C CPU, 83°C GPU)
- **Gradual Adjustments**: No sudden aggressive changes
- **Rollback System**: Undo last optimization
- **Simulation Mode**: Test profiles without applying
- **Anomaly Alerts**: Warns of unusual behavior

## 📦 Creating Standalone Executable

### Windows (.exe)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "SmartRigTuner" smartrig_tuner.py
# Find exe in dist/ folder
```

### Linux (AppImage)
```bash
pip install pyinstaller
pyinstaller --onefile --name smartrig_tuner smartrig_tuner.py
# Or use AppImageKit for distribution
```

### macOS (.app)
```bash
pip install py2app
py2applet --make-setup smartrig_tuner.py
python setup.py py2app
```

## 🔍 Troubleshooting

### No GPU Data
- Install GPUtil: `pip install gputil`
- For NVIDIA: Ensure nvidia-smi is in PATH
- For AMD: Limited support (contributions welcome!)

### Temperature Reading Issues
- **Windows**: Install WMI: `pip install wmi`
- **Linux**: Install lm-sensors: `sudo apt install lm-sensors`
- **Default**: Falls back to safe 45°C if unavailable

### Model Training Fails
- Need minimum 100 data points
- Check disk space in ~/.smartrig_tuner/
- Clear old logs if database is corrupted

### High CPU Usage
- Reduce update frequency in monitoring_loop()
- Disable graph animations
- Close other monitoring tools

## 🎯 Performance Tips

1. **Initial Training**: Let it run for 24 hours for best AI accuracy
2. **Game Profiles**: Create separate profiles for each game
3. **Regular Retraining**: Retrain monthly as usage patterns change
4. **Profile Sharing**: Import community profiles for popular games
5. **Thermal Paste**: AI can't fix hardware - maintain your system!

## 📊 Understanding Metrics

### Performance Score (0-100)
- **80-100**: Excellent, no optimization needed
- **60-79**: Good, minor tweaks possible
- **40-59**: Moderate, optimization recommended
- **0-39**: Poor, significant bottleneck detected

### Thermal Risk Levels
- **Low**: <65°C, safe for 24/7 operation
- **Medium**: 65-75°C, normal gaming temps
- **High**: 75-85°C, consider better cooling
- **Critical**: >85°C, throttling imminent

## 🚀 Advanced Tweaks

### Modify ML Parameters
```python
# In MLPredictor class
self.performance_model = RandomForestRegressor(
    n_estimators=200,  # More trees = better accuracy
    max_depth=15,      # Deeper trees = complex patterns
    min_samples_split=5
)
```

### Add Custom Metrics
```python
# In SystemMonitor.get_cpu_info()
# Add cache misses, context switches, etc.
```

### Integrate with MSI Afterburner
```python
# Use subprocess to control Afterburner profiles
subprocess.run(["MSIAfterburner.exe", "-Profile1"])
```

## 🤝 Contributing

This is a fully functional tool, but contributions are welcome!

### Areas for Improvement
- AMD GPU support via ROCm
- Integration with more overclocking tools
- Reinforcement learning for profile optimization
- Web dashboard for remote monitoring
- Mobile app companion

## ⚖️ License

MIT License - Free to use, modify, and distribute

## ⚠️ Disclaimer

This tool provides suggestions based on data analysis. Users are responsible for:
- Understanding their hardware limits
- Monitoring temperatures during use
- Backing up data before overclocking
- Not exceeding manufacturer specifications

**The developers are not responsible for any hardware damage from improper use.**

## 🎉 Credits

Built with Python and love for the PC gaming community. Special thanks to:
- scikit-learn team for amazing ML tools
- psutil developers for system monitoring
- The overclocking community for inspiration

---

**Remember**: The best optimization is good cooling and clean hardware. This tool helps you get the most from what you have, but can't replace proper maintenance!

Happy Gaming! 🎮