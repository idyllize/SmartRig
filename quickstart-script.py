#!/usr/bin/env python3
"""
SmartRig AI Tuner Pro - Quick Start Script
This script checks dependencies and helps set up the application
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print("Please download from: https://www.python.org/downloads/")
        return False
    
    print("✅ Python version OK")
    return True

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required = {
        'psutil': 'psutil',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }
    
    optional = {
        'GPUtil': 'gputil'
    }
    
    if platform.system() == 'Windows':
        optional['wmi'] = 'wmi'
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for import_name, install_name in required.items():
        if check_package(import_name):
            print(f"✅ {install_name} installed")
        else:
            print(f"❌ {install_name} missing")
            missing_required.append(install_name)
    
    # Check optional packages
    for import_name, install_name in optional.items():
        if check_package(import_name):
            print(f"✅ {install_name} installed (optional)")
        else:
            print(f"⚠️  {install_name} not installed (optional)")
            missing_optional.append(install_name)
    
    # Install missing required packages
    if missing_required:
        print(f"\n📥 Installing required packages: {', '.join(missing_required)}")
        response = input("Install now? (y/n): ")
        
        if response.lower() == 'y':
            for package in missing_required:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("✅ Required packages installed!")
        else:
            print("❌ Cannot run without required packages")
            return False
    
    # Ask about optional packages
    if missing_optional:
        print(f"\n📦 Optional packages for better features: {', '.join(missing_optional)}")
        response = input("Install optional packages? (y/n): ")
        
        if response.lower() == 'y':
            for package in missing_optional:
                try:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except:
                    print(f"⚠️  Failed to install {package} (not critical)")
    
    return True

def create_desktop_shortcut():
    """Create a desktop shortcut (Windows only)"""
    if platform.system() != 'Windows':
        return
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "SmartRig AI Tuner.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = 'smartrig_tuner.py'
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.IconLocation = sys.executable
        shortcut.save()
        
        print("✅ Desktop shortcut created!")
    except:
        print("ℹ️  Could not create desktop shortcut (not critical)")

def run_application():
    """Run the main application"""
    print("\n🚀 Starting SmartRig AI Tuner Pro...")
    print("-" * 50)
    
    # Check if main file exists
    if not Path("smartrig_tuner.py").exists():
        print("❌ smartrig_tuner.py not found!")
        print("Please ensure the main application file is in the same directory")
        return False
    
    try:
        # Import and run the main application
        import smartrig_tuner
        smartrig_tuner.main()
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        print("Please run this script again to install dependencies")
        return False
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    
    return True

def main():
    """Main setup and run function"""
    print("=" * 60)
    print("   SmartRig AI Tuner Pro - Quick Start")
    print("   No API keys required - All local processing!")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        return
    
    # Install dependencies
    if not install_dependencies():
        input("\nPress Enter to exit...")
        return
    
    # Optional: Create desktop shortcut
    if platform.system() == 'Windows':
        response = input("\n📌 Create desktop shortcut? (y/n): ")
        if response.lower() == 'y':
            create_desktop_shortcut()
    
    # Run the application
    print("\n" + "=" * 60)
    run_application()

if __name__ == "__main__":
    main()