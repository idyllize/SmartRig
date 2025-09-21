#!/usr/bin/env python3
"""
SmartRig AI Tuner Pro - Build Script
Creates standalone executables for Windows, Linux, and macOS
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path
import json

# Build configuration
APP_NAME = "SmartRig_AI_Tuner"
VERSION = "1.0.0"
MAIN_SCRIPT = "smartrig_tuner.py"

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True

def create_spec_file():
    """Create a custom PyInstaller spec file"""
    spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.json', '.'),
        ('requirements.txt', '.'),
        ('*.json', 'profiles'),
    ],
    hiddenimports=[
        'sklearn.ensemble',
        'sklearn.preprocessing',
        'sklearn.model_selection',
        'sklearn.ensemble._forest',
        'sklearn.ensemble._iforest',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'matplotlib.backends.backend_tkagg',
        'psutil._psutil_windows' if platform.system() == 'Windows' else '',
        'psutil._psutil_linux' if platform.system() == 'Linux' else '',
        'psutil._psutil_osx' if platform.system() == 'Darwin' else '',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'pytest-cov',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{APP_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console={str(platform.system() != 'Windows').lower()},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='smartrig.ico' if os.path.exists('smartrig.ico') else None,
    version='version_info.txt' if os.path.exists('version_info.txt') else None,
)

# For macOS, create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='{APP_NAME}.app',
        icon='smartrig.icns' if os.path.exists('smartrig.icns') else None,
        bundle_identifier='com.smartrig.tuner',
        info_plist={{
            'CFBundleName': '{APP_NAME}',
            'CFBundleDisplayName': 'SmartRig AI Tuner Pro',
            'CFBundleVersion': '{VERSION}',
            'CFBundleShortVersionString': '{VERSION}',
            'NSHighResolutionCapable': 'True',
            'NSRequiresAquaSystemAppearance': 'False',
        }},
    )
'''
    
    with open(f"{APP_NAME}.spec", "w") as f:
        f.write(spec_content)
    
    print(f"✅ Created {APP_NAME}.spec")

def create_version_info():
    """Create version info file for Windows"""
    if platform.system() != "Windows":
        return
    
    version_info = f'''
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({VERSION.replace(".", ", ")}, 0),
    prodvers=({VERSION.replace(".", ", ")}, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'SmartRig Development Team'),
        StringStruct(u'FileDescription', u'SmartRig AI Tuner Pro - AI-powered PC optimization'),
        StringStruct(u'FileVersion', u'{VERSION}'),
        StringStruct(u'InternalName', u'{APP_NAME}'),
        StringStruct(u'LegalCopyright', u'Copyright (c) 2025 SmartRig Team'),
        StringStruct(u'OriginalFilename', u'{APP_NAME}.exe'),
        StringStruct(u'ProductName', u'SmartRig AI Tuner Pro'),
        StringStruct(u'ProductVersion', u'{VERSION}')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    with open("version_info.txt", "w") as f:
        f.write(version_info)
    
    print("✅ Created version_info.txt")

def build_executable():
    """Build the executable using PyInstaller"""
    system = platform.system()
    
    print(f"\n🔨 Building for {system}...")
    
    # Build command
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
    ]
    
    if os.path.exists(f"{APP_NAME}.spec"):
        cmd.append(f"{APP_NAME}.spec")
    else:
        # Fallback to basic build
        cmd.extend([
            "--onefile",
            "--name", APP_NAME,
        ])
        
        if system == "Windows":
            cmd.append("--windowed")
            if os.path.exists("smartrig.ico"):
                cmd.extend(["--icon", "smartrig.ico"])
        
        cmd.append(MAIN_SCRIPT)
    
    # Run PyInstaller
    try:
        subprocess.check_call(cmd)
        print(f"✅ Build successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

def create_installer_script():
    """Create platform-specific installer script"""
    system = platform.system()
    
    if system == "Windows":
        # Create NSIS installer script
        nsis_script = f'''
!include "MUI2.nsh"

Name "SmartRig AI Tuner Pro"
OutFile "SmartRig_Setup_{VERSION}.exe"
InstallDir "$PROGRAMFILES\\SmartRig AI Tuner"
RequestExecutionLevel admin

!define MUI_ICON "smartrig.ico"
!define MUI_UNICON "smartrig.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath $INSTDIR
    File "dist\\{APP_NAME}.exe"
    File "config.json"
    File "README.md"
    
    CreateDirectory "$SMPROGRAMS\\SmartRig AI Tuner"
    CreateShortcut "$SMPROGRAMS\\SmartRig AI Tuner\\SmartRig AI Tuner.lnk" "$INSTDIR\\{APP_NAME}.exe"
    CreateShortcut "$DESKTOP\\SmartRig AI Tuner.lnk" "$INSTDIR\\{APP_NAME}.exe"
    
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\SmartRigTuner" \\
                     "DisplayName" "SmartRig AI Tuner Pro"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\SmartRigTuner" \\
                     "UninstallString" "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\{APP_NAME}.exe"
    Delete "$INSTDIR\\config.json"
    Delete "$INSTDIR\\README.md"
    Delete "$INSTDIR\\Uninstall.exe"
    
    RMDir "$INSTDIR"
    
    Delete "$SMPROGRAMS\\SmartRig AI Tuner\\*.lnk"
    RMDir "$SMPROGRAMS\\SmartRig AI Tuner"
    Delete "$DESKTOP\\SmartRig AI Tuner.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\SmartRigTuner"
SectionEnd
'''
        
        with open("installer.nsi", "w") as f:
            f.write(nsis_script)
        
        print("✅ Created installer.nsi (requires NSIS to build)")
        
    elif system == "Linux":
        # Create AppImage recipe
        appimage_script = f'''#!/bin/bash
# SmartRig AI Tuner AppImage Builder

APP={APP_NAME}
VERSION={VERSION}

# Create AppDir structure
mkdir -p $APP.AppDir/usr/bin
mkdir -p $APP.AppDir/usr/share/applications
mkdir -p $APP.AppDir/usr/share/icons/hicolor/256x256/apps

# Copy executable
cp dist/$APP $APP.AppDir/usr/bin/

# Create desktop entry
cat > $APP.AppDir/usr/share/applications/$APP.desktop << EOF
[Desktop Entry]
Type=Application
Name=SmartRig AI Tuner Pro
Exec=$APP
Icon=$APP
Categories=System;Monitor;
EOF

# Copy icon (create a simple one if doesn't exist)
if [ ! -f smartrig.png ]; then
    # Create a simple icon using ImageMagick if available
    convert -size 256x256 xc:blue -fill white -gravity center \\
            -pointsize 72 -annotate +0+0 'SR' smartrig.png 2>/dev/null || \\
    echo "No icon found"
fi
cp smartrig.png $APP.AppDir/usr/share/icons/hicolor/256x256/apps/$APP.png 2>/dev/null

# Create AppRun script
cat > $APP.AppDir/AppRun << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${{SELF%/*}}
export PATH="${{HERE}}/usr/bin:${{PATH}}"
exec "${{HERE}}/usr/bin/{APP_NAME}" "$@"
EOF

chmod +x $APP.AppDir/AppRun

# Download appimagetool if not present
if [ ! -f appimagetool-x86_64.AppImage ]; then
    wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x appimagetool-x86_64.AppImage
fi

# Build AppImage
./appimagetool-x86_64.AppImage $APP.AppDir $APP-$VERSION.AppImage

echo "✅ AppImage created: $APP-$VERSION.AppImage"
'''
        
        with open("build_appimage.sh", "w") as f:
            f.write(appimage_script)
        
        os.chmod("build_appimage.sh", 0o755)
        print("✅ Created build_appimage.sh")
        
    elif system == "Darwin":
        # Create DMG builder script
        dmg_script = f'''#!/bin/bash
# SmartRig AI Tuner DMG Builder

APP={APP_NAME}
VERSION={VERSION}
DMG_NAME="SmartRig_AI_Tuner_{VERSION}.dmg"

# Create a temporary directory for DMG contents
mkdir -p dmg_contents
cp -r "dist/$APP.app" dmg_contents/
ln -s /Applications dmg_contents/Applications

# Create DMG
hdiutil create -volname "SmartRig AI Tuner Pro" \\
               -srcfolder dmg_contents \\
               -ov -format UDZO \\
               "$DMG_NAME"

# Clean up
rm -rf dmg_contents

echo "✅ DMG created: $DMG_NAME"
'''
        
        with open("build_dmg.sh", "w") as f:
            f.write(dmg_script)
        
        os.chmod("build_dmg.sh", 0o755)
        print("✅ Created build_dmg.sh")

def package_distribution():
    """Create distribution package with all necessary files"""
    dist_dir = Path(f"SmartRig_AI_Tuner_{VERSION}_{platform.system()}")
    
    # Create distribution directory
    dist_dir.mkdir(exist_ok=True)
    
    # Copy executable
    exe_name = f"{APP_NAME}.exe" if platform.system() == "Windows" else APP_NAME
    exe_path = Path("dist") / exe_name
    
    if exe_path.exists():
        shutil.copy2(exe_path, dist_dir)
    
    # Copy supporting files
    files_to_copy = [
        "config.json",
        "requirements.txt",
        "README.md",
        "cyberpunk2077_profile.json",
        "launch_smartrig.bat" if platform.system() == "Windows" else "launch_smartrig.sh",
    ]
    
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, dist_dir)
    
    # Create profiles directory
    (dist_dir / "profiles").mkdir(exist_ok=True)
    
    # Create archive
    archive_name = f"SmartRig_AI_Tuner_{VERSION}_{platform.system()}"
    shutil.make_archive(archive_name, 'zip', dist_dir)
    
    print(f"✅ Created distribution: {archive_name}.zip")
    
    # Clean up
    shutil.rmtree(dist_dir)

def main():
    """Main build process"""
    print("=" * 60)
    print(f"   SmartRig AI Tuner Pro - Build Script")
    print(f"   Version {VERSION} for {platform.system()}")
    print("=" * 60)
    
    # Check PyInstaller
    if not check_pyinstaller():
        print("❌ Failed to install PyInstaller")
        return 1
    
    # Check if main script exists
    if not Path(MAIN_SCRIPT).exists():
        print(f"❌ {MAIN_SCRIPT} not found!")
        return 1
    
    # Create spec file
    create_spec_file()
    
    # Create version info for Windows
    if platform.system() == "Windows":
        create_version_info()
    
    # Build executable
    if not build_executable():
        return 1
    
    # Create installer scripts
    create_installer_script()
    
    # Package distribution
    package_distribution()
    
    print("\n" + "=" * 60)
    print("✅ Build complete!")
    print(f"   Executable: dist/{APP_NAME}")
    print(f"   Distribution: SmartRig_AI_Tuner_{VERSION}_{platform.system()}.zip")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())