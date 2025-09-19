# 🛠️ Build Instructions

This document provides comprehensive instructions for building and exporting the Smart File Organizer application.

## 📋 Prerequisites

### System Requirements

- **macOS**: 10.12+ (for APFS support in DMG)
- **Python**: 3.8+ (3.13+ recommended)
- **Node.js**: 14+ (16+ recommended)
- **Git**: For version control

### Required Tools

- **PyInstaller**: For Python backend packaging
- **Electron Builder**: For Electron app packaging
- **Virtual Environment**: Python virtual environment

## 🚀 Quick Build

### Automated Build (Recommended)

```bash
# Run the automated build script
./build.sh
```

This script will:

1. Clean previous builds
2. Build Python backend with PyInstaller
3. Build Electron frontend
4. Create DMG installer
5. Verify all builds

### Export Package

```bash
# Create a clean export package
./export.sh
```

## 🔧 Manual Build Process

### Step 1: Environment Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd smart-file-organizer
   ```

2. **Create Python virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Step 2: Build Python Backend

```bash
# Activate virtual environment
source venv/bin/activate

# Build with PyInstaller
pyinstaller SmartFileOrganizerBackend.spec
```

**Output**: `dist/SmartFileOrganizerBackend.app`

### Step 3: Build Electron Frontend

```bash
cd frontend

# Build directory version
npm run package

# Build DMG installer
npx electron-builder --mac dmg
```

**Outputs**:

- `frontend/build/mac-arm64/smart-file-organizer.app`
- `frontend/build/smart-file-organizer-0.1.0-arm64.dmg`

## 📦 Build Artifacts

### Python Backend

- **Location**: `dist/SmartFileOrganizerBackend.app`
- **Type**: macOS Application Bundle
- **Size**: ~200MB
- **Dependencies**: Self-contained (includes Python runtime)

### Electron Frontend

- **Location**: `frontend/build/mac-arm64/smart-file-organizer.app`
- **Type**: macOS Application Bundle
- **Size**: ~100MB
- **Dependencies**: Includes Python backend

### DMG Installer

- **Location**: `frontend/build/smart-file-organizer-0.1.0-arm64.dmg`
- **Type**: macOS Disk Image
- **Size**: ~170MB
- **Format**: APFS (supports macOS 10.12+)

## 🔍 Build Verification

### Check Python Backend

```bash
# Test Python backend
./dist/SmartFileOrganizerBackend.app/Contents/MacOS/SmartFileOrganizerBackend --help
```

### Check Electron App

```bash
# Open Electron app
open frontend/build/mac-arm64/smart-file-organizer.app
```

### Check DMG Installer

```bash
# Mount and verify DMG
hdiutil attach frontend/build/smart-file-organizer-0.1.0-arm64.dmg
ls /Volumes/smart-file-organizer-0.1.0-arm64/
hdiutil detach /Volumes/smart-file-organizer-0.1.0-arm64/
```

## 🐛 Troubleshooting

### Common Build Issues

#### PyInstaller Issues

```bash
# Clean PyInstaller cache
rm -rf build/ dist/
pyinstaller SmartFileOrganizerBackend.spec --clean
```

#### Electron Builder Issues

```bash
# Clean Electron cache
cd frontend
rm -rf build/ node_modules/
npm install
npm run package
```

#### Permission Issues

```bash
# Fix script permissions
chmod +x build.sh export.sh
```

#### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Build Failures

#### Python Backend Build Fails

- Check Python version: `python --version`
- Verify virtual environment: `which python`
- Check dependencies: `pip list`
- Review PyInstaller spec file

#### Electron Build Fails

- Check Node.js version: `node --version`
- Verify npm dependencies: `npm list`
- Check electron-builder configuration
- Review package.json build settings

#### DMG Creation Fails

- Ensure macOS 10.12+ for APFS support
- Check available disk space
- Verify code signing settings (optional)

## 📊 Build Configuration

### PyInstaller Configuration

- **Spec File**: `SmartFileOrganizerBackend.spec`
- **Entry Point**: `backend/cli.py`
- **Output**: macOS app bundle
- **Console**: Disabled (windowed app)

### Electron Builder Configuration

- **Config**: `frontend/package.json` (build field)
- **Platform**: macOS ARM64
- **Target**: DMG installer
- **Backend**: Included as extra resource

## 🚀 Distribution

### For End Users

1. **DMG Installer**: `smart-file-organizer-0.1.0-arm64.dmg`

   - Double-click to mount
   - Drag app to Applications folder
   - Eject DMG when done

2. **Direct App**: `smart-file-organizer.app`
   - Right-click and "Open" (first time)
   - Or run from terminal: `open smart-file-organizer.app`

### For Developers

1. **Python Backend**: `SmartFileOrganizerBackend.app`
   - Standalone Python application
   - Can be run independently
   - Useful for CLI operations

## 🔧 Advanced Configuration

### Custom Build Settings

#### PyInstaller Options

Edit `SmartFileOrganizerBackend.spec`:

```python
# Add custom hidden imports
hiddenimports=['custom.module']

# Exclude unnecessary modules
excludes=['unused.module']

# Add custom data files
datas=[('custom/data', 'custom/data')]
```

#### Electron Builder Options

Edit `frontend/package.json`:

```json
{
  "build": {
    "mac": {
      "category": "public.app-category.utilities",
      "target": "dmg",
      "icon": "assets/icon.icns"
    }
  }
}
```

### Code Signing (Optional)

```bash
# Set code signing identity
export CSC_NAME="Developer ID Application: Your Name"
npx electron-builder --mac dmg
```

## 📈 Performance Optimization

### Build Size Reduction

- Use `--onefile` for PyInstaller (single executable)
- Exclude unnecessary modules
- Compress assets
- Use UPX compression (if available)

### Build Speed

- Use build cache
- Parallel builds where possible
- Skip unnecessary steps in development

## 🎯 Next Steps

After successful build:

1. Test the application thoroughly
2. Create installer documentation
3. Set up automated builds (CI/CD)
4. Consider code signing for distribution
5. Test on different macOS versions

---

**Need Help?** Check the main README.md or create an issue in the repository.
