#!/bin/bash

# Smart File Organizer - Build Script
# This script builds both the Python backend and Electron frontend

set -e  # Exit on any error

echo "🪄 Smart File Organizer - Build Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf dist/
rm -rf build/
rm -rf frontend/build/
print_success "Previous builds cleaned"

# Build Python Backend
print_status "Building Python backend..."
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run 'python -m venv venv' and install dependencies first"
    exit 1
fi

source venv/bin/activate
print_status "Activated virtual environment"

# Install/update Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
print_success "Python dependencies installed"

# Build Python backend with PyInstaller
print_status "Building Python backend with PyInstaller..."
pyinstaller SmartFileOrganizerBackend.spec
print_success "Python backend built successfully"

# Verify Python backend build
if [ ! -d "dist/SmartFileOrganizerBackend.app" ]; then
    print_error "Python backend build failed - SmartFileOrganizerBackend.app not found"
    exit 1
fi

print_success "Python backend verified: dist/SmartFileOrganizerBackend.app"

# Build Electron Frontend
print_status "Building Electron frontend..."

# Install/update Node.js dependencies
print_status "Installing Node.js dependencies..."
cd frontend
# Show npm output to surface potential errors during CI/local builds
npm install
print_success "Node.js dependencies installed"

# Build Electron app (directory)
print_status "Building Electron app (directory)..."
npm run package
print_success "Electron app built successfully"

# Build DMG installer
print_status "Building DMG installer..."
npx electron-builder --mac dmg
print_success "DMG installer built successfully"

cd ..

# Verify builds
print_status "Verifying builds..."

# Check Python backend
if [ -d "dist/SmartFileOrganizerBackend.app" ]; then
    print_success "✓ Python backend: dist/SmartFileOrganizerBackend.app"
else
    print_error "✗ Python backend build missing"
fi

# Check Electron app (robust detection)
APP_CANDIDATES=(frontend/build/mac-arm64/*.app)
APP_BUNDLE=""
if [ -e "${APP_CANDIDATES[0]}" ]; then
    APP_BUNDLE="${APP_CANDIDATES[0]}"
    print_success "✓ Electron app: ${APP_BUNDLE}"
else
    print_error "✗ Electron app build missing"
fi

# Check DMG installer (robust detection)
DMG_CANDIDATES=(frontend/build/*.dmg)
DMG_FILE=""
if [ -e "${DMG_CANDIDATES[0]}" ]; then
    DMG_FILE="${DMG_CANDIDATES[0]}"
    print_success "✓ DMG installer: ${DMG_FILE}"
    DMG_SIZE=$(du -h "${DMG_FILE}" | cut -f1)
    print_status "DMG size: $DMG_SIZE"
else
    print_error "✗ DMG installer missing"
fi

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "📦 Build artifacts:"
echo "   • Python backend: dist/SmartFileOrganizerBackend.app"
if [ -n "$APP_BUNDLE" ]; then echo "   • Electron app: $APP_BUNDLE"; else echo "   • Electron app: (missing)"; fi
if [ -n "$DMG_FILE" ]; then echo "   • DMG installer: $DMG_FILE"; else echo "   • DMG installer: (missing)"; fi
echo ""
echo "🚀 To test the app:"
echo "   • Open: frontend/build/mac-arm64/smart-file-organizer.app"
echo "   • Or install: frontend/build/smart-file-organizer-0.1.0-arm64.dmg"
echo ""
print_success "Build script completed!"
