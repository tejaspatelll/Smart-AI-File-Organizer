#!/bin/bash

# Smart File Organizer - Export Script
# This script exports the built app to a convenient location

set -e  # Exit on any error

echo "📦 Smart File Organizer - Export Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create export directory
EXPORT_DIR="SmartFileOrganizer-Export-$(date +%Y%m%d-%H%M%S)"
print_status "Creating export directory: $EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

# Copy Python backend
if [ -d "dist/SmartFileOrganizerBackend.app" ]; then
    print_status "Copying Python backend..."
    cp -r "dist/SmartFileOrganizerBackend.app" "$EXPORT_DIR/"
    print_success "Python backend copied"
else
    print_error "Python backend not found. Please run build.sh first"
    exit 1
fi

# Copy Electron app
if [ -d "frontend/build/mac-arm64/smart-file-organizer.app" ]; then
    print_status "Copying Electron app..."
    cp -r "frontend/build/mac-arm64/smart-file-organizer.app" "$EXPORT_DIR/"
    print_success "Electron app copied"
else
    print_error "Electron app not found. Please run build.sh first"
    exit 1
fi

# Copy DMG installer
if [ -f "frontend/build/smart-file-organizer-0.1.0-arm64.dmg" ]; then
    print_status "Copying DMG installer..."
    cp "frontend/build/smart-file-organizer-0.1.0-arm64.dmg" "$EXPORT_DIR/"
    print_success "DMG installer copied"
else
    print_error "DMG installer not found. Please run build.sh first"
    exit 1
fi

# Create README for export
cat > "$EXPORT_DIR/README.txt" << EOF
Smart File Organizer - Export Package
=====================================

This package contains the built Smart File Organizer application.

Contents:
- SmartFileOrganizerBackend.app: Python backend (standalone)
- smart-file-organizer.app: Complete Electron application
- smart-file-organizer-0.1.0-arm64.dmg: macOS installer

Installation:
1. For the complete app: Install smart-file-organizer-0.1.0-arm64.dmg
2. Or run directly: Open smart-file-organizer.app

Requirements:
- macOS 10.12+ (for APFS support)
- Internet connection (for AI features)

Notes:
- The app requires a Google Gemini API key for AI features
- Set GEMINI_API_KEY environment variable or create a .env file
- The Python backend can be run independently if needed

Build Date: $(date)
EOF

print_success "README created"

# Show export summary
echo ""
echo "🎉 Export completed successfully!"
echo ""
echo "📁 Export directory: $EXPORT_DIR"
echo ""
echo "📦 Contents:"
ls -la "$EXPORT_DIR"
echo ""
echo "🚀 Ready for distribution!"
echo "   • Install: $EXPORT_DIR/smart-file-organizer-0.1.0-arm64.dmg"
echo "   • Or run: $EXPORT_DIR/smart-file-organizer.app"
