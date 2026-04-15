#!/bin/zsh

# BGEM3 Service Fix Script
# Run with sudo to fix security and installation issues

echo "🔧 Fixing BGEM3 service..."

# Disable Gatekeeper temporarily
echo "🔓 Disabling Gatekeeper..."
sudo spctl --master-disable

# Code sign the Python binary if it exists
echo "🔏 Code signing Python binary..."
if [ -f "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/bin/python3" ]; then
    sudo codesign --force --deep --sign - "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/bin/python3"
    echo "✅ Python binary signed"
else
    echo "⚠️ Python binary not found at expected location"
fi

# Re-enable Gatekeeper
echo "🔒 Re-enabling Gatekeeper..."
sudo spctl --master-enable

# Install required Python packages
echo "📦 Installing required packages..."
uv pip install --system --break-system-packages fastapi uvicorn sentence-transformers

# Create proper virtual environment
echo "🔄 Creating virtual environment..."
uv venv bgem3-venv

# Start the server
echo "🚀 Starting BGEM3 server..."
./scripts/start_server_fixed.sh

echo "✅ BGEM3 service should now be running on http://10.230.57.109:8000"
echo "💡 Test with: curl -X POST http://10.230.57.109:8000/embed -H \"X-API-Key: m1macmini\" -H \"Content-Type: application/json\" -d '[\"hello world\"]'"