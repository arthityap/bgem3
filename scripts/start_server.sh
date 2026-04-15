#!/bin/zsh
LOG_DIR="$HOME/Library/Logs/bgem3"
mkdir -p "$LOG_DIR"
cd "$(dirname "$0")/../../.."
ZT_IP=$(./scripts/get_zt_ip.sh)
echo "🚀 Starting on ZeroTier IP: $ZT_IP" | tee -a "$LOG_DIR/server.log"
uv run uvicorn rag_server:app --host "$ZT_IP" --port 8000 2>&1 | tee -a "$LOG_DIR/server.log"