#!/bin/zsh
LOG_DIR="$(dirname "$0")/../logs"
mkdir -p "$LOG_DIR"
cd "$(dirname "$0")/.."
ZT_IP=$(./scripts/get_zt_ip.sh)
echo "🔌 Starting MCP server on ZeroTier IP: $ZT_IP" | tee -a "$LOG_DIR/mcp_server.log"
uv run python mcp_server.py 2>&1 | tee -a "$LOG_DIR/mcp_server.log"
