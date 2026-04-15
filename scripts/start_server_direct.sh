#!/bin/zsh
LOG_DIR="$HOME/Library/Logs/bgem3"
mkdir -p "$LOG_DIR"
cd "$(dirname "$0")/../../.."
ZT_IP=$(./scripts/get_zt_ip.sh)
echo "🚀 Starting on ZeroTier IP: $ZT_IP" | tee -a "$LOG_DIR/server.log"
.venv/bin/python3 -c "
import sys
sys.path.append('$(dirname "$0")/../../..')
from rag_server import app
import uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='$ZT_IP', port=8000)
" 2>&1 | tee -a "$LOG_DIR/server.log"