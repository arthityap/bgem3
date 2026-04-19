#!/bin/bash
set -e

ZT_IP="10.230.57.109"
LOG_DIR="logs"

echo "=== Stopping existing services ==="
pkill -f "uvicorn bgem3_embed" 2>/dev/null || true
pkill -f "uvicorn bgem3_rerank" 2>/dev/null || true
pkill -f "uvicorn bgem3_mcp" 2>/dev/null || true
sleep 2

echo "=== Starting bgem3_embed (port 8000) ==="
nohup uv run uvicorn bgem3_embed:app --host "$ZT_IP" --port 8000 > "$LOG_DIR/bgem3_embed.log" 2>&1 &
sleep 8

echo "=== Starting bgem3_rerank (port 8002) ==="
nohup uv run uvicorn bgem3_rerank:app --host "$ZT_IP" --port 8002 > "$LOG_DIR/bgem3_rerank.log" 2>&1 &
sleep 8

echo "=== Starting bgem3_mcp (port 8001) ==="
nohup uv run uvicorn "bgem3_mcp:mcp.http_app" --host "$ZT_IP" --port 8001 --factory > "$LOG_DIR/bgem3_mcp.log" 2>&1 &
sleep 10

echo "=== Verifying ==="
curl -s "http://$ZT_IP:8000/health" | grep -q "healthy" && echo "✓ embed OK" || echo "✗ embed FAILED"
curl -s "http://$ZT_IP:8002/health" | grep -q "healthy" && echo "✓ rerank OK" || echo "✗ rerank FAILED"
curl -s "http://$ZT_IP:8001/mcp" | grep -q "jsonrpc" && echo "✓ mcp OK" || echo "✗ mcp FAILED"

echo "=== All services started ==="