# BGEM3 Embedding Service

Self-hosted embedding generation service using the [BAAI/bge-m3](https://github.com/flagattribute/FlagEmbedding) model, optimized for Apple Silicon Macs (M1/M2/M3) via Metal Performance Shaders (MPS).

## Purpose

- Generate **dense embeddings** (1024-dimensional) for text retrieval
- Generate **hybrid embeddings** (dense + sparse) for hybrid search
- Exposed via REST API and MCP (Model Context Protocol) for AI agent integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BGEM3 Service                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌───────┐│
│  │bgem3_embed:8000   │ │bgem3_rerank:8002   │ │  MCP ││
│  │   (FastAPI)       │ │   (FastAPI)       │ │ 8001 ││
│  │ - /embed         │ │ - /rerank         │ │      ││
│  │ - /embed/hybrid  │ │ - /health        │ │embed ││
│  │ - /health       │ │ - /info         │ │hybrid││
│  │ - /info         │ └─────────────────────┘ │rerank│
│  └────────┬────────┘                           │      │
│           │         BGE-M3 + Reranker            │      │
│           │    (FlagEmbedding on MPS)           │      │
│           └──────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- ZeroTier installed and connected to network `10.230.57.x`
- **Python 3.11** (required — see `.python-version`)

### Install Dependencies

```bash
cd /Users/yapilymm/Downloads/projects/bgem3
uv sync
```

### Start Services

```bash
uv run python start.py
```

This script:
1. Kills any stale processes on ports 8000/8001/8002
2. Runs preflight checks
3. Starts bgem3_embed (port 8000)
4. Starts bgem3_rerank (port 8002)
5. Starts bgem3_mcp (port 8001)
6. Runs smoke tests
7. Exits — services stay running in background

### Test It

```bash
uv run python test_service.py
```

### Stop Services

```bash
lsof -ti :8000 | xargs kill
lsof -ti :8001 | xargs kill
lsof -ti :8002 | xargs kill
```

## Auto-Start on Boot (launchd)

For the Mac mini to start services automatically on boot:

```bash
launchctl load -w ~/Library/LaunchAgents/com.bgem3.server.plist
```

To unload:
```bash
launchctl unload ~/Library/LaunchAgents/com.bgem3.server.plist
```

## Files

| File | Purpose |
|------|----------|
| `start.py` | Main startup script (kills stale, runs preflight, starts all 3 servers) |
| `preflight.py` | Environment checks (Python 3.11, packages, ports, model cache) |
| `test_service.py` | Smoke tests for all services + MCP tools |
| `bgem3_embed.py` | FastAPI embedding service (port 8000) |
| `bgem3_rerank.py` | FastAPI reranker service (port 8002) |
| `bgem3_mcp.py` | FastMCP server with embed/hybrid/rerank tools (port 8001) |
| `.python-version` | Pins to Python 3.11 |
| `pyproject.toml` | Project dependencies |
| `.env` | API key configuration |

## API Reference

### REST API (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate dense embeddings (max 32 texts) |
| `/embed/hybrid` | POST | Generate dense + sparse embeddings (max 8 texts) |
| `/health` | GET | Health check with system metrics |
| `/info` | GET | Service and model information |

#### Authentication

Include Bearer token in header:
```
Authorization: Bearer m1macmini
```

#### Request Examples

**POST /embed**
```bash
curl -X POST http://10.230.57.109:8000/embed \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'

# Response
{
  "embeddings": [[0.0123, -0.0456, ...]],  // 1024-dim vector
  "count": 1,
  "dimensions": 1024,
  "model": "BAAI/bge-m3"
}
```

**POST /embed/hybrid**
```bash
curl -X POST http://10.230.57.109:8000/embed/hybrid \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'

# Response
{
  "dense_embeddings": [[0.0123, -0.0456, ...]],
  "sparse_embeddings": [{"token": 0.5}, {"hello": 0.3}, ...],
  "model": "BAAI/bge-m3"
}
```

### MCP Server (port 8001)

Connect to: `http://10.230.57.109:8001/mcp`

Tools:
| Tool | Description |
|------|-------------|
| `embed(texts: list[str])` | Generate dense embeddings |
| `embed_hybrid(texts: list[str])` | Generate dense + sparse embeddings |

## Preflight Checks

Run `uv run python preflight.py` to verify:

- Python == 3.11.x
- All required packages installed
- MPS available (Apple Silicon GPU)
- `.env` file exists with `EMBEDDING_API_KEY`
- BGE-M3 + Reranker weights cached locally
- bgem3_mcp.py has 3 MCP tools defined
- Ports 8000, 8001, 8002 are free

## Troubleshooting

### Server Won't Start

1. **Check ZeroTier**:
   ```bash
   ifconfig zt0
   ```

2. **Check ports**:
   ```bash
   lsof -i :8000
   lsof -i :8001
   ```

3. **Check logs**:
   ```bash
   tail -50 logs/rag_server.log
   tail -50 logs/mcp_server.log
   ```

### Health Check

```bash
curl http://10.230.57.109:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "service": "bgem3",
  "cpu_percent": 10.5,
  "memory_percent": 65.2,
  "mps_available": true
}
```

### Model Loading Issues

- First start downloads ~2.3GB model to `~/.cache/huggingface/`
- Reranker model is ~1.1GB (downloaded on first use)
- Check internet connection
- Ensure macOS on Apple Silicon

### Memory Issues

- Monitor via `/health` endpoint
- Reduce batch_size in `bgem3_embed.py` if needed

### MCP Tool Calls (Important!)

FastMCP requires a session. Call `initialize` first, then use the session ID:

```python
import httpx
import json

client = httpx.Client(follow_redirects=True)

# 1. Initialize session
init_resp = client.post(
    "http://10.230.57.109:8001/mcp/",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    },
    json={
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "client", "version": "1.0"},
        },
    },
)
session_id = init_resp.headers.get("mcp-session-id")

# 2. Call tools with session ID
tool_resp = client.post(
    "http://10.230.57.109:8001/mcp/",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-session-id": session_id,
    },
    json={
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {
            "name": "embed",
            "arguments": {"texts": ["your text here"]},
        },
    },
)

# 3. Parse response (starts with "event: message" then "data: {...}")
for line in tool_resp.text.splitlines():
    if line.startswith("data:"):
        result = json.loads(line[5:].strip())
        break
```

Key points:
- Must call `initialize` before any tool calls
- Session ID is in `mcp-session-id` header (not JSON body)
- Accept header must include both `application/json` and `text/event-stream`
- Response format is `event: message\ndata: {...}`