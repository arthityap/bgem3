# BGEM3 Agent Guide

This document provides essential information for AI agents working in the BGEM3 repository.

## CRITICAL CONVENTIONS - NEVER FORGET

> **ALWAYS use `uv`** for all Python commands — never `pip`, never bare `python`
> 
> **ALWAYS use IP `10.230.57.109`** — never `0.0.0.0`, never `127.0.0.1`
> 
> **ALWAYS use Authorization Bearer token** — not `X-API-Key` header

## Project Overview

BGEM3 provides embedding generation via BAAI/bge-m3 and reranking via bge-reranker-v2-m3 through a FastMCP interface. Designed for Apple Silicon Macs (M1/M2/M3) using MPS backend.

## Essential Commands

### Quick Start (recommended)

```bash
cd /Users/yapilymm/Downloads/projects/bgem3
./restart.sh
```

### Full Test

```bash
uv run test_service.py
```

Expected: `All 11 tests passed.`

### Preflight Checks

```bash
uv run preflight.py
```

Validates: Python 3.11, packages, GPU, cached weights, ports, `.env` has API key.

### Manual Service Start

If you need to start services individually:

```bash
# embed (port 8000)
uv run uvicorn bgem3_embed:app --host 10.230.57.109 --port 8000 &

# rerank (port 8002)
uv run uvicorn bgem3_rerank:app --host 10.230.57.109 --port 8002 &

# MCP (port 8001) - MUST use --factory with mcp.http_app
uv run uvicorn "bgem3_mcp:mcp.http_app" --host 10.230.57.109 --port 8001 --factory &
```

### Stop Services

```bash
pkill -f "uvicorn bgem3_embed"
pkill -f "uvicorn bgem3_rerank"
pkill -f "uvicorn bgem3_mcp"
```

## Architecture and Data Flow

### Components

- `bgem3_embed.py`: FastAPI embedding service (port 8000)
- `bgem3_mcp.py`: FastMCP server exposing embed, embed_hybrid, rerank tools (port 8001)
- `bgem3_rerank.py`: FastAPI reranking service (port 8002)
- `restart.sh`: Primary startup script — kills stale, starts all 3 services
- `start.py`: Legacy smoke test runner
- `preflight.py`: Pre-startup validation
- `test_service.py`: Smoke tests

### Port Allocation

- 8000: bgem3_embed (BGE-M3 embeddings)
- 8001: bgem3_mcp (FastMCP tools)
- 8002: bgem3_rerank (bge-reranker-v2-m3)

### Control Flow (restart.sh)

1. Kill existing uvicorn processes for each service
2. Start bgem3_embed on 10.230.57.109:8000
3. Start bgem3_rerank on 10.230.57.109:8002
4. Start bgem3_mcp on 10.230.57.109:8001 with `--factory`
5. Verify all services healthy

## API Usage

### Embedding (port 8000)

**CORRECT**:
```bash
curl -X POST http://10.230.57.109:8000/embed \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'
```

**WRONG** (never do this):
```bash
# Using 0.0.0.0
curl http://0.0.0.0:8000/embed

# Using X-API-Key header
curl -X POST http://10.230.57.109:8000/embed \
  -H "X-API-Key: m1macmini" \
  ...
```

### MCP (port 8001)

MCP requires session initialization. Always call `initialize` before tool calls:

```python
import httpx
import json

client = httpx.Client(follow_redirects=True)

# 1. Initialize → get session ID
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

# 2. Call tool with session ID
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

# 3. Parse SSE response
for line in tool_resp.text.splitlines():
    if line.startswith("data:"):
        result = json.loads(line[5:].strip())
        break
```

## Key Patterns and Conventions

### load_dotenv Required

Both `bgem3_embed.py` and `bgem3_rerank.py` call `load_dotenv()` at module level:

```python
from dotenv import load_dotenv
load_dotenv()
```

This is critical. Without it, `.env` is never loaded and `EMBEDDING_API_KEY` is empty → auth disabled.

### Model Loading

Models loaded at module level on import. Uses MPS backend:

```python
from FlagEmbedding import BGEM3FlagModel
_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="mps")
```

- Loaded once on import
- Uses MPS backend (Apple GPU)
- No hot-reload

### Logging

Logs in `logs/` directory:
- `logs/bgem3_embed.log`
- `logs/bgem3_mcp.log`
- `logs/bgem3_rerank.log`

### Authentication

- API key: `m1macmini` (set in `.env` via `EMBEDDING_API_KEY`)
- Use header: `Authorization: Bearer m1macmini`

## Gotchas and Non-Obvious Patterns

### MCP Initialize Required

FastMCP streamable-http requires session initialization. Newer FastMCP versions (>=3.x) require this before any `tools/call`.

### MCPCall with Factory

FastMCP 3.x requires `--factory` flag and the callable `mcp.http_app`:

```bash
# CORRECT:
uv run uvicorn "bgem3_mcp:mcp.http_app" --host 10.230.57.109 --port 8001 --factory

# WRONG - missing --factory:
uv run uvicorn bgem3_mcp:app --host 10.230.57.109 --port 8001
```

### stateless_http Parameter Location

FastMCP versions >= 3.x don't accept `stateless_http` in `FastMCP.__init__()`. Pass it to `run()` instead:

```python
# WRONG (raises TypeError):
mcp = FastMCP("BGEM3", stateless_http=True)

# CORRECT:
mcp = FastMCP("BGEM3")
mcp.run(transport="streamable-http", host=IP, port=PORT, stateless_http=True)
```

Without this, FastMCP holds SSE connections open waiting for session state that never completes.

### Explicit Timeouts

FastMCP tools call downstream services via httpx. Timeouts are defined:

```python
_EMBED_TIMEOUT  = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
_RERANK_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
_HYBRID_TIMEOUT = httpx.Timeout(connect=5.0, read=45.0, write=5.0, pool=5.0)
```

- `connect=5.0`: Fail fast if service is down
- `read=30/60`: Allow time for model inference

### Error Handling

HTTP errors are caught and raised as `ValueError` with readable messages:

```python
except httpx.HTTPStatusError as e:
    raise ValueError(f"bgem3_embed error {e.response.status_code}: {e.response.text}") from e
```

### ZeroTier IP

Always look up the current IP if ZeroTier network changed:

```bash
ifconfig | grep "inet " | grep -v 127.0
```

If IP changes, update all references to `10.230.57.109` in:
- `.launch/*.plist` files
- `restart.sh`
- `bgem3_mcp.py` constants

### Testing

```bash
uv run test_service.py
```

Runs 11 tests:
- `/health` on all 3 services
- `/info` on embed
- `/embed` returning 1024-dim vector
- `/embed/hybrid` returning dense + sparse
- `/embed` rejects bad API key (401)
- `/rerank` returning sorted passages
- MCP embed tool
- MCP embed_hybrid tool
- MCP rerank tool