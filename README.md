# BGEM3 Embedding Service

Self-hosted embedding generation service using the [BAAI/bge-m3](https://github.com/flagattribute/FlagEmbedding) model, optimized for Apple Silicon Macs (M1/M2/M3) via Metal Performance Shaders (MPS).

## Purpose

- Generate **dense embeddings** (1024-dimensional) for text retrieval
- Generate **hybrid embeddings** (dense + sparse) for hybrid search
- Expose via REST API and MCP (Model Context Protocol) for AI agent integration

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

## CRITICAL - Always Use These Conventions

> **ALWAYS use `uv` for all Python commands** — never `pip`, never bare `python`
> 
> **ALWAYS use the ZeroTier IP `10.230.57.109`** — never `0.0.0.0`, never `127.0.0.1`
> 
> **ALWAYS use Authorization Bearer token** — not `X-API-Key` header

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- ZeroTier connected to network `10.230.57.x` (check with `ifconfig` to get IP)
- **Python 3.11** (required — see `.python-version`)

### Start Services

**Option 1: Use restart.sh (recommended)**
```bash
cd /Users/yapilymm/Downloads/projects/bgem3
./restart.sh
```

**Option 2: Manual uvicorn**
```bash
cd /Users/yapilymm/Downloads/projects/bgem3
uv run uvicorn bgem3_embed:app --host 10.230.57.109 --port 8000 &
uv run uvicorn bgem3_rerank:app --host 10.230.57.109 --port 8002 &
uv run uvicorn "bgem3_mcp:mcp.http_app" --host 10.230.57.109 --port 8001 --factory &
```

**Option 3: launchctl (after reboot)**
```bash
launchctl load .launch/com.bgem3.embed.plist
launchctl load .launch/com.bgem3.rerank.plist
launchctl load .launch/com.bgem3.mcp.plist
```

### Test It

```bash
uv run test_service.py
```

Expected output: `All 11 tests passed.`

### Stop Services

```bash
pkill -f "uvicorn bgem3_embed"
pkill -f "uvicorn bgem3_rerank"
pkill -f "uvicorn bgem3_mcp"
```

## Files

| File | Purpose |
|------|----------|
| `restart.sh` | **Primary startup script** — kills stale, starts all 3 services with correct IPs |
| `start.py` | Legacy — now just a smoke test runner |
| `preflight.py` | Environment checks (Python 3.11, packages, ports, model cache) |
| `test_service.py` | Smoke tests for all services + MCP tools |
| `bgem3_embed.py` | FastAPI embedding service (port 8000) |
| `bgem3_rerank.py` | FastAPI reranker service (port 8002) |
| `bgem3_mcp.py` | FastMCP server with embed/hybrid/rerank tools (port 8001) |
| `.env` | API key: `EMBEDDING_API_KEY=m1macmini` |
| `.launch/*.plist` | launchd plists for auto-start on boot |
| `.python-version` | Pins to Python 3.11 |
| `pyproject.toml` | Project dependencies |

## API Reference

### REST API (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate dense embeddings (max 32 texts) |
| `/embed/hybrid` | POST | Generate dense + sparse embeddings (max 8 texts) |
| `/health` | GET | Health check with system metrics |
| `/info` | GET | Service and model information |

#### Authentication

**CORRECT** — use Authorization Bearer:
```bash
curl -X POST http://10.230.57.109:8000/embed \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'
```

**WRONG** — never use X-API-Key:
```bash
# DON'T DO THIS
curl -X POST http://10.230.57.109:8000/embed \
  -H "X-API-Key: m1macmini" \
  ...
```

#### Response Examples

**POST /embed**
```json
{
  "embeddings": [[0.0123, -0.0456, ...]],
  "count": 1,
  "dimensions": 1024,
  "model": "BAAI/bge-m3"
}
```

**POST /embed/hybrid**
```json
{
  "dense_embeddings": [[0.0123, -0.0456, ...]],
  "sparse_embeddings": [{"token": 0.5}, {"hello": 0.3}, ...],
  "model": "BAAI/bge-m3"
}
```

### Reranker API (port 8002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rerank` | POST | Rerank passages for a query |
| `/health` | GET | Health check |
| `/info` | GET | Model info |

```bash
curl -X POST http://10.230.57.109:8002/rerank \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is foo?", "passages": ["foo is a bar", "bar is foo", "baz"]}'
```

### MCP Server (port 8001)

Connect to: `http://10.230.57.109:8001/mcp`

Tools available:
| Tool | Description |
|------|-------------|
| `embed(texts: list[str])` | Generate dense embeddings |
| `embed_hybrid(texts: list[str])` | Generate dense + sparse embeddings |
| `rerank(query: str, passages: list[str])` | Rerank passages |

## Preflight Checks

Run `uv run preflight.py` to verify:

- Python == 3.11.x
- All required packages installed
- MPS available (Apple Silicon GPU)
- `.env` file exists with `EMBEDDING_API_KEY=m1macmini`
- BGE-M3 + Reranker weights cached locally
- Ports 8000, 8001, 8002 are free

## Troubleshooting

### Services Won't Start

1. **Check ZeroTier IP is `10.230.57.109`**:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0
   ```
   If IP changed, update all files that reference `10.230.57.109`

2. **Check ports**:
   ```bash
   lsof -i :8000 -i :8001 -i :8002
   ```

3. **Check running processes**:
   ```bash
   ps aux | grep uvicorn | grep -v grep
   ```

4. **Check logs**:
   ```bash
   tail -30 logs/bgem3_embed.log
   tail -30 logs/bgem3_mcp.log
   tail -30 logs/bgem3_rerank.log
   ```

### Health Check

```bash
curl http://10.230.57.109:8000/health
curl http://10.230.57.109:8002/health
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

### Auth Not Working

If `/info` shows `"auth_enabled": false` but `.env` has the key:

**Cause**: Services started without loading `.env`

**Fix**: Services MUST call `load_dotenv()` at module level. This is already done in `bgem3_embed.py` and `bgem3_rerank.py`. If you see auth disabled, the service may be an old process — restart with `restart.sh`

### MCP Tool Calls Hang

**Cause**: Using FastMCP without `stateless_http=True`

**Fix**: Start MCP with `--factory` and the callable `mcp.http_app`:
```bash
uv run uvicorn "bgem3_mcp:mcp.http_app" --host 10.230.57.109 --port 8001 --factory
```

### Model Loading Issues

- First start downloads ~2.3GB model to `~/.cache/huggingface/`
- Reranker model is ~1.1GB (downloaded on first use)
- Check internet connection to HuggingFace

### Memory Issues

- Monitor via `/health` endpoint
- Reduce batch_size in `bgem3_embed.py` if needed

## Key Technical Details

### load_dotenv Required

Both `bgem3_embed.py` and `bgem3_rerank.py` must call `load_dotenv()` at the top to read the API key from `.env`:

```python
from dotenv import load_dotenv
load_dotenv()
```

Without this, `EMBEDDING_API_KEY` is empty so auth is disabled even if `.env` has the key.

### MCP Session Flow

FastMCP streamable-http requires session initialization:

1. POST `/mcp/` with `initialize` method → get `mcp-session-id` header
2. POST `/mcp/` with `tools/call` → include `mcp-session-id` header
3. Parse SSE: extract `data:` line from response

### Explicit Timeouts

MCP tools have explicit httpx timeouts:

- `_EMBED_TIMEOUT`: connect=5s, read=30s
- `_RERANK_TIMEOUT`: connect=5s, read=60s
- `_HYBRID_TIMEOUT`: connect=5s, read=45s

### launchd Plists Location

- `.launch/com.bgem3.embed.plist` — starts bgem3_embed
- `.launch/com.bgem3.rerank.plist` — starts bgem3_rerank  
- `.launch/com.bgem3.mcp.plist` — starts bgem3_mcp

All use ZeroTier IP `10.230.57.109` — never `0.0.0.0`

To enable auto-start on boot:
```bash
launchctl load .launch/com.bgem3.embed.plist
launchctl load .launch/com.bgem3.rerank.plist
launchctl load .launch/com.bgem3.mcp.plist
```