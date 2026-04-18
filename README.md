# BGEM3 Embedding Service

Self-hosted embedding generation service using the [BAAI/bge-m3](https://github.com/_flagattribute/FlagEmbedding) model, optimized for Apple Silicon Macs (M1/M2/M3) via Metal Performance Shaders (MPS).

## Purpose

- Generate **dense embeddings** (1024-dimensional) for text retrieval
- Generate **hybrid embeddings** (dense + sparse) for hybrid search
- Exposed via REST API and MCP (Model Context Protocol) for AI agent integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BGEM3 Service                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐      ┌─────────────────────┐      │
│  │  rag_server:8000    │      │  mcp_server:8001    │      │
│  │  (FastAPI)          │      │  (FastMCP)          │      │
│  │  - /embed          │      │  - embed() tool     │      │
│  │  - /embed/hybrid  │      │  - embed_hybrid()  │      │
│  │  - /health        │      │    tool           │      │
│  │  - /info          │      │                   │      │
│  └─────────┬──────────┘      └─────────┬─────────┘      │
│            │                          │                  │
│            │    BGE-M3 Model         │                  │
│            │  (FlagEmbedding)        │                  │
│            │  on MPS (Apple GPU)     │                  │
│            └──────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- ZeroTier installed and connected to network `10.230.57.x`
- Python 3.11

### Start Services

```bash
# Terminal 1: Embedding API (port 8000)
./scripts/start_server.sh

# Terminal 2: MCP Server (port 8001)
./scripts/start_mcp.sh
```

### Test It

```bash
# Test embedding API
curl -X POST http://10.230.57.109:8000/embed \
  -H "X-API-Key: m1macmini" \
  -H "Content-Type: application/json" \
  -d '["hello world", "test embedding"]'

# Health check
curl http://10.230.57.109:8000/health
```

## API Reference

### REST API (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate dense embeddings (max 32 texts) |
| `/embed/hybrid` | POST | Generate dense + sparse embeddings (max 8 texts) |
| `/health` | GET | Health check with system metrics |
| `/info` | GET | Service and model information |

#### Authentication

Include API key in header:
```
X-API-Key: m1macmini
```

#### Request/Response Examples

**POST /embed**
```bash
# Request
curl -X POST http://10.230.57.109:8000/embed \
  -H "X-API-Key: m1macmini" \
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
# Request
curl -X POST http://10.230.57.109:8000/embed/hybrid \
  -H "X-API-Key: m1macmini" \
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

The MCP server exposes two tools for AI agent integration:

| Tool | Description |
|------|-------------|
| `embed(texts: list[str])` | Generate dense embeddings |
| `embed_hybrid(texts: list[str])` | Generate dense + sparse embeddings |

Connect to: `http://10.230.57.109:8001/mcp`

## Configuration

### Key Files

| File | Purpose |
|------|----------|
| `rag_server.py` | FastAPI embedding service |
| `mcp_server.py` | FastMCP server for AI agents |
| `scripts/start_server.sh` | Start embedding API |
| `scripts/start_mcp.sh` | Start MCP server |
| `scripts/get_zt_ip.sh` | Get ZeroTier IP address |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `m1macmini` | API authentication key |
| `MCP_PORT` | `8001` | MCP server port |

### Model Configuration

- **Model**: BAAI/bge-m3
- **Dimensions**: 1024
- **Device**: MPS (Apple Silicon GPU)
- **Batch size**: 8 (embed), 4 (hybrid)
- **Max texts**: 32 (embed), 8 (hybrid)

## Troubleshooting

### Server Won't Start

1. **Check ZeroTier is running**:
   ```bash
   # Verify network connection
   ifconfig zt0
   ```

2. **Check ports are available**:
   ```bash
   lsof -i :8000
   lsof -i :8001
   ```

3. **Check logs**:
   ```bash
   tail -50 logs/server.log
   tail -50 logs/mcp_server.log
   ```

### Slow Embeddings

- First request triggers model warmup (expected)
- Check MPS is available: `curl http://10.230.57.109:8000/health`
- Reduce batch size in `rag_server.py` if memory issues

### Authentication Errors

- Verify API key: `X-API-Key: m1macmini`
- Check header format: exactly as shown above

### Model Loading Errors

- Ensure sufficient disk space for model (~500MB)
- Check internet connection (initial model download)
- Verify macOS on Apple Silicon

### Memory Issues

- Monitor with: `curl http://10.230.57.109:8000/health`
- Reduce batch_size in rag_server.py
- Restart services to clear MPS memory

## Making Private

To make this repository private:

1. **GitHub**:
   - Go to repository Settings → Danger Zone → Change repository visibility
   - Select "Make private"

2. **Local changes**:
   ```bash
   # Commit any sensitive files first
   git add .
   git commit -m "Add documentation"
   git push origin main
   ```

3. **API Key**:
   - Change `API_KEY` in `mcp_server.py` before pushing
   - Or use environment variable

## Security Notes

- API key is hardcoded (`m1macmini`) — change before production
- No rate limiting implemented
- No request logging (add if needed)
- Logs go to `logs/*.log` (gitignored)