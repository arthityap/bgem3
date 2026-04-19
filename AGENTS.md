# BGEM3 Agent Guide

This document provides essential information for AI agents working in the BGEM3 repository. It focuses on non-obvious patterns, commands, and gotchas that would otherwise require trial-and-error discovery.

## Project Overview

BGEM3 provides embedding generation via BAAI/bge-m3 and reranking via bge-reranker-v2-m3 through a FastMCP interface. Designed for Apple Silicon Macs (M1/M2) using MPS backend.

## Essential Commands

### Starting All Services

```bash
python start.py
```

This starts three services:
- `bgem3_embed` (port 8000) — BGE-M3 embeddings
- `bgem3_mcp` (port 8001) — FastMCP server (HTTP transport)
- `bgem3_rerank` (port 8002) — bge-reranker-v2-m3

### Preflight Checks

```bash
python preflight.py
```

Validates: Python 3.11, packages, GPU, cached weights, ports, MCP tools.

### API Usage

Embed endpoint (port 8000):
```bash
curl -X POST http://<host>:8000/embed \
  -H "X-API-Key: m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'
```

MCP tools (port 8001) — use JSON-RPC with session initialization:
```bash
# 1. Initialize → get mcp-session-id
# 2. Call tool with mcp-session-id header
```

## Architecture and Data Flow

### Components

- `bgem3_embed.py`: FastAPI embedding service (port 8000)
- `bgem3_mcp.py`: FastMCP server exposing embed, embed_hybrid, rerank tools (port 8001)
- `bgem3_rerank.py`: FastAPI reranking service (port 8002)
- `start.py`: Orchestrates starting all three services
- `preflight.py`: Pre-startup validation
- `test_service.py`: Smoke tests for all services

### Control Flow (start.py)

1. Run preflight checks
2. Start bgem3_embed, wait for /health
3. Start bgem3_rerank, wait for /health  
4. Start bgem3_mcp, wait for port open
5. Initialize MCP session → list tools

### MCP Session Flow

FastMCP streamable-http requires:
1. POST `/mcp/` with `initialize` method → get `mcp-session-id` from header
2. POST `/mcp/` with `tools/call` → include `mcp-session-id` header
3. Parse SSE: extract `data:` line from response

## Key Patterns and Conventions

### API Key Authentication

- Key: `m1macmini` (set via `EMBEDDING_API_KEY` in `.env`)
- Header: `X-API-Key`

### Model Loading

Models loaded at module level:
```python
model = SentenceTransformer('BAAI/bge-m3', device='mps')
```

- Loaded once on import
- Uses MPS backend
- No hot-reload

### Logging

Logs in `logs/` directory:
- `logs/bgem3_embed.log`
- `logs/bgem3_mcp.log`
- `logs/bgem3_rerank.log`

## Gotchas and Non-Obvious Patterns

### MCP Initialize Required

Newer FastMCP versions require session initialization before `tools/list` or `tools/call`. The `start.py` `list_mcp_tools()` function now handles this.

### Port Allocation

- 8000: bgem3_embed
- 8001: bgem3_mcp
- 8002: bgem3_rerank

### Testing

Use `test_service.py` — runs smoke tests for all three services including MCP session flow.
