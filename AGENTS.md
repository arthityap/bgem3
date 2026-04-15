# BGEM3 Agent Guide

This document provides essential information for AI agents working in the BGEM3 repository. It focuses on non-obvious patterns, commands, and gotchas that would otherwise require trial-and-error discovery.

## Project Overview

BGEM3 is a service that provides embedding generation via the BAAI/bge-m3 model through a FastAPI interface. The service is designed to run on Apple Silicon Macs (M1/M2) using the MPS (Metal Performance Shaders) backend for GPU acceleration.

## Essential Commands

### Starting the Server

The server is started using uv with Python 3.11:

```bash
./scripts/start_server.sh
```

### API Usage

The server provides a single endpoint for generating embeddings:

```bash
curl -X POST http://<server-ip>:8000/embed \
  -H "X-API-Key: m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here", "another text"]'
```

## Architecture and Data Flow

### Components

- `rag_server.py`: FastAPI application that serves the embedding model
- `pyproject.toml`: Project configuration with dependencies
- `scripts/`: Startup and utility scripts
- `logs/`: Directory for server logs (though logs go to `~/Library/Logs/bgem3`)

### Control Flow

1. Server starts via the start script
2. ZeroTier IP is retrieved from `get_zt_ip.sh`
3. FastAPI app is initialized with the BAAI/bge-m3 model on MPS device
4. On startup, a warmup request is sent to load the model
5. API requests to `/embed` are processed by encoding texts with batch size 4

## Key Patterns and Conventions

### API Key Authentication

The server uses a simple API key authentication:
- Key is hardcoded as `m1macmini` (set via `API_KEY` environment variable with this default)
- Key is passed in the `X-API-Key` header
- Authentication is enforced on the `/embed` endpoint via dependency injection

### Model Loading

The model is loaded globally at module level:
```python
model = SentenceTransformer('BAAI/bge-m3', device='mps')
```

This means:
- Model is loaded once when the module is imported
- Uses Apple Silicon GPU via MPS backend
- No model reloading or hot-swapping capability

### Logging Strategy

All server logs are written to:
```
~/Library/Logs/bgem3/server.log
```

This is configured in the startup script using `tee` to capture both stdout and stderr.

## Gotchas and Non-Obvious Patterns

### Duplicate Code in Authentication

The `verify_api_key` function in `rag_server.py` contains duplicate code - the if-check and exception are repeated. This appears to be a mistake, but the function still works correctly.

### ZeroTier IP is Hardcoded

The `scripts/get_zt_ip.sh` script simply echoes a hardcoded IP address (`10.230.57.109`). This is not dynamically determined from the network interface.

### Three Different Startup Methods

The project now uses uv for dependency management and virtual environment creation. All startup scripts have been updated to use uv instead of direct venv paths.

### Warmup Function

The server includes a warmup function that runs on startup:
```python
@app.on_event("startup")
def warmup():
    model.encode(["warmup"], batch_size=1)
    print("✅ Server warmed up")
```

This is important because:
- It ensures the model is fully loaded and warmed up before serving requests
- Prevents the first API request from having unusually high latency
- Uses a single-item batch for warmup

## Testing Approach

No automated tests are present in the repository. Testing must be done manually via:

1. Starting the server with the startup script
2. Making curl requests to the `/embed` endpoint
3. Verifying the response contains embeddings (lists of floats)

Example test:
```bash
./scripts/start_server.sh &
sleep 10  # Wait for server to start
curl -X POST http://10.230.57.109:8000/embed \
  -H "X-API-Key: m1macmini" \
  -H "Content-Type: application/json" \
  -d '["hello world", "test"]'
```

## Deployment Considerations

### Environment Assumptions

The service assumes:
- Running on macOS with Apple Silicon (for MPS device)
- ZeroTier network is configured
- Python 3.11 and required packages are installed via uv

### No Configuration Files

All configuration is hardcoded:
- Model name and device
- API key
- Host and port
- Logging path

This makes the service difficult to customize without code changes.

## Future Improvements

While not required for current operation, future improvements could include:
- Configuration file for model, port, API key, etc.
- Proper error handling and logging
- Automated tests
- Docker containerization for portability
- Model reloading endpoint
- Health check endpoint
- Rate limiting
- Better authentication mechanism

However, the current implementation is minimal and focused on its core purpose of serving embeddings via the BAAI/bge-m3 model.
