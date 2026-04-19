import asyncio
import os
import time
from typing import Any, Dict

import anyio
import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from FlagEmbedding import BGEM3FlagModel

# Configure logging
# Custom logging handler removed to prevent recursion
# Was causing maximum recursion depth exceeded error

# Module-level semaphore and model variable
_mps_lock: asyncio.Semaphore
_model: BGEM3FlagModel = None

# Optional API key auth for embedding endpoints.
# Set EMBEDDING_API_KEY in .env to enable. Leave empty to disable auth.
# /health and /info are always public.
_EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
_bearer_scheme = HTTPBearer(auto_error=False)


def _check_api_key(credentials: HTTPAuthorizationCredentials | None) -> None:
    """Raise HTTP 401 if auth is enabled and the token is wrong/missing."""
    if not _EMBEDDING_API_KEY:
        return  # auth disabled
    if credentials is None or credentials.credentials != _EMBEDDING_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def lifespan(app: FastAPI):
    global _mps_lock, _model
    _mps_lock = asyncio.Semaphore(1)
    try:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="mps")
        # torch.compile with aot_eager backend (NOT inductor — crashes on MPS)
        try:
            import torch

            _model.model = torch.compile(_model.model, backend="aot_eager")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
        # Warmup with 2 texts at real batch_size to exercise compiled path
        _model.encode(
            ["warmup text one", "warmup text two"],
            batch_size=2,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        print("BGE-M3 ready on MPS")
        if _EMBEDDING_API_KEY:
            print("Auth enabled: Bearer token required for /embed endpoints")
        else:
            print("Auth disabled: /embed endpoints are public")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise
    yield
    # Cleanup
    if _model is not None:
        del _model
        torch.mps.empty_cache()


app = FastAPI(title="BGEM3 Embedding Service", lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint with system metrics (always public)"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    return {
        "status": "healthy",
        "service": "bgem3",
        "timestamp": int(time.time()),
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available": memory.available,
        "mps_available": torch.backends.mps.is_available(),
        "semaphore_locked": _mps_lock.locked() if _mps_lock is not None else False,
    }


@app.get("/info")
def info() -> Dict[str, Any]:
    """Comprehensive service information endpoint (always public)"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
        }
    elif _model is not None and torch.backends.mps.is_available():
        gpu_info = {"mps_available": True, "device": "Apple Silicon GPU"}

    return {
        "model": "BAAI/bge-m3",
        "dimensions": 1024,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "batch_size": 4,
        "framework": "FlagEmbedding",
        "framework_version": torch.__version__,
        "gpu_info": gpu_info,
        "torch_version": torch.__version__,
        "auth_enabled": bool(_EMBEDDING_API_KEY),
    }


@app.post("/embed")
async def embed(
    texts: list[str],
    credentials: HTTPAuthorizationCredentials | None = __import__("fastapi").Depends(
        _bearer_scheme
    ),
) -> Dict[str, Any]:
    """Generate dense embeddings for input texts.

    Args:
        texts: List of strings to embed (max 32 on M1 16GB)
        credentials: Bearer token — required when EMBEDDING_API_KEY is set in .env

    Returns:
        {embeddings, count, dimensions, model}

    Raises:
        HTTP 401 if auth is enabled and token is wrong/missing
        HTTP 422 if texts is empty
        HTTP 429 if more than 32 texts
        HTTP 500 on model error
    """
    _check_api_key(credentials)

    if not texts:
        raise HTTPException(status_code=422, detail="Text list cannot be empty")
    if len(texts) > 32:
        raise HTTPException(
            status_code=429, detail="max 32 texts per request on M1 16GB"
        )

    try:
        async with _mps_lock:
            result = await anyio.to_thread.run_sync(
                lambda: _model.encode(
                    texts,
                    batch_size=8,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            )
        return {
            "embeddings": result["dense_vecs"].tolist(),
            "count": len(texts),
            "dimensions": 1024,
            "model": "BAAI/bge-m3",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate embeddings: {str(e)}"
        )


@app.post("/embed/hybrid")
async def embed_hybrid(
    texts: list[str],
    credentials: HTTPAuthorizationCredentials | None = __import__("fastapi").Depends(
        _bearer_scheme
    ),
) -> Dict[str, Any]:
    """Hybrid embedding endpoint returning dense + sparse vectors in one pass.

    Args:
        texts: List of strings to embed (max 8 on M1 16GB)
        credentials: Bearer token — required when EMBEDDING_API_KEY is set in .env

    Returns:
        {dense_embeddings, sparse_embeddings, model}

    Raises:
        HTTP 401 if auth is enabled and token is wrong/missing
        HTTP 429 if more than 8 texts
        HTTP 500 on model error
    """
    _check_api_key(credentials)

    if len(texts) > 8:
        raise HTTPException(status_code=429, detail="max 8 texts for hybrid on M1 16GB")

    try:
        async with _mps_lock:
            result = await anyio.to_thread.run_sync(
                lambda: _model.encode(
                    texts,
                    batch_size=4,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
            )
        sparse = [
            {str(k): float(v) for k, v in vec.items()}
            for vec in result["lexical_weights"]
        ]
        return {
            "dense_embeddings": result["dense_vecs"].tolist(),
            "sparse_embeddings": sparse,
            "model": "BAAI/bge-m3",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate embeddings: {str(e)}"
        )
