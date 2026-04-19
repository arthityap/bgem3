"""
bgem3_rerank.py — Cross-encoder reranking service using BAAI/bge-reranker-v2-m3.

Runs on port 8002. Called AFTER vector search to re-score and re-rank
candidate passages for higher precision before feeding to the LLM.

Endpoints:
    POST /rerank      — rerank passages for a query, returns sorted scored list
    GET  /health      — liveness check (always public)
    GET  /info        — model/device info (always public)

Auth:
    Same EMBEDDING_API_KEY as rag_server. Set in .env.
    If empty, auth is disabled. If set, Bearer token required on /rerank.

Usage:
    uv run uvicorn reranker_server:app --host 0.0.0.0 --port 8002

Model: BAAI/bge-reranker-v2-m3 (~1.1 GB, pairs perfectly with bge-m3 embeddings)
"""

import asyncio
import os
import time
from typing import Any

import anyio
import psutil
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from FlagEmbedding import FlagReranker
from pydantic import BaseModel

load_dotenv()

# ── Auth ────────────────────────────────────────────────────────────────────────────────────
# Reads same key as rag_server — one key for all local services.
_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
_bearer_scheme = HTTPBearer(auto_error=False)


def _check_api_key(credentials: HTTPAuthorizationCredentials | None) -> None:
    """Raise HTTP 401 if auth enabled and token is wrong/missing."""
    if not _API_KEY:
        return
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Module-level state ───────────────────────────────────────────────────────────────────────────────────────
MAX_QUEUE: int = int(os.getenv("RERANK_MAX_QUEUE", "4"))
INFERENCE_TIMEOUT: float = float(os.getenv("RERANK_INFERENCE_TIMEOUT", "60.0"))

_mps_lock: asyncio.Semaphore
_reranker: FlagReranker = None
_queue_depth: int = 0


async def _run_with_gpu_lock(fn) -> Any:
    """
    Acquire the MPS semaphore, run *fn* in a thread, and enforce INFERENCE_TIMEOUT.

    Raises:
        HTTP 503 if the queue is full (queue depth >= MAX_QUEUE).
        HTTP 504 if inference does not complete within INFERENCE_TIMEOUT seconds.
    """
    global _queue_depth

    if _queue_depth >= MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {_queue_depth} requests queued (limit {MAX_QUEUE}). Retry later.",
        )

    _queue_depth += 1
    try:
        async with _mps_lock:
            try:
                result = await asyncio.wait_for(
                    anyio.to_thread.run_sync(fn),
                    timeout=INFERENCE_TIMEOUT,
                )
                return result
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Inference timed out after {INFERENCE_TIMEOUT}s. Try fewer passages.",
                )
    finally:
        _queue_depth -= 1


async def lifespan(app: FastAPI):
    global _mps_lock, _reranker
    _mps_lock = asyncio.Semaphore(1)
    try:
        # use_fp16=True saves ~half RAM on MPS; safe for reranking scores
        _reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
            device="mps",
        )
        # Warmup
        _reranker.compute_score(
            [
                ["warmup query", "warmup passage one"],
                ["warmup query", "warmup passage two"],
            ],
            normalize=True,
        )
        print("bge-reranker-v2-m3 ready on MPS")
        print(
            f"Concurrency: MAX_QUEUE={MAX_QUEUE}, INFERENCE_TIMEOUT={INFERENCE_TIMEOUT}s"
        )
        if _API_KEY:
            print("Auth enabled: Bearer token required for /rerank")
        else:
            print("Auth disabled: /rerank is public")
    except Exception as e:
        print(f"Failed to initialise reranker: {e}")
        raise
    yield
    if _reranker is not None:
        del _reranker
        torch.mps.empty_cache()


app = FastAPI(title="BGE Reranker Service", lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}s"
    return response


# ── Request/response models ────────────────────────────────────────────────────────────────────────────────────


class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    # top_n: how many to return after reranking (0 = return all)
    top_n: int = 0


class ScoredPassage(BaseModel):
    index: int  # original position in passages[]
    score: float  # normalised relevance score 0–1
    text: str


class RerankResponse(BaseModel):
    results: list[ScoredPassage]
    query: str
    model: str
    total_passages: int
    returned: int


# ── Endpoints ──────────────────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check — always public."""
    mem = psutil.virtual_memory()
    return {
        "status": "healthy",
        "service": "reranker",
        "model": "BAAI/bge-reranker-v2-m3",
        "timestamp": int(time.time()),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": mem.percent,
        "mps_available": torch.backends.mps.is_available(),
        "semaphore_locked": _mps_lock.locked() if _mps_lock is not None else False,
        "queue_depth": _queue_depth,
        "max_queue": MAX_QUEUE,
        "inference_timeout": INFERENCE_TIMEOUT,
    }


@app.get("/info")
def info() -> dict[str, Any]:
    """Model and device info — always public."""
    return {
        "model": "BAAI/bge-reranker-v2-m3",
        "type": "cross-encoder",
        "device": "mps",
        "fp16": True,
        "auth_enabled": bool(_API_KEY),
        "torch_version": torch.__version__,
        "max_queue": MAX_QUEUE,
        "inference_timeout": INFERENCE_TIMEOUT,
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    credentials: HTTPAuthorizationCredentials | None = __import__("fastapi").Depends(
        _bearer_scheme
    ),
) -> RerankResponse:
    """Rerank passages for a query using cross-encoder scoring.

    Args:
        req.query:    The search query.
        req.passages: Candidate passages from vector search (typically 10-20).
        req.top_n:    How many top results to return (0 = return all, sorted by score).

    Returns:
        Passages sorted by relevance score (highest first), with original index preserved.

    Raises:
        HTTP 401 if auth is enabled and token is wrong/missing.
        HTTP 422 if query or passages are empty.
        HTTP 429 if more than 100 passages (M1 16GB limit).
        HTTP 503 if queue is full (retry later).
        HTTP 504 if inference times out.
        HTTP 500 on model error.
    """
    _check_api_key(credentials)

    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")
    if not req.passages:
        raise HTTPException(status_code=422, detail="passages list cannot be empty")
    if len(req.passages) > 100:
        raise HTTPException(
            status_code=429, detail="max 100 passages per request on M1 16GB"
        )

    pairs = [[req.query, p] for p in req.passages]

    try:
        scores: list[float] = await _run_with_gpu_lock(
            lambda: _reranker.compute_score(pairs, normalize=True)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {e}")

    # Sort by score descending, keep original index
    scored = sorted(
        [
            ScoredPassage(index=i, score=float(s), text=p)
            for i, (p, s) in enumerate(zip(req.passages, scores))
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    top_n = req.top_n if req.top_n > 0 else len(scored)
    results = scored[:top_n]

    return RerankResponse(
        results=results,
        query=req.query,
        model="BAAI/bge-reranker-v2-m3",
        total_passages=len(req.passages),
        returned=len(results),
    )
