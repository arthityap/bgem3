import os

import httpx
from fastmcp import FastMCP

mcp = FastMCP("BGEM3Embedder")

BGEM3_URL    = "http://10.230.57.109:8000"
RERANK_URL   = "http://10.230.57.109:8002"
# API key must match EMBEDDING_API_KEY in .env — shared by all three services.
API_KEY      = os.getenv("EMBEDDING_API_KEY", "m1macmini")
ZT_IP        = "10.230.57.109"
MCP_PORT     = 8001


@mcp.tool()
async def embed(texts: list[str]) -> list[list[float]]:
    """Generate BGE-M3 dense embeddings for a list of texts.

    Args:
        texts: List of strings to embed. Max 32 texts per call.

    Returns:
        List of 1024-dimensional float vectors, one per input text.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BGEM3_URL}/embed",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=texts,
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()["embeddings"]


@mcp.tool()
async def embed_hybrid(texts: list[str]) -> dict:
    """Generate BGE-M3 dense + sparse embeddings in a single pass.

    Use this for hybrid vector search (dense + BM25-style sparse).

    Args:
        texts: List of strings to embed. Max 8 texts per call.

    Returns:
        Dict with 'dense_embeddings' (1024-dim) and 'sparse_embeddings' (token weights).
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BGEM3_URL}/embed/hybrid",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=texts,
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def rerank(query: str, passages: list[str], top_n: int = 0) -> list[dict]:
    """Rerank candidate passages for a query using bge-reranker-v2-m3 cross-encoder.

    Call this AFTER vector search to get precision-ranked results before
    passing context to the LLM. The cross-encoder reads the full (query, passage)
    pair — much more accurate than embedding cosine distance alone.

    Args:
        query:    The user's search query or question.
        passages: Candidate passages from vector search (typically 10-20).
        top_n:    How many top results to return. 0 = return all, sorted by score.

    Returns:
        List of dicts sorted by relevance (highest first), each with:
          - index (int):  original position in the passages list
          - score (float): normalised relevance score 0–1
          - text  (str):  the passage text
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{RERANK_URL}/rerank",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"query": query, "passages": passages, "top_n": top_n},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["results"]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=ZT_IP, port=MCP_PORT)
