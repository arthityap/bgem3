import os

import httpx
from fastmcp import FastMCP

mcp = FastMCP("BGEM3Embedder")

BGEM3_URL = "http://10.230.57.109:8000"
# API key must match EMBEDDING_API_KEY set in rag_server's .env.
# Read from env so it's not hardcoded; falls back to 'm1macmini' for local dev.
API_KEY   = os.getenv("EMBEDDING_API_KEY", "m1macmini")
ZT_IP     = "10.230.57.109"
MCP_PORT  = 8001


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
            # Authorization: Bearer — must match rag_server.py EMBEDDING_API_KEY
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
            # Authorization: Bearer — must match rag_server.py EMBEDDING_API_KEY
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=texts,
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=ZT_IP, port=MCP_PORT)
