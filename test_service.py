"""
test_service.py — Smoke-test all three services + MCP tools.

Tests:
  rag_server (8000):
    1. GET  /health          — alive + healthy
    2. GET  /info            — model/device info
    3. POST /embed           — returns 1024-dim vector
    4. POST /embed/hybrid    — returns dense + sparse
    5. POST /embed (bad key) — returns HTTP 401 when auth enabled

  reranker_server (8002):
    6. GET  /health          — alive + healthy
    7. POST /rerank          — returns sorted scored passages

  mcp_server (8001) — HTTP transport:
    8. GET  /                — FastMCP root responds
    9. POST /mcp/ embed      — MCP tool returns 1024-dim vector
   10. POST /mcp/ embed_hybrid — MCP tool returns dense + sparse
   11. POST /mcp/ rerank     — MCP tool returns sorted scored passages

Usage:
    uv run python test_service.py
"""

import json
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv()

RAG_URL      = "http://10.230.57.109:8000"
MCP_URL      = "http://10.230.57.109:8001"
RERANK_URL   = "http://10.230.57.109:8002"
API_KEY      = os.getenv("EMBEDDING_API_KEY", "m1macmini")
HEADERS      = {"Authorization": f"Bearer {API_KEY}"}
TEST_TEXT    = "The Ten Gods in Bazi represent the relationship between the Day Master and other elements."
TEST_QUERY   = "What are the Ten Gods in Bazi?"
TEST_PASSAGES = [
    "The Ten Gods describe relationships between the Day Master and the other nine stems.",
    "Bazi uses four pillars: Year, Month, Day, and Hour.",
    "The Ten Gods are classified into Wealth, Officer, Resource, Companion, and Output categories.",
]

PASS = "\033[32m OK  \033[0m"
FAIL = "\033[31m FAIL\033[0m"

passed = failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        print(f"[{PASS}] {label}")
        passed += 1
    else:
        print(f"[{FAIL}] {label}: {detail}")
        failed += 1


def mcp_call(client: httpx.Client, tool: str, arguments: dict) -> dict | None:
    """Send a JSON-RPC 2.0 tools/call to the FastMCP streamable-http endpoint."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    try:
        r = client.post(
            f"{MCP_URL}/mcp/",
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            content=json.dumps(payload),
            timeout=60,
        )
        # FastMCP streamable-http may return SSE or plain JSON depending on version.
        # Parse the first data: line if SSE, else parse body directly.
        text = r.text.strip()
        if text.startswith("data:"):
            # SSE — grab first data line
            for line in text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[len("data:"):].strip())
        return r.json()
    except Exception as e:
        return {"error": str(e)}


print("\n=== BGE-M3 + Reranker + MCP Service Tests ===\n")

with httpx.Client(timeout=60) as client:

    # ── rag_server (8000) ──────────────────────────────────────────────────────────────
    print("--- rag_server (8000) ---")

    try:
        r = client.get(f"{RAG_URL}/health")
        data = r.json()
        check("/health → 200 + healthy", r.status_code == 200 and data.get("status") == "healthy")
        print(f"       cpu={data.get('cpu_percent')}%  mem={data.get('memory_percent')}%  mps={data.get('mps_available')}")
    except Exception as e:
        check("/health", False, str(e))

    try:
        r = client.get(f"{RAG_URL}/info")
        data = r.json()
        check("/info → 200 + model info", r.status_code == 200 and "model" in data)
        print(f"       model={data.get('model')}  device={data.get('device')}  auth={data.get('auth_enabled')}")
    except Exception as e:
        check("/info", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        vecs = data.get("embeddings", [])
        ok_ = r.status_code == 200 and len(vecs) == 1 and len(vecs[0]) == 1024
        check("/embed → 1x1024 vector", ok_, f"status={r.status_code}")
    except Exception as e:
        check("/embed", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed/hybrid", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        ok_ = r.status_code == 200 and "dense_embeddings" in data and "sparse_embeddings" in data
        check("/embed/hybrid → dense + sparse", ok_, f"status={r.status_code}")
    except Exception as e:
        check("/embed/hybrid", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed", headers={"Authorization": "Bearer wrongkey"}, json=[TEST_TEXT])
        if API_KEY:
            check("/embed rejects bad key → 401", r.status_code == 401, f"got {r.status_code}")
        else:
            check("/embed auth disabled", r.status_code == 200)
    except Exception as e:
        check("/embed bad-key test", False, str(e))

    # ── reranker_server (8002) ─────────────────────────────────────────────────────────
    print("\n--- reranker_server (8002) ---")

    try:
        r = client.get(f"{RERANK_URL}/health")
        data = r.json()
        check("/health → 200 + healthy", r.status_code == 200 and data.get("status") == "healthy")
        print(f"       model={data.get('model')}  mps={data.get('mps_available')}")
    except Exception as e:
        check("/health", False, str(e))

    try:
        r = client.post(
            f"{RERANK_URL}/rerank",
            headers=HEADERS,
            json={"query": TEST_QUERY, "passages": TEST_PASSAGES, "top_n": 2},
        )
        data = r.json()
        results = data.get("results", [])
        ok_ = (
            r.status_code == 200
            and len(results) == 2
            and all("score" in x and "text" in x and "index" in x for x in results)
            and results[0]["score"] >= results[1]["score"]
        )
        check("/rerank → 2 sorted scored passages", ok_, f"status={r.status_code} results={len(results)}")
        if results:
            print(f"       top score={results[0]['score']:.4f}  idx={results[0]['index']}")
    except Exception as e:
        check("/rerank", False, str(e))

    # ── mcp_server (8001) ─────────────────────────────────────────────────────────────
    print("\n--- mcp_server (8001) ---")

    # 8. Root alive
    try:
        r = client.get(f"{MCP_URL}/", timeout=5)
        check("GET / → port responding", r.status_code < 500, f"status={r.status_code}")
    except Exception as e:
        check("GET / port responding", False, str(e))

    # 9. MCP tool: embed
    resp = mcp_call(client, "embed", {"texts": [TEST_TEXT]})
    err = resp.get("error") if resp else "no response"
    if err and not isinstance(err, dict):
        check("MCP tool: embed → 1024-dim vector", False, str(err))
    else:
        # result is in resp["result"]["content"][0]["text"] as JSON string
        try:
            content = resp.get("result", {}).get("content", [{}])[0].get("text", "[]")
            vecs = json.loads(content) if isinstance(content, str) else content
            ok_ = isinstance(vecs, list) and len(vecs) == 1 and len(vecs[0]) == 1024
            check("MCP tool: embed → 1x1024 vector", ok_, f"got shape {len(vecs)}x{len(vecs[0]) if vecs else 0}")
        except Exception as e:
            check("MCP tool: embed", False, str(e))

    # 10. MCP tool: embed_hybrid
    resp = mcp_call(client, "embed_hybrid", {"texts": [TEST_TEXT]})
    err = resp.get("error") if resp else "no response"
    if err and not isinstance(err, dict):
        check("MCP tool: embed_hybrid → dense + sparse", False, str(err))
    else:
        try:
            content = resp.get("result", {}).get("content", [{}])[0].get("text", "{}")
            data = json.loads(content) if isinstance(content, str) else content
            ok_ = "dense_embeddings" in data and "sparse_embeddings" in data
            check("MCP tool: embed_hybrid → dense + sparse", ok_, f"keys={list(data.keys())}")
        except Exception as e:
            check("MCP tool: embed_hybrid", False, str(e))

    # 11. MCP tool: rerank
    resp = mcp_call(client, "rerank", {"query": TEST_QUERY, "passages": TEST_PASSAGES, "top_n": 2})
    err = resp.get("error") if resp else "no response"
    if err and not isinstance(err, dict):
        check("MCP tool: rerank → sorted passages", False, str(err))
    else:
        try:
            content = resp.get("result", {}).get("content", [{}])[0].get("text", "[]")
            results = json.loads(content) if isinstance(content, str) else content
            ok_ = (
                isinstance(results, list)
                and len(results) == 2
                and results[0]["score"] >= results[1]["score"]
            )
            check("MCP tool: rerank → 2 sorted passages", ok_, f"got {len(results)} results")
            if results:
                print(f"       top score={results[0]['score']:.4f}  idx={results[0]['index']}")
        except Exception as e:
            check("MCP tool: rerank", False, str(e))

# Summary
print()
if failed == 0:
    print(f"\033[32mAll {passed} tests passed.\033[0m")
else:
    print(f"\033[31m{failed} test(s) failed, {passed} passed.\033[0m")
    sys.exit(1)
