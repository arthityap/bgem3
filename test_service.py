"""
test_service.py — Smoke-test all three services.

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

  mcp_server (8001):
    8. GET  /                — FastMCP root responds

Usage:
    uv run python test_service.py
"""

import os
import sys

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


print("\n=== BGE-M3 + Reranker Service Tests ===\n")

with httpx.Client(timeout=60) as client:

    # ── rag_server ────────────────────────────────────────────────────────────────────────
    print("--- rag_server (8000) ---")

    try:
        r = client.get(f"{RAG_URL}/health")
        data = r.json()
        check("/health returns 200 + status=healthy", r.status_code == 200 and data.get("status") == "healthy")
        print(f"       cpu={data.get('cpu_percent')}%  mem={data.get('memory_percent')}%  mps={data.get('mps_available')}")
    except Exception as e:
        check("/health", False, str(e))

    try:
        r = client.get(f"{RAG_URL}/info")
        data = r.json()
        check("/info returns 200 + model info", r.status_code == 200 and "model" in data)
        print(f"       model={data.get('model')}  device={data.get('device')}  auth={data.get('auth_enabled')}")
    except Exception as e:
        check("/info", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        vecs = data.get("embeddings", [])
        ok = r.status_code == 200 and len(vecs) == 1 and len(vecs[0]) == 1024
        check("/embed returns 1x1024 vector", ok, f"status={r.status_code}")
    except Exception as e:
        check("/embed", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed/hybrid", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        ok = r.status_code == 200 and "dense_embeddings" in data and "sparse_embeddings" in data
        check("/embed/hybrid returns dense + sparse", ok, f"status={r.status_code}")
    except Exception as e:
        check("/embed/hybrid", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed", headers={"Authorization": "Bearer wrongkey"}, json=[TEST_TEXT])
        if API_KEY:
            check("/embed rejects bad key with 401", r.status_code == 401, f"got {r.status_code}")
        else:
            check("/embed auth disabled", r.status_code == 200)
    except Exception as e:
        check("/embed bad-key test", False, str(e))

    # ── reranker_server ─────────────────────────────────────────────────────────────────
    print("\n--- reranker_server (8002) ---")

    try:
        r = client.get(f"{RERANK_URL}/health")
        data = r.json()
        check("/health returns 200 + status=healthy", r.status_code == 200 and data.get("status") == "healthy")
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
        ok = (
            r.status_code == 200
            and len(results) == 2
            and all("score" in r and "text" in r and "index" in r for r in results)
            and results[0]["score"] >= results[1]["score"]  # sorted descending
        )
        check("/rerank returns 2 sorted scored passages", ok, f"status={r.status_code} results={len(results)}")
        if results:
            print(f"       top passage score={results[0]['score']:.4f}  idx={results[0]['index']}")
    except Exception as e:
        check("/rerank", False, str(e))

    # ── mcp_server ─────────────────────────────────────────────────────────────────────────
    print("\n--- mcp_server (8001) ---")

    try:
        r = client.get(f"{MCP_URL}/", timeout=5)
        check("mcp_server port 8001 responding", r.status_code < 500, f"status={r.status_code}")
    except Exception as e:
        check("mcp_server port 8001 responding", False, str(e))

# Summary
print()
if failed == 0:
    print(f"\033[32mAll {passed} tests passed.\033[0m")
else:
    print(f"\033[31m{failed} test(s) failed, {passed} passed.\033[0m")
    sys.exit(1)
