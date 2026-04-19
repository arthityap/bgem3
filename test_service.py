"""
test_service.py — Smoke-test rag_server (port 8000) and mcp_server (port 8001).

Tests:
  rag_server (8000):
    1. GET  /health          — service alive and reports healthy
    2. GET  /info            — returns model/device info
    3. POST /embed           — returns 1024-dim vector for a test sentence
    4. POST /embed/hybrid    — returns dense + sparse for a test sentence
    5. POST /embed (bad key) — returns HTTP 401 when auth is enabled

  mcp_server (8001):
    6. GET  /                — FastMCP root responds (service alive)

Usage:
    uv run python test_service.py
"""

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

RAG_URL  = "http://10.230.57.109:8000"
MCP_URL  = "http://10.230.57.109:8001"
API_KEY  = os.getenv("EMBEDDING_API_KEY", "m1macmini")
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}
TEST_TEXT = "The Ten Gods in Bazi represent the relationship between the Day Master and other elements."

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


print("\n=== BGE-M3 Service Tests ===\n")

with httpx.Client(timeout=30) as client:

    # 1. /health
    try:
        r = client.get(f"{RAG_URL}/health")
        data = r.json()
        check("/health returns 200 + status=healthy", r.status_code == 200 and data.get("status") == "healthy")
        print(f"       cpu={data.get('cpu_percent')}%  mem={data.get('memory_percent')}%  mps={data.get('mps_available')}")
    except Exception as e:
        check("/health", False, str(e))

    # 2. /info
    try:
        r = client.get(f"{RAG_URL}/info")
        data = r.json()
        check("/info returns 200 + model info", r.status_code == 200 and "model" in data)
        print(f"       model={data.get('model')}  device={data.get('device')}  auth={data.get('auth_enabled')}")
    except Exception as e:
        check("/info", False, str(e))

    # 3. POST /embed
    try:
        r = client.post(f"{RAG_URL}/embed", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        vecs = data.get("embeddings", [])
        ok = r.status_code == 200 and len(vecs) == 1 and len(vecs[0]) == 1024
        check("/embed returns 1x1024 vector", ok, f"status={r.status_code} shape={len(vecs)}x{len(vecs[0]) if vecs else 0}")
    except Exception as e:
        check("/embed", False, str(e))

    # 4. POST /embed/hybrid
    try:
        r = client.post(f"{RAG_URL}/embed/hybrid", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        ok = r.status_code == 200 and "dense_embeddings" in data and "sparse_embeddings" in data
        check("/embed/hybrid returns dense + sparse", ok, f"status={r.status_code}")
    except Exception as e:
        check("/embed/hybrid", False, str(e))

    # 5. Auth rejection (wrong key) — only meaningful if auth is enabled
    try:
        r = client.post(f"{RAG_URL}/embed", headers={"Authorization": "Bearer wrongkey"}, json=[TEST_TEXT])
        if API_KEY:
            check("/embed rejects bad key with 401", r.status_code == 401, f"got {r.status_code} instead")
        else:
            check("/embed auth disabled (no EMBEDDING_API_KEY set)", r.status_code == 200, warn=True)
    except Exception as e:
        check("/embed bad-key test", False, str(e))

    # 6. MCP server alive
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
