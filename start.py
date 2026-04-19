"""
start.py — One-shot startup validator for all three services.

  bgem3_embed    port 8000  (BGE-M3 embeddings)
  bgem3_rerank   port 8002  (bge-reranker-v2-m3 cross-encoder)
  bgem3_mcp      port 8001  (FastMCP — exposes embed, embed_hybrid, rerank)

NOTE: launchctl owns process lifecycle via .launch/*.plist.
      This script is a one-shot validator only — it does NOT start or stop
      services, and exits cleanly after confirming everything is healthy.

Steps (fails fast on any error):
  1. Run preflight checks (validates tools defined in bgem3_mcp.py)
  2. Wait for bgem3_embed /health
  3. Wait for bgem3_rerank /health
  4. Wait for bgem3_mcp port open, list exposed MCP tools
  5. Run smoke tests (test_service.py)
  6. Report status and exit 0

Usage:
    uv run python start.py

Logs:
    logs/bgem3_embed.log
    logs/bgem3_rerank.log
    logs/bgem3_mcp.log
"""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────────────────────────
ZT_IP = "10.230.57.109"
RAG_PORT = 8000
MCP_PORT = 8001
RERANK_PORT = 8002
RAG_URL = f"http://{ZT_IP}:{RAG_PORT}"
RERANK_URL = f"http://{ZT_IP}:{RERANK_PORT}"
MCP_URL = f"http://{ZT_IP}:{MCP_PORT}"
HEALTH_TIMEOUT = 60  # seconds (model load ~30s each)
HEALTH_POLL = 2
LOG_DIR = "logs"

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────


def info(msg: str) -> None:
    print(f"{BOLD}[INFO]{RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")
    sys.exit(1)


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def wait_for_health(url: str, timeout: int, poll: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def wait_for_port(host: str, port: int, timeout: int, poll: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(host, port):
            return True
        time.sleep(poll)
    return False


def _mcp_init(url: str) -> str | None:
    """Initialize MCP session and return session ID from header."""
    payload = {
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "start.py", "version": "1.0"},
        },
    }
    try:
        r = httpx.post(
            f"{url}/mcp/",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            content=json.dumps(payload),
            timeout=10,
        )
        return r.headers.get("mcp-session-id")
    except Exception:
        return None


def list_mcp_tools(url: str) -> list[str]:
    """Query the MCP server for its tool list via JSON-RPC tools/list."""
    session_id = _mcp_init(url)
    if not session_id:
        return []

    payload = {
        "jsonrpc": "2.0",
        "id": "preflight",
        "method": "tools/list",
        "params": {},
    }
    try:
        r = httpx.post(
            f"{url}/mcp/",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id,
            },
            content=json.dumps(payload),
            timeout=10,
        )
        text = r.text.strip()
        if text.startswith("data:"):
            for line in text.splitlines():
                if line.startswith("data:"):
                    text = line[len("data:"):].strip()
                    break
        data = json.loads(text)
        tools = data.get("result", {}).get("tools", [])
        return [t["name"] for t in tools]
    except Exception:
        return []


# ── Main ──────────────────────────────────────────────────────────────────────────────────────────

print(f"\n{BOLD}=== BGE-M3 + Reranker + MCP Startup Validator ==={RESET}\n")
os.makedirs(LOG_DIR, exist_ok=True)

# Step 1: Preflight
info("Step 1/5: Running preflight checks...")
result = subprocess.run([sys.executable, "preflight.py"])
if result.returncode != 0:
    fail("Preflight failed. Fix issues above before starting.")
ok("Preflight passed")

# Step 2: bgem3_embed health
info(f"Step 2/5: Waiting for bgem3_embed /health at {RAG_URL} ...")
if not wait_for_health(RAG_URL, HEALTH_TIMEOUT, HEALTH_POLL):
    fail(f"bgem3_embed not healthy after {HEALTH_TIMEOUT}s. Check logs/bgem3_embed.log")
ok(f"bgem3_embed healthy → {RAG_URL}")

# Step 3: bgem3_rerank health
info(f"Step 3/5: Waiting for bgem3_rerank /health at {RERANK_URL} ...")
if not wait_for_health(RERANK_URL, HEALTH_TIMEOUT, HEALTH_POLL):
    fail(f"bgem3_rerank not healthy after {HEALTH_TIMEOUT}s. Check logs/bgem3_rerank.log")
ok(f"bgem3_rerank healthy → {RERANK_URL}")

# Step 4: bgem3_mcp port + tool list
info(f"Step 4/5: Waiting for bgem3_mcp port {MCP_PORT} ...")
if not wait_for_port(ZT_IP, MCP_PORT, timeout=20, poll=1):
    fail(f"bgem3_mcp did not open port {MCP_PORT} within 20s. Check logs/bgem3_mcp.log")
ok(f"bgem3_mcp live → {MCP_URL}")

time.sleep(2)  # brief settle
tools = list_mcp_tools(MCP_URL)
if tools:
    ok(f"MCP tools exposed: {', '.join(tools)}")
else:
    warn("Could not retrieve MCP tool list — server may still be initialising")

# Step 5: Smoke tests
info("Step 5/5: Running smoke tests...")
result = subprocess.run([sys.executable, "test_service.py"])
if result.returncode != 0:
    warn("Smoke tests failed — services running but something is wrong. Check logs/.")
    sys.exit(1)
ok("All smoke tests passed")

# Done — exit cleanly; launchctl owns the processes
print(f"\n{GREEN}{BOLD}=== All services up and healthy ==={RESET}")
print(f"  bgem3_embed   → {RAG_URL}    log: logs/bgem3_embed.log")
print(f"  bgem3_rerank  → {RERANK_URL}  log: logs/bgem3_rerank.log")
print(f"  bgem3_mcp     → {MCP_URL}    log: logs/bgem3_mcp.log")
if tools:
    print(f"  MCP tools     : {', '.join(tools)}")
print()
sys.exit(0)
