"""
start.py — Full lifecycle manager for all three services.

  bgem3_embed    port 8000  (BGE-M3 embeddings)
  bgem3_rerank   port 8002  (bge-reranker-v2-m3 cross-encoder)
  bgem3_mcp      port 8001  (FastMCP — exposes embed, embed_hybrid, rerank)

Steps (fails fast on any error):
  1. Kill stale processes on ports 8000, 8001, 8002
  2. Run preflight checks (validates tools defined in bgem3_mcp.py)
  3. Start bgem3_embed, wait for /health
  4. Start bgem3_rerank, wait for /health
  5. Start bgem3_mcp, wait for port open, list exposed MCP tools
  6. Run smoke tests (test_service.py)
  7. Report PIDs + log paths, stay alive (Ctrl+C stops all)

Usage:
    uv run python start.py

Logs:
    logs/bgem3_embed.log
    logs/bgem3_rerank.log
    logs/bgem3_mcp.log
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────────────────
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


# ── Helpers ─────────────────────────────────────────────────────────────────────────────


def info(msg: str) -> None:
    print(f"{BOLD}[INFO]{RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")
    sys.exit(1)


def port_pids(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"TCP:{port}", "-s", "TCP:LISTEN"], text=True
        ).strip()
        return [int(p) for p in out.split() if p.isdigit()]
    except subprocess.CalledProcessError:
        return []


def kill_port(port: int) -> None:
    pids = port_pids(port)
    if not pids:
        info(f"Port {port}: nothing to kill")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            info(f"Port {port}: sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            pass
    time.sleep(1)
    for pid in port_pids(port):
        try:
            os.kill(pid, signal.SIGKILL)
            warn(f"Port {port}: force-killed PID {pid}")
        except ProcessLookupError:
            pass


def kill_by_name(*names: str) -> None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "|".join(names)], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return
    own_pid = os.getpid()
    for pid in (int(p) for p in out.split() if p.isdigit()):
        if pid == own_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            info(f"Killed stale process PID {pid}")
        except ProcessLookupError:
            pass


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


def start_uvicorn(module: str, port: int, log_path: str) -> subprocess.Popen:
    log_file = open(log_path, "w")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module}:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )


def list_mcp_tools(url: str) -> list[str]:
    """Query the MCP server for its tool list via JSON-RPC tools/list."""
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
            },
            content=json.dumps(payload),
            timeout=10,
        )
        text = r.text.strip()
        # Handle SSE response
        if text.startswith("data:"):
            for line in text.splitlines():
                if line.startswith("data:"):
                    text = line[len("data:") :].strip()
                    break
        data = json.loads(text)
        tools = data.get("result", {}).get("tools", [])
        return [t["name"] for t in tools]
    except Exception:
        return []


# ── Main ────────────────────────────────────────────────────────────────────────────────

print(f"\n{BOLD}=== BGE-M3 + Reranker + MCP Startup ==={RESET}\n")
os.makedirs(LOG_DIR, exist_ok=True)

# Step 1: Kill stale
info("Step 1/6: Killing stale processes on ports 8000, 8001, 8002...")
for port in (RAG_PORT, MCP_PORT, RERANK_PORT):
    kill_port(port)
kill_by_name("bgem3_embed", "bgem3_rerank", "bgem3_mcp", "uvicorn")
time.sleep(1)
ok("Stale processes cleared")

# Step 2: Preflight
info("Step 2/6: Running preflight checks...")
result = subprocess.run([sys.executable, "preflight.py"])
if result.returncode != 0:
    fail("Preflight failed. Fix issues above before starting.")
ok("Preflight passed")

# Step 3: bgem3_embed
info(f"Step 3/6: Starting bgem3_embed on port {RAG_PORT}...")
rag_proc = start_uvicorn(
    "bgem3_embed", RAG_PORT, os.path.join(LOG_DIR, "bgem3_embed.log")
)
info(f"bgem3_embed PID {rag_proc.pid} — waiting for /health (model load ~30s)...")
if not wait_for_health(RAG_URL, HEALTH_TIMEOUT, HEALTH_POLL):
    rag_proc.terminate()
    fail(f"bgem3_embed not healthy after {HEALTH_TIMEOUT}s. Check logs/bgem3_embed.log")
ok(f"bgem3_embed healthy → {RAG_URL}")

# Step 4: bgem3_rerank
info(f"Step 4/6: Starting bgem3_rerank on port {RERANK_PORT}...")
rerank_proc = start_uvicorn(
    "bgem3_rerank", RERANK_PORT, os.path.join(LOG_DIR, "bgem3_rerank.log")
)
info(f"bgem3_rerank PID {rerank_proc.pid} — waiting for /health (model load ~20s)...")
if not wait_for_health(RERANK_URL, HEALTH_TIMEOUT, HEALTH_POLL):
    rerank_proc.terminate()
    rag_proc.terminate()
    fail(
        f"bgem3_rerank not healthy after {HEALTH_TIMEOUT}s. Check logs/bgem3_rerank.log"
    )
ok(f"bgem3_rerank healthy → {RERANK_URL}")

# Step 5: bgem3_mcp
info(f"Step 5/6: Starting bgem3_mcp on port {MCP_PORT}...")
mcp_log = open(os.path.join(LOG_DIR, "bgem3_mcp.log"), "w")
mcp_proc = subprocess.Popen(
    [sys.executable, "bgem3_mcp.py"],
    stdout=mcp_log,
    stderr=mcp_log,
    start_new_session=True,
)
info(f"bgem3_mcp PID {mcp_proc.pid} — waiting for port {MCP_PORT}...")
if not wait_for_port(ZT_IP, MCP_PORT, timeout=20, poll=1):
    mcp_proc.terminate()
    rerank_proc.terminate()
    rag_proc.terminate()
    fail(f"bgem3_mcp did not open port {MCP_PORT} within 20s. Check logs/bgem3_mcp.log")
ok(f"bgem3_mcp live → {MCP_URL}")

# List exposed MCP tools
time.sleep(2)  # brief settle
tools = list_mcp_tools(MCP_URL)
if tools:
    ok(f"MCP tools exposed: {', '.join(tools)}")
else:
    warn("Could not retrieve MCP tool list — server may still be initialising")

# Step 6: Smoke tests
info("Step 6/6: Running smoke tests...")
result = subprocess.run([sys.executable, "test_service.py"])
if result.returncode != 0:
    warn("Smoke tests failed — services running but something is wrong. Check logs/.")
    sys.exit(1)
ok("All smoke tests passed")

# Done
print(f"\n{GREEN}{BOLD}=== All services up and healthy ==={RESET}")
print(
    f"  bgem3_embed       → {RAG_URL}    (PID {rag_proc.pid})  log: logs/bgem3_embed.log"
)
print(
    f"  bgem3_rerank  → {RERANK_URL} (PID {rerank_proc.pid})  log: logs/bgem3_rerank.log"
)
print(f"  bgem3_mcp       → {MCP_URL}    (PID {mcp_proc.pid})  log: logs/bgem3_mcp.log")
if tools:
    print(f"  MCP tools        : {', '.join(tools)}")
print(f"\n{GREEN}Press Ctrl+C to stop all services.{RESET}\n")

# Keep alive, forward Ctrl+C
try:
    rag_proc.wait()
except KeyboardInterrupt:
    info("Shutting down all services...")
    for proc in (rag_proc, rerank_proc, mcp_proc):
        proc.terminate()
    for proc in (rag_proc, rerank_proc, mcp_proc):
        proc.wait()
    ok("All services stopped cleanly.")
