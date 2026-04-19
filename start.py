"""
start.py — Full lifecycle manager for rag_server (8000) and mcp_server (8001).

Steps (fails fast on any error):
  1. Kill any existing process on ports 8000 and 8001
  2. Kill any stale rag_server.py / mcp_server.py / uvicorn processes by name
  3. Run preflight checks (preflight.py)
  4. Start rag_server on port 8000, wait until /health responds
  5. Start mcp_server on port 8001, wait until port is open
  6. Run smoke tests (test_service.py)
  7. Report final status

Usage:
    uv run python start.py

Logs from both services are written to:
    logs/rag_server.log
    logs/mcp_server.log
"""

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
RAG_URL = f"http://{ZT_IP}:{RAG_PORT}"
HEALTH_TIMEOUT = 60  # seconds to wait for /health to respond
HEALTH_POLL = 2  # seconds between polls
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
    """Return PIDs listening on the given port (macOS/Linux via lsof)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"TCP:{port}", "-s", "TCP:LISTEN"],
            text=True,
        ).strip()
        return [int(p) for p in out.split() if p.isdigit()]
    except subprocess.CalledProcessError:
        return []


def kill_port(port: int) -> None:
    """Kill all processes listening on port."""
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
    # SIGKILL any survivors
    for pid in port_pids(port):
        try:
            os.kill(pid, signal.SIGKILL)
            warn(f"Port {port}: force-killed PID {pid}")
        except ProcessLookupError:
            pass


def kill_by_name(*names: str) -> None:
    """Kill processes whose command line contains any of the given names."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "|".join(names)], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return  # no matches
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
    """Poll GET /health until 200 or timeout."""
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


# ── Main ────────────────────────────────────────────────────────────────────────────────

print(f"\n{BOLD}=== BGE-M3 Service Startup ==={RESET}\n")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Step 1 & 2: Kill stale processes ────────────────────────────────────────────
info("Step 1/4: Killing any processes on ports 8000 and 8001...")
kill_port(RAG_PORT)
kill_port(MCP_PORT)
kill_by_name("rag_server", "mcp_server", "uvicorn")
time.sleep(1)
ok("Stale processes cleared")

# ── Step 2: Preflight ───────────────────────────────────────────────────────────────
info("Step 2/4: Running preflight checks...")
result = subprocess.run([sys.executable, "preflight.py"])
if result.returncode != 0:
    fail("Preflight failed. Fix issues above before starting.")
ok("Preflight passed")

# ── Step 3: Start rag_server ─────────────────────────────────────────────────────────
info(f"Step 3/4: Starting rag_server on port {RAG_PORT}...")
rag_log = open(os.path.join(LOG_DIR, "rag_server.log"), "w")
rag_proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "rag_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(RAG_PORT),
        "--log-level",
        "info",
    ],
    stdout=rag_log,
    stderr=rag_log,
    start_new_session=True,
)
info(
    f"rag_server PID {rag_proc.pid} — waiting for /health (up to {HEALTH_TIMEOUT}s, model load takes ~30s)..."
)
if not wait_for_health(RAG_URL, HEALTH_TIMEOUT, HEALTH_POLL):
    rag_proc.terminate()
    fail(
        f"rag_server did not become healthy within {HEALTH_TIMEOUT}s. Check logs/rag_server.log"
    )
ok(f"rag_server healthy on {RAG_URL}")

# ── Step 4: Start mcp_server ─────────────────────────────────────────────────────────
info(f"Step 4/4: Starting mcp_server on port {MCP_PORT}...")
mcp_log = open(os.path.join(LOG_DIR, "mcp_server.log"), "w")
mcp_proc = subprocess.Popen(
    [sys.executable, "mcp_server.py"],
    stdout=mcp_log,
    stderr=mcp_log,
    start_new_session=True,
)
info(f"mcp_server PID {mcp_proc.pid} — waiting for port {MCP_PORT}...")
if not wait_for_port(ZT_IP, MCP_PORT, timeout=20, poll=1):
    mcp_proc.terminate()
    rag_proc.terminate()
    fail(
        f"mcp_server did not open port {MCP_PORT} within 20s. Check logs/mcp_server.log"
    )
ok(f"mcp_server live on {ZT_IP}:{MCP_PORT}")

# ── Done ──────────────────────────────────────────────────────────────────────────────────
print(f"\n{GREEN}{BOLD}=== All services up and healthy ==={RESET}")
print(f"  rag_server  → {RAG_URL}          (PID {rag_proc.pid})")
print(f"  mcp_server  → http://{ZT_IP}:{MCP_PORT}  (PID {mcp_proc.pid})")
print(f"  Logs        → {LOG_DIR}/rag_server.log, {LOG_DIR}/mcp_server.log")
print(f"\n{GREEN}Services are running in the background.{RESET}\n")

# Detach processes so they survive terminal close
rag_log.close()
mcp_log.close()
