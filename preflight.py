"""
preflight.py — Run this before starting rag_server or mcp_server.

Checks:
  1. Python version >= 3.11
  2. Required packages importable (torch, FlagEmbedding, fastapi, fastmcp, anyio, psutil)
  3. MPS (Apple Silicon GPU) available
  4. .env file exists and EMBEDDING_API_KEY is set
  5. BGE-M3 model weights cached locally (no re-download needed)
  6. Ports 8000 and 8001 are free (not already in use)

Usage:
    uv run python preflight.py

All checks must pass (OK) before starting services.
FAIL = hard blocker. WARN = non-fatal but worth fixing.
"""

import importlib
import os
import socket
import sys

from dotenv import load_dotenv

load_dotenv()

PASS  = "\033[32m OK  \033[0m"
FAIL  = "\033[31m FAIL\033[0m"
WARN  = "\033[33m WARN\033[0m"

failures = 0


def check(label: str, ok: bool, msg: str = "", warn: bool = False) -> None:
    global failures
    if ok:
        print(f"[{PASS}] {label}")
    elif warn:
        print(f"[{WARN}] {label}: {msg}")
    else:
        print(f"[{FAIL}] {label}: {msg}")
        failures += 1


print("\n=== BGE-M3 Preflight Check ===\n")

# 1. Python version
pv = sys.version_info
check(
    f"Python >= 3.11  (found {pv.major}.{pv.minor}.{pv.micro})",
    pv >= (3, 11),
    "upgrade Python to 3.11+",
)

# 2. Required packages
REQUIRED_PACKAGES = [
    ("torch",        "torch"),
    ("FlagEmbedding", "FlagEmbedding"),
    ("fastapi",      "fastapi"),
    ("fastmcp",      "fastmcp"),
    ("anyio",        "anyio"),
    ("psutil",       "psutil"),
    ("httpx",        "httpx"),
    ("uvicorn",      "uvicorn"),
    ("dotenv",       "python-dotenv"),
]
for mod, pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(mod)
        check(f"Package: {pkg}", True)
    except ImportError:
        check(f"Package: {pkg}", False, f"run: uv add {pkg}")

# 3. MPS available
try:
    import torch
    mps_ok = torch.backends.mps.is_available()
    check("MPS (Apple Silicon GPU) available", mps_ok, "model will fall back to CPU (slow)", warn=not mps_ok)
except Exception as e:
    check("MPS check", False, str(e))

# 4. .env and EMBEDDING_API_KEY
env_exists = os.path.exists(".env")
check(".env file exists", env_exists, "create .env from .env.example")
api_key = os.getenv("EMBEDDING_API_KEY", "")
if api_key:
    check(f"EMBEDDING_API_KEY set (value: {api_key})", True)
else:
    check("EMBEDDING_API_KEY set", False, "set EMBEDDING_API_KEY in .env — auth will be disabled without it", warn=True)

# 5. BGE-M3 model weights cached
HF_HOME    = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
model_path = os.path.join(HF_HOME, "hub", "models--BAAI--bge-m3")
check(
    f"BGE-M3 weights cached ({model_path})",
    os.path.isdir(model_path),
    "first run will download ~2.3 GB from HuggingFace",
    warn=not os.path.isdir(model_path),
)

# 6. Ports free
for port, name in [(8000, "rag_server"), (8001, "mcp_server")]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        in_use = s.connect_ex(("127.0.0.1", port)) == 0
    check(
        f"Port {port} free ({name})",
        not in_use,
        f"port {port} already in use — kill existing process first",
        warn=in_use,
    )

# Summary
print()
if failures == 0:
    print("\033[32mAll checks passed. Ready to start services.\033[0m")
    print("  uv run uvicorn rag_server:app --host 0.0.0.0 --port 8000")
    print("  uv run python mcp_server.py")
else:
    print(f"\033[31m{failures} check(s) failed. Fix above issues before starting.\033[0m")
    sys.exit(1)
