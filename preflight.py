"""
preflight.py — Run this before starting rag_server, reranker_server, or mcp_server.

Checks:
  1. Python version == 3.11.x
  2. Required packages importable
  3. MPS (Apple Silicon GPU) available
  4. .env file exists and EMBEDDING_API_KEY is set
  5. BGE-M3 model weights cached locally
  6. BGE-reranker-v2-m3 weights cached locally
  7. Ports 8000, 8001, 8002 are free

Usage:
    uv run python preflight.py

FAIL = hard blocker. WARN = non-fatal but worth fixing.
"""

import importlib
import os
import socket
import sys

from dotenv import load_dotenv

load_dotenv()

PASS = "\033[32m OK  \033[0m"
FAIL = "\033[31m FAIL\033[0m"
WARN = "\033[33m WARN\033[0m"

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


print("\n=== BGE-M3 + Reranker Preflight Check ===\n")

# 1. Python version (must be exactly 3.11)
pv = sys.version_info
check(
    f"Python == 3.11.x  (found {pv.major}.{pv.minor}.{pv.micro})",
    pv >= (3, 11) and pv < (3, 12),
    "requires Python 3.11 (set .python-version to 3.11)",
)

# 2. Required packages
REQUIRED_PACKAGES = [
    ("torch",         "torch"),
    ("FlagEmbedding", "FlagEmbedding"),
    ("fastapi",       "fastapi"),
    ("fastmcp",       "fastmcp"),
    ("anyio",         "anyio"),
    ("psutil",        "psutil"),
    ("httpx",         "httpx"),
    ("uvicorn",       "uvicorn"),
    ("dotenv",        "python-dotenv"),
    ("pydantic",      "pydantic"),
]
for mod, pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(mod)
        check(f"Package: {pkg}", True)
    except ImportError:
        check(f"Package: {pkg}", False, f"run: uv add {pkg}")

# Verify FlagReranker is available inside FlagEmbedding
try:
    from FlagEmbedding import FlagReranker  # noqa: F401
    check("FlagReranker available in FlagEmbedding", True)
except ImportError:
    check("FlagReranker available in FlagEmbedding", False, "run: uv add FlagEmbedding --upgrade")

# 3. MPS available
try:
    import torch
    mps_ok = torch.backends.mps.is_available()
    check(
        "MPS (Apple Silicon GPU) available",
        mps_ok,
        "models will fall back to CPU (slow)",
        warn=not mps_ok,
    )
except Exception as e:
    check("MPS check", False, str(e))

# 4. .env and EMBEDDING_API_KEY
env_exists = os.path.exists(".env")
check(".env file exists", env_exists, "create .env from .env.example")
api_key = os.getenv("EMBEDDING_API_KEY", "")
if api_key:
    check(f"EMBEDDING_API_KEY set (value: {api_key})", True)
else:
    check(
        "EMBEDDING_API_KEY set",
        False,
        "set EMBEDDING_API_KEY in .env — auth will be disabled without it",
        warn=True,
    )

# 5. BGE-M3 embedder weights cached
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
bgem3_path = os.path.join(HF_HOME, "hub", "models--BAAI--bge-m3")
check(
    f"BGE-M3 weights cached ({bgem3_path})",
    os.path.isdir(bgem3_path),
    "first run will download ~2.3 GB from HuggingFace",
    warn=not os.path.isdir(bgem3_path),
)

# 6. BGE-reranker-v2-m3 weights cached
reranker_path = os.path.join(HF_HOME, "hub", "models--BAAI--bge-reranker-v2-m3")
check(
    f"bge-reranker-v2-m3 weights cached ({reranker_path})",
    os.path.isdir(reranker_path),
    "first run will download ~1.1 GB from HuggingFace",
    warn=not os.path.isdir(reranker_path),
)

# 7. Ports free
for port, name in [(8000, "rag_server"), (8001, "mcp_server"), (8002, "reranker_server")]:
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
    print("  uv run python start.py")
else:
    print(f"\033[31m{failures} check(s) failed. Fix above issues before starting.\033[0m")
    sys.exit(1)
