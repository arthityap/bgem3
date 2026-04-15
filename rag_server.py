import os

import logging
import sys
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import torch
import time
import psutil

# Configure logging
# Custom logging handler removed to prevent recursion
# Was causing maximum recursion depth exceeded error


app = FastAPI(title="BGEM3 Embedding Service", version="1.0.0")

# Initialize model with error handling
try:
    print("Initializing BAAI/bge-m3 model on MPS device...")
    model = SentenceTransformer('BAAI/bge-m3', device='mps')
    print(f"Model initialized successfully on {model._target_device}")
except Exception as e:
    print(f"Failed to initialize model: {str(e)}", file=sys.stderr)
    raise

API_KEY = os.getenv('API_KEY', 'm1macmini')

def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"), 
    api_key: str = None
) -> str:
    """Verify API key from either header or parameter
    
    Args:
        x_api_key: API key from X-API-Key header
        api_key: API key from query parameter or request body
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: 401 for unauthorized access
        
    Tags:
        - authentication
        - security
    """
    key = api_key or x_api_key
    
    if not key:
        raise HTTPException(
            status_code=401, 
            detail="API Key is required in X-API-Key header or api_key parameter"
        )
        
    if key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API Key"
        )
    return key

@app.on_event("startup")
async def warmup():
    try:
        print("Warming up model...")
        model.encode(["warmup"], batch_size=1)
        print("✅ Server warmed up")
        print(f"Model device: {model._target_device}")
        print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Log startup metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"Startup metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%")
        
    except Exception as e:
        print(f"Failed to warm up model: {str(e)}", file=sys.stderr)
        raise

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint with system metrics"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy", 
        "service": "bgem3",
        "timestamp": int(time.time()),
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available": memory.available
    }

class HealthCheckResponse(Dict[str, Any]):
    status: str
    service: str
    timestamp: int
    cpu_percent: float
    memory_percent: float
    memory_available: int

@app.get("/info")
def info() -> Dict[str, Any]:
    """Comprehensive service information endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved()
        }
    elif str(model._target_device) == "mps":
        gpu_info = {
            "mps_available": True,
            "device": "Apple Silicon GPU"
        }
    
    return {
        "model": "BAAI/bge-m3",
        "dimensions": model.get_sentence_embedding_dimension(),
        "device": str(model._target_device),
        "batch_size": 4,
        "framework": "sentence-transformers",
        "framework_version": torch.__version__,
        "gpu_info": gpu_info,
        "torch_version": torch.__version__,
        "sentence_transformers_version": "5.4.1"
    }

class InfoResponse(Dict[str, Any]):
    model: str
    dimensions: int
    device: str
    batch_size: int
    framework: str
    framework_version: str
    gpu_info: Dict[str, Any]
    torch_version: str
    sentence_transformers_version: str

@app.post('/embed')
def embed(texts: list[str], api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Generate embeddings for input texts with comprehensive error handling
    
    Args:
        texts: List of strings to generate embeddings for
        api_key: Authentication key (passed automatically via header)
        
    Returns:
        Dictionary containing embeddings and metadata
        
    Raises:
        HTTPException: 422 for validation errors, 500 for server errors
        
    Tags:
        - embeddings
        - inference
        - bge-m3
    """
    try:
        if not texts:
            raise HTTPException(
                status_code=422, 
                detail="Text list cannot be empty"
            )
        
        if len(texts) > 100:
            raise HTTPException(
                status_code=429, 
                detail="Too many texts in single request. Maximum is 100 texts."
            )
            
        # Log request size for monitoring
        print(f"Processing {len(texts)} texts")
        
        # Generate embeddings
        embeddings = model.encode(texts, batch_size=4).tolist()
        
        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "model": "BAAI/bge-m3",
            "success": True
        }
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate embeddings: {str(e)}"
        )
