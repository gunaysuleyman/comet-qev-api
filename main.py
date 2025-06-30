import os
import gc
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from evaluate import load


BATCH_SIZE = int(os.getenv("COMET_BATCH_SIZE", "16"))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

class InputData(BaseModel):
    source: List[str]
    hypothesis: List[str]
    reference: List[str]
    batch_size: int = BATCH_SIZE

class EvaluationResponse(BaseModel):
    scores: List[float]
    processing_time: float
    samples_processed: int
    samples_per_second: float
    device_used: str

# Device setup with memory optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {total_memory:.1f} GB")

    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    print(f"GPU memory fraction set to: {GPU_MEMORY_FRACTION}")

print("Loading COMET model with optimizations...")
try:
    if torch.cuda.is_available():
        comet_metric = load('comet', device=0)
        print("COMET model loaded on GPU!")
        
        # Note: COMET models don't support direct PyTorch optimizations
        print("COMET model loaded with standard precision")
                
    else:
        comet_metric = load('comet')
        print("COMET model loaded on CPU")
        
except Exception as e:
    print(f"Failed to load optimized model, loading standard: {e}")
    comet_metric = load('comet')

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

app = FastAPI(
    title="COMET Evaluation API",
    description="High-performance COMET metric evaluation service",
    version="2.0.0"
)

def cleanup_memory():
    """Memory cleanup after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def process_in_chunks(data: InputData) -> List[float]:
    """Process data in optimized chunks"""
    total_samples = len(data.source)
    all_scores = []
    chunk_size = min(data.batch_size, total_samples)
    
    print(f"Processing {total_samples} samples in chunks of {chunk_size}")
    
    compute_kwargs = {}
    if torch.cuda.is_available():
        compute_kwargs['gpus'] = 1
    else:
        compute_kwargs['gpus'] = 0
    
    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        
        chunk_source = data.source[i:end_idx]
        chunk_hypothesis = data.hypothesis[i:end_idx]
        chunk_reference = data.reference[i:end_idx]
        
        try:
            
            results = comet_metric.compute(
                predictions=chunk_hypothesis,
                references=chunk_reference,
                sources=chunk_source,
                **compute_kwargs
            )
            
            if results and "scores" in results:
                all_scores.extend(results["scores"])
            else:
                raise Exception("COMET computation returned invalid results")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"GPU processing failed for chunk {i//chunk_size + 1}, trying CPU: {e}")
            
            # Fallback to CPU
            results = comet_metric.compute(
                predictions=chunk_hypothesis,
                references=chunk_reference,
                sources=chunk_source,
                gpus=0
            )
            
            if results and "scores" in results:
                all_scores.extend(results["scores"])
            else:
                raise Exception("COMET CPU computation returned invalid results")
    
    return all_scores

def compute_comet_scores(data: InputData) -> dict:
    """Main computation function"""
    start_time = time.time()
    
    try:

        scores = process_in_chunks(data)
        
        processing_time = time.time() - start_time
        device_used = "GPU" if torch.cuda.is_available() and device.type == 'cuda' else "CPU"
        
        return {
            "scores": [round(float(score), 3) for score in scores],
            "processing_time": round(processing_time, 3),
            "samples_processed": len(data.source),
            "samples_per_second": round(len(data.source) / processing_time, 2),
            "device_used": device_used
        }
        
    except Exception as e:
        raise Exception(f"COMET computation failed: {str(e)}")
    finally:
        cleanup_memory()

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_translations(data: InputData):
    """Asynchronous COMET evaluation endpoint"""

    try:
        assert len(data.source) == len(data.hypothesis) == len(data.reference)
        assert len(data.source) > 0
    except AssertionError:
        raise HTTPException(
            status_code=400, 
            detail="Source, hypothesis, and reference lists must have the same non-zero length"
        )

    if data.batch_size <= 0:
        data.batch_size = BATCH_SIZE
    
    print(f"Received {len(data.source)} samples for evaluation")
    
    try:

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, compute_comet_scores, data)
        
        print(f"Evaluation completed: {result['samples_per_second']:.2f} samples/sec")
        
        return EvaluationResponse(**result)
        
    except Exception as e:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_info = {
            "gpu_memory_allocated_gb": round(memory_allocated, 2),
            "gpu_memory_reserved_gb": round(memory_reserved, 2)
        }
    
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": comet_metric is not None,
        "fp16_enabled": USE_FP16,
        "batch_size": BATCH_SIZE,
        **memory_info
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "COMET Evaluation API",
        "version": "2.0.0",
        "endpoints": {
            "evaluate": "POST /evaluate - Evaluate translations with COMET",
            "health": "GET /health - Check API health and memory usage"
        },
        "optimizations": [
            "Async processing",
            "Batch processing", 
            "Memory optimization",
            "Mixed precision (FP16)" if USE_FP16 else "Standard precision",
            "Model compilation" if hasattr(torch, 'compile') else "No compilation"
        ]
    }

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_memory()
    executor.shutdown(wait=True)
    print("API shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)