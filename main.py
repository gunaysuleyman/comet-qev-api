from fastapi import HTTPException

from evaluate import load
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import torch

class InputData(BaseModel):
    source: List[str]
    hypothesis: List[str]
    reference: List[str]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("Loading COMET model...")
try:
    if torch.cuda.is_available():
        comet_metric = load('comet', device=0)
        print("COMET model loaded directly on GPU!")
    else:
        comet_metric = load('comet')
        print("COMET model loaded on CPU")
except Exception as e:
    print(f"Failed to load on GPU, loading on CPU: {e}")
    comet_metric = load('comet')
    print("COMET model loaded on CPU (fallback)")

app = FastAPI()


@app.post("/evaluate", response_model=List[float])
def process_items(data: InputData):

    try:
        assert len(data.source) == len(data.hypothesis) == len(data.reference)
    except AssertionError:
        raise HTTPException(status_code=400, detail="The three groups (source, hypothesis and reference) must have the same number of segments.")

    print(f"Processing {len(data.source)} samples on {device}")
    
    compute_kwargs = {}
    if torch.cuda.is_available():
        compute_kwargs['gpus'] = 1 if device.type == 'cuda' else 0
    
    try:
        results = comet_metric.compute(
            predictions=data.hypothesis,
            references=data.reference,
            sources=data.source,
            **compute_kwargs
        )
        print(f"Evaluation completed successfully on {device}")
    except Exception as e:
        print(f"GPU evaluation failed, retrying on CPU: {e}")

        results = comet_metric.compute(
            predictions=data.hypothesis,
            references=data.reference,
            sources=data.source,
            gpus=0
        )
        print("Evaluation completed on CPU fallback")
    
    return [round(v, 3) for v in results["scores"]]