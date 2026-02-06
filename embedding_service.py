#!/usr/bin/env python3
"""
Embedding Microservice
Serves PubMedBERT embeddings via FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from sentence_transformers import SentenceTransformer
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Protein Embedding Service", description="Microservice for PubMedBERT embeddings")

# Global model variable
model = None

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.on_event("startup")
async def load_model():
    global model
    print("🔄 Loading PubMedBERT model...")
    # Use the same model as in the original code
    model_name = 'neuml/pubmedbert-base-embeddings'
    model = SentenceTransformer(model_name)
    print("✅ Model loaded successfully!")

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Check if input is string or list
        sentences = request.text
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Generate embeddings
        embeddings = model.encode(sentences).tolist()
        
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    # Run server
    port = int(os.getenv("EMBEDDING_PORT", 8000))
    print(f"🚀 Starting Embedding Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
