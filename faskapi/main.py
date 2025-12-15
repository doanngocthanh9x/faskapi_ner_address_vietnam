"""
FastAPI application for NER Address Vietnam
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import config
from .schemas import NERRequest, NERResponse, HealthResponse, TokenLabel
from .ner_model import get_model, ONNXNERModel
import os
from typing import Dict


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model instance (lazy loaded)
model: ONNXNERModel = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        print("Loading NER model...")
        model = get_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model will be loaded on first request")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to NER Address Vietnam API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    global model
    model_loaded = model is not None
    
    if not model_loaded:
        # Check if model file exists
        model_exists = os.path.exists(config.MODEL_FILE)
        if not model_exists:
            return HealthResponse(
                status="warning",
                model_loaded=False,
                message="Model file not found. Please run download_model.py"
            )
        return HealthResponse(
            status="ok",
            model_loaded=False,
            message="Model file exists but not loaded yet"
        )
    
    return HealthResponse(
        status="ok",
        model_loaded=True,
        message="Service is healthy"
    )


@app.post("/predict", response_model=NERResponse, tags=["NER"])
async def predict(request: NERRequest):
    """
    Perform NER prediction on input text
    
    Args:
        request: NERRequest with text field
        
    Returns:
        NERResponse with tokens, labels, and extracted entities
    """
    global model
    
    # Lazy load model if not loaded yet
    if model is None:
        try:
            model = get_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    try:
        # Get predictions
        token_labels = model.predict(request.text)
        
        # Extract entities
        entities = model.extract_entities(request.text)
        
        # Format response
        tokens = [
            TokenLabel(token=token, label=label)
            for token, label in token_labels
        ]
        
        return NERResponse(
            text=request.text,
            tokens=tokens,
            entities=entities
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/extract", response_model=Dict, tags=["NER"])
async def extract_entities(request: NERRequest):
    """
    Extract only the entities from input text
    
    Args:
        request: NERRequest with text field
        
    Returns:
        Dictionary with extracted entities grouped by type
    """
    global model
    
    # Lazy load model if not loaded yet
    if model is None:
        try:
            model = get_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    try:
        # Extract entities
        entities = model.extract_entities(request.text)
        
        return {
            "text": request.text,
            "entities": entities
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
