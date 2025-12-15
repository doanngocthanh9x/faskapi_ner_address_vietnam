from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging

from app.onnx_predictor import ONNXPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vietnamese Address NER API",
    description="Named Entity Recognition for Vietnamese addresses using ONNX Runtime",
    version="1.0.0"
)

# Initialize predictor
predictor = None


class TextInput(BaseModel):
    text: str


class NERResult(BaseModel):
    text: str
    entities: List[Dict[str, str]]


@app.on_event("startup")
async def startup_event():
    """Initialize the ONNX model on startup"""
    global predictor
    try:
        predictor = ONNXPredictor()
        logger.info("ONNX model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        logger.warning("Starting without model - endpoints will return errors")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vietnamese Address NER API",
        "status": "running",
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=NERResult)
async def predict(input_data: TextInput):
    """
    Predict named entities in Vietnamese address text
    
    Args:
        input_data: Input text to analyze
        
    Returns:
        NERResult with extracted entities
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        entities = predictor.predict(input_data.text)
        return NERResult(text=input_data.text, entities=entities)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(texts: List[TextInput]):
    """
    Predict named entities for multiple texts
    
    Args:
        texts: List of input texts to analyze
        
    Returns:
        List of NERResult with extracted entities
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        results = []
        for text_input in texts:
            entities = predictor.predict(text_input.text)
            results.append(NERResult(text=text_input.text, entities=entities))
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
