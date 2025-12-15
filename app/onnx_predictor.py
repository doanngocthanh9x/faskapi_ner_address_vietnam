from typing import List, Dict
import os
import logging
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import MODEL_PATH, TOKENIZER_NAME, MAX_SEQUENCE_LENGTH, LABEL_MAP, ONNX_PROVIDERS
except ImportError:
    # Fallback defaults if config is not available
    MODEL_PATH = "models/model.onnx"
    TOKENIZER_NAME = "vinai/phobert-base"
    MAX_SEQUENCE_LENGTH = 256
    LABEL_MAP = {
        0: "O", 1: "B-CITY", 2: "I-CITY", 3: "B-DISTRICT",
        4: "I-DISTRICT", 5: "B-WARD", 6: "I-WARD", 7: "B-STREET", 8: "I-STREET"
    }
    ONNX_PROVIDERS = ["CPUExecutionProvider"]

try:
    import onnxruntime as ort
    import numpy as np
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    logging.warning(f"ONNX Runtime or dependencies not available: {e}")
    logging.info("Install with: pip install onnxruntime numpy transformers")

logger = logging.getLogger(__name__)


class ONNXPredictor:
    """ONNX-based predictor for Vietnamese address NER"""
    
    def __init__(self, model_path: str = None, tokenizer_name: str = None):
        """
        Initialize ONNX predictor
        
        Args:
            model_path: Path to the ONNX model file
            tokenizer_name: Name or path of the tokenizer
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime and dependencies not available. "
                "Install with: pip install onnxruntime numpy transformers"
            )
        
        # Use config defaults if not provided
        self.model_path = model_path or MODEL_PATH
        tokenizer_name = tokenizer_name or TOKENIZER_NAME
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}")
            logger.info("To use this API, you need to:")
            logger.info("1. Train a NER model")
            logger.info("2. Convert it to ONNX format")
            logger.info("3. Place it at models/model.onnx")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            self.model_path,
            providers=ONNX_PROVIDERS
        )
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            logger.info("Using a fallback simple tokenizer")
            self.tokenizer = None
        
        # Use label mapping from config
        self.id2label = LABEL_MAP
        self.max_length = MAX_SEQUENCE_LENGTH
        
        logger.info(f"ONNX model loaded from {self.model_path}")
        logger.info(f"Model inputs: {[i.name for i in self.session.get_inputs()]}")
        logger.info(f"Model outputs: {[o.name for o in self.session.get_outputs()]}")
    
    def predict(self, text: str) -> List[Dict[str, str]]:
        """
        Predict named entities in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of entities with labels and text
        """
        if not text or not text.strip():
            return []
        
        # Tokenize input
        if self.tokenizer:
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Prepare input for ONNX
            ort_inputs = {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64),
            }
            
            # Add token_type_ids if required by model
            input_names = [i.name for i in self.session.get_inputs()]
            if 'token_type_ids' in input_names and 'token_type_ids' in inputs:
                ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)
        else:
            # Fallback: No tokenizer available - this is not ideal for production
            logger.warning("No tokenizer available - predictions will be inaccurate")
            raise RuntimeError(
                "Tokenizer not available. Please install transformers and ensure "
                "the tokenizer can be loaded from Hugging Face Hub or local path."
            )
        
        # Run inference
        outputs = self.session.run(None, ort_inputs)
        logits = outputs[0]
        
        # Get predictions
        predictions = np.argmax(logits, axis=-1)[0]
        
        # Convert to entities
        entities = self._decode_predictions(text, predictions)
        
        return entities
    
    def _decode_predictions(self, text: str, predictions) -> List[Dict[str, str]]:
        """
        Decode predictions to entity list
        
        Args:
            text: Original text
            predictions: Predicted label IDs
            
        Returns:
            List of entities
        """
        entities = []
        words = text.split()
        
        current_entity = None
        current_text = []
        
        # Ensure we don't go beyond the length of words
        num_tokens = min(len(predictions), len(words))
        
        for i in range(num_tokens):
            pred_id = predictions[i]
            label = self.id2label.get(int(pred_id), "O")
            
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append({
                        "entity": current_entity,
                        "text": " ".join(current_text),
                        "start": i - len(current_text),
                        "end": i
                    })
                # Start new entity
                current_entity = label[2:]
                current_text = [words[i]]
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                current_text.append(words[i])
            else:
                # Save previous entity if exists
                if current_entity:
                    entities.append({
                        "entity": current_entity,
                        "text": " ".join(current_text),
                        "start": i - len(current_text),
                        "end": i
                    })
                current_entity = None
                current_text = []
        
        # Don't forget the last entity
        if current_entity:
            entities.append({
                "entity": current_entity,
                "text": " ".join(current_text),
                "start": num_tokens - len(current_text),
                "end": num_tokens
            })
        
        return entities
