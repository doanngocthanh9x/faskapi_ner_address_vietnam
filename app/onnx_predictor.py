from typing import List, Dict
import os
import logging

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
    
    def __init__(self, model_path: str = "models/model.onnx", 
                 tokenizer_name: str = "vinai/phobert-base"):
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
        
        self.model_path = model_path
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            logger.info("To use this API, you need to:")
            logger.info("1. Train a NER model")
            logger.info("2. Convert it to ONNX format")
            logger.info("3. Place it at models/model.onnx")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            logger.info("Using a fallback simple tokenizer")
            self.tokenizer = None
        
        # Define label mapping (adjust based on your dataset)
        self.id2label = {
            0: "O",
            1: "B-CITY",
            2: "I-CITY",
            3: "B-DISTRICT",
            4: "I-DISTRICT",
            5: "B-WARD",
            6: "I-WARD",
            7: "B-STREET",
            8: "I-STREET",
        }
        
        logger.info(f"ONNX model loaded from {model_path}")
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
                max_length=256
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
            # Fallback: simple tokenization
            words = text.split()
            # Create dummy inputs (this is a placeholder)
            ort_inputs = {
                'input_ids': np.array([[i for i in range(len(words))]], dtype=np.int64),
                'attention_mask': np.ones((1, len(words)), dtype=np.int64),
            }
        
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
        
        for i, pred_id in enumerate(predictions[:len(words)]):
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
                current_text = [words[i]] if i < len(words) else []
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                if i < len(words):
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
                "start": len(predictions) - len(current_text),
                "end": len(predictions)
            })
        
        return entities
